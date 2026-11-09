# voice/agent.py   →   cd api && uv run --env-file ../.env python -m voice.agent console   (talk in the terminal)
#                      cd api && uv run --env-file ../.env python -m voice.agent dev       (real calls via LiveKit)
# v9: the voice worker. Joins every call, and connects   👂 STT → 🧠 your graph → 👄 TTS.

import os

from langchain_core.messages import AIMessageChunk
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext
from livekit.plugins import langchain, openai, silero

from app import db
from app.graph import build_graph
from app.tools import UserContext

db.init_tables()                                         # make sure the v9 tables exist
server = AgentServer()

# DEMO SHORTCUT (not in the guide): calls from these numbers count as verified, no SMS code.
# Only safe for testing — caller ID can be faked. In .env: TRUSTED_CALLER_NUMBERS=+15551234567
TRUSTED_NUMBERS = {n.strip() for n in os.environ.get("TRUSTED_CALLER_NUMBERS", "").split(",") if n.strip()}


class SpokenOnly:
    """Sits between LiveKit and the graph, and passes on ONLY text that should be spoken:
    - "messages": typed-out answer text (AIMessageChunk), same rule as visible() in main.py
    - "custom":   finished text from get_stream_writer() (checked policy answers, fixed replies)
    Why it's needed (not in the guide): with subgraphs=True LangGraph sends 3 parts
    (namespace, mode, data), but LiveKit's adapter only understands 2 (mode, data), so it would
    stay silent. It would also speak fixed replies twice (once as a message, once as custom)."""

    def __init__(self, graph):
        self.graph = graph

    async def astream(self, *args, **kwargs):
        async for _ns, mode, data in self.graph.astream(*args, **kwargs):
            if mode == "custom":
                yield ("custom", data)
            elif mode == "messages" and isinstance(data[0], AIMessageChunk) and data[0].text:
                yield ("messages", data)


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    await ctx.connect()                                  # join the room (the call)

    if ctx.is_fake_job():
        # console mode: there is no real caller, so pretend to be Alice calling from the browser
        context = UserContext(user_id="alice", role="customer", channel="web_voice", call_id="console")
    else:
        caller = await ctx.wait_for_participant()        # wait until the caller is in the room
        phone = caller.attributes.get("sip.phoneNumber")    # present when the call came through Twilio
        if phone:
            user = db.fetch_user_by_phone(phone)             # caller ID is only a hint: tools ask for an SMS code
            context = UserContext(user_id=user["id"] if user else "unknown", role="customer",
                                  channel="phone", call_id=ctx.room.name)
            if user and phone in TRUSTED_NUMBERS:
                db.mark_call_verified(ctx.room.name)     # demo shortcut: skip the SMS code for your own number
        else:
            user = db.fetch_user(caller.identity)            # browser call: identity from /voice/token
            context = UserContext(user_id=user["id"], role=user["role"],
                                  channel="web_voice", call_id=ctx.room.name)

    agent = Agent(
        instructions="",                                 # prompts live in the graph
        llm=langchain.LLMAdapter(
            graph=SpokenOnly(build_graph(checkpointer=None, channel="voice")),   # LiveKit sends the history each turn
            context=context,
            stream_mode=["messages", "custom"],          # tokens + checked answers from get_stream_writer
            subgraphs=True,                              # speak tokens from the specialist agents
            config={                                     # same labels as /chat, so LangSmith shows channel: phone / web_voice
                "recursion_limit": 25,
                "metadata": {"user_role": context.role, "channel": context.channel, "app_version": "v9"},
                "tags": ["voice"],
            },
        ),
    )
    session = AgentSession(
        stt=openai.STT(model="gpt-4o-mini-transcribe"),       # 👂 bills to your OpenAI key
        tts=openai.TTS(model="gpt-4o-mini-tts", voice="ash",
                       instructions="Warm, calm customer-support tone."),   # 👄
        vad=silero.VAD.load(),                                # detects when the caller is speaking
    )
    await session.start(agent=agent, room=ctx.room)
    await session.say("Hi, this is NovaMart support. How can I help?")


if __name__ == "__main__":
    agents.cli.run_app(server)
