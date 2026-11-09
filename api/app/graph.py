# app/graph.py
from typing import Literal
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, PIIMiddleware
from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph

from app.models import fast_model, smart_model
from app.policy_rag import build_policy_rag
from app.tools import (UserContext, list_my_orders, get_order, get_photo_assessment,
                       check_return_eligibility, create_return_request, issue_refund,
                       send_code, check_code, request_refund)

class SupportState(MessagesState):
    intent: str

class Route(BaseModel):
    intent: Literal["policy", "order", "return", "human", "other"]

# v9: "nostream" = LangGraph never puts this call's tokens in the stream.
# Voice reads the stream directly, so router JSON must never get there (it would be spoken).
router = (fast_model
          .with_structured_output(Route)
          .with_config(tags=["nostream"]))

ROUTER_PROMPT = """Label the customer's latest message for a store support desk:
policy (rules, shipping, warranty), order (status of their orders), return (returns, refunds,
damaged items), human (asks for a person), other (unrelated to the store).
If the customer is just answering the assistant's last question (yes/no, an order number,
a verification code), use the label of the request they are answering."""

def classify(state: SupportState):
    route = router.invoke([("system", ROUTER_PROMPT), *state["messages"][-6:]])
    return {"intent": route.intent}

ORDERS_PROMPT = """You help customers with the status of their NovaMart orders.
- If the customer gives no order number, use list_my_orders.
- For a specific order, use get_order. Never guess order details.
- If an order isn't found, say so. Don't speculate.
- Prices in the data are in cents: 8999 means $89.99.
- Keep answers short."""

RETURNS_PROMPT = """You handle returns and refunds for NovaMart.
1. Always call check_return_eligibility first. If it's not eligible, explain why and stop.
2. If eligible, confirm the order and the reason with the customer, then call create_return_request.
3. For a refund, call issue_refund with the amount in dollars, never more than the order total.
   A staff member approves every refund; tell the customer it's waiting for approval.
- Prices in the data are in cents: 8999 means $89.99.
- Keep answers short."""

# v9: added to every specialist when build_graph(channel="voice")
VOICE_STYLE = """You are speaking on a phone call. Use short spoken sentences.
No lists, markdown or emoji. Say numbers and dates naturally.
Ask one question at a time. Read back order numbers before acting on them."""

# v9: how phone callers prove who they are (the tools refuse until this is done)
VOICE_VERIFY = """If a tool says the caller isn't verified: call send_code, ask the caller to read
the code from their text message, then call check_code with it. Then retry what they asked for."""

# v6: hides card numbers in the user's messages before the model (and the traces) see them
pii = PIIMiddleware("credit_card", strategy="redact", apply_to_input=True)


def build_specialists(channel: str):
    """The orders and returns agents. Voice gets voice prompts, the code tools,
    and request_refund (queued) instead of issue_refund (which pauses for staff)."""
    if channel == "voice":
        orders = create_agent(smart_model, tools=[send_code, check_code, list_my_orders, get_order],
                              system_prompt=f"{ORDERS_PROMPT}\n\n{VOICE_STYLE}\n\n{VOICE_VERIFY}",
                              context_schema=UserContext, middleware=[pii])
        returns = create_agent(
            smart_model,
            tools=[send_code, check_code, get_order, get_photo_assessment, check_return_eligibility,
                   create_return_request, request_refund],
            system_prompt=(f"{RETURNS_PROMPT.replace('issue_refund', 'request_refund')}\n\n"
                           f"{VOICE_STYLE}\n\n{VOICE_VERIFY}"),
            context_schema=UserContext,
            middleware=[pii],                    # no human-in-the-loop: a call can't wait for staff
        )
        return orders, returns

    orders = create_agent(smart_model, tools=[list_my_orders, get_order],
                          system_prompt=ORDERS_PROMPT, context_schema=UserContext,
                          middleware=[pii])
    returns = create_agent(
        smart_model,
        tools=[get_order, get_photo_assessment, check_return_eligibility, create_return_request, issue_refund],
        system_prompt=RETURNS_PROMPT,
        context_schema=UserContext,
        middleware=[
            pii,
            HumanInTheLoopMiddleware(interrupt_on={"issue_refund": {"allowed_decisions": ["approve", "reject"]}}),
        ],
    )
    return orders, returns


def handoff(state: SupportState):
    # create a ticket with a short summary of the conversation, then:
    text = "I've passed this to our team. A person will reply within 24 hours."
    get_stream_writer()(text)                   # send it to the user ("custom" stream)
    return {"messages": [AIMessage(text)]}      # and save it in the conversation memory

def decline(state: SupportState):
    text = "I can only help with NovaMart orders, returns and policies."
    get_stream_writer()(text)                   # send it to the user ("custom" stream)
    return {"messages": [AIMessage(text)]}      # and save it in the conversation memory

def build_graph(checkpointer=None, channel: str = "chat"):
    """channel="chat" for the API, "voice" for the LiveKit worker (v9)."""
    from app.guards import input_guard          # imported here, not at the top: guards.py imports graph.py
    orders_agent, returns_agent = build_specialists(channel)
    g = StateGraph(SupportState, context_schema=UserContext)
    g.add_node("guard", input_guard)
    g.add_node("classify", classify)
    g.add_node("policy", build_policy_rag(VOICE_STYLE if channel == "voice" else None))   # v6: replaces policy_agent
    g.add_node("orders", orders_agent)
    g.add_node("returns", returns_agent)
    g.add_node("handoff", handoff)
    g.add_node("decline", decline)
    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", lambda s: "decline" if s["intent"] == "blocked" else "classify",
                            ["decline", "classify"])
    g.add_conditional_edges("classify", lambda s: s["intent"], {
        "policy": "policy", "order": "orders", "return": "returns",
        "human": "handoff", "other": "decline",
    })
    for node in ["policy", "orders", "returns", "handoff", "decline"]:
        g.add_edge(node, END)
    return g.compile(checkpointer=checkpointer)
