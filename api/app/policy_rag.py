# app/policy_rag.py
# The new policy specialist (v6). Replaces policy_agent in graph.py.
# Instead of "search once and answer", it checks its own work:
#   search → are the results useful? → (if not: rephrase + search again) → write a draft → is the draft true? → answer

from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel

from app.models import fast_model, smart_model
from app.rag import format_docs
from app.tools import ALLOWED_AUDIENCE, UserContext, store


# The system prompt from v2 (main.py had it then; now it lives here, where it's used).
RAG_SYSTEM = """You are NovaMart's support assistant.
Answer ONLY from the context below and cite sources like [returns.md].
If the context doesn't contain the answer, say you don't know and offer to connect a human.

<context>
{context}
</context>"""


# This mini-graph's own notepad. It has the messages, plus 4 working fields.
class PolicyState(MessagesState):
    question: str        # what we search for (may get rephrased once)
    context: str         # the policy text found by the search
    draft: str           # the answer, before it's checked
    tries: int           # how many times we've rephrased (max 1)

# Forced yes/no answers for the two checker models.
class Relevant(BaseModel):
    relevant: bool       # "are these search results useful for the question?"

class Grounded(BaseModel):
    supported: bool      # "is every claim in the draft backed by the policy text?"


# 4 small helpers. All tagged "nostream" (v9) → their tokens never enter the stream,
# so neither chat nor voice can show/speak them. Only respond/fallback send text.
grader = fast_model.with_structured_output(Relevant).with_config(tags=["nostream"])
checker = fast_model.with_structured_output(Grounded).with_config(tags=["nostream"])
rewriter = fast_model.with_config(tags=["nostream"])
writer = smart_model.with_config(tags=["nostream"])   # draft is checked before it's shown


# ---------- Nodes (the boxes) ----------

# 1. Search the policies (customers only see "public" docs, staff also "internal").
def retrieve(state: PolicyState, runtime: Runtime[UserContext]):
    question = state.get("question") or state["messages"][-1].text   # rephrased question, or the user's message
    allowed = ALLOWED_AUDIENCE[runtime.context.role]
    docs = store.similarity_search(question, k=4, filter={"audience": {"$in": allowed}})
    return {"question": question, "context": format_docs(docs)}

# 2. Decide where to go after searching (this returns a NODE NAME, not state).
def grade(state: PolicyState) -> str:
    r = grader.invoke(f"Does this context help answer the question?\nQuestion: {state['question']}\n"
                      f"<context>{state['context']}</context>")
    if r.relevant:
        return "generate"                                       # useful → write the answer
    return "rewrite" if state.get("tries", 0) < 1 else "fallback"   # not useful → try once more, then give up

# 3. Rephrase the question into a better search query.
def rewrite(state: PolicyState):
    better = rewriter.invoke(f"Rewrite as a short search query about store policies: {state['question']}").text
    return {"question": better, "tries": state.get("tries", 0) + 1}

# 4. Write a draft answer from the found text (not shown to the user yet).
#    (built inside build_policy_rag below, because voice adds its own style to the prompt)

# 5. Fact-check the draft against the found text (returns a NODE NAME).
def check(state: PolicyState) -> str:
    v = checker.invoke(f"Is every claim in the ANSWER supported by the CONTEXT?\n"
                       f"<context>{state['context']}</context>\n<answer>{state['draft']}</answer>")
    return "respond" if v.supported else "fallback"

# 6a. Draft passed → send it to the user.
def respond(state: PolicyState):
    get_stream_writer()(state["draft"])           # send only the checked answer
    return {"messages": [AIMessage(state["draft"])]}

# 6b. Nothing useful found, or the draft failed the check → safe answer instead of a guess.
def fallback(state: PolicyState):
    text = "I couldn't find that in our policies. Want me to connect you with a person?"
    get_stream_writer()(text)
    return {"messages": [AIMessage(text)]}


# ---------- Wiring (the arrows) ----------

def build_policy_rag(voice_style: str | None = None):
    """Chat: build_policy_rag(). Voice (v9): build_policy_rag(VOICE_STYLE) adds the voice style
    to the answer, and says a short filler while the (slow) search + fact-check run."""
    system = RAG_SYSTEM + (f"\n\n{voice_style}" if voice_style else "")

    def retrieve_step(state: PolicyState, runtime: Runtime[UserContext]):
        if voice_style and not state.get("tries"):              # v9: fill the silence, first search only
            get_stream_writer()("Let me check that for you.")
        return retrieve(state, runtime)

    def generate(state: PolicyState):
        draft = writer.invoke([("system", system.format(context=state["context"])), *state["messages"][-6:]])
        return {"draft": draft.text}

    p = StateGraph(PolicyState, context_schema=UserContext)
    for name, fn in [("retrieve", retrieve_step), ("rewrite", rewrite), ("generate", generate),
                     ("respond", respond), ("fallback", fallback)]:
        p.add_node(name, fn)
    p.add_edge(START, "retrieve")
    p.add_conditional_edges("retrieve", grade, ["generate", "rewrite", "fallback"])   # grade() picks one
    p.add_edge("rewrite", "retrieve")                                                 # loop back: search again
    p.add_conditional_edges("generate", check, ["respond", "fallback"])               # check() picks one
    p.add_edge("respond", END)
    p.add_edge("fallback", END)
    return p.compile()            # in build_graph: g.add_node("policy", build_policy_rag(...))
