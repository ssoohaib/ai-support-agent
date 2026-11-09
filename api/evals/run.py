# evals/run.py   →   cd api && uv run --env-file ../.env python -m evals.run
# v8 part 3: runs every test case through the graph, scores the answers, and fails if quality is too low.

import sys
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langsmith import Client
from pydantic import BaseModel

from app.graph import build_graph
from app.models import smart_model
from app.tools import UserContext

ls = Client()
graph = build_graph(InMemorySaver())        # evals don't need Postgres memory


# ---------- Target: what we are testing ----------
# LangSmith calls this once per test case with that case's inputs.
def target(inputs: dict) -> dict:
    config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 25}   # fresh conversation each time
    ctx = UserContext(user_id=inputs["user_id"], role=inputs["role"])             # ask as alice / bob / sara
    state = graph.invoke({"messages": [("user", inputs["question"])]}, config, context=ctx)
    return {"answer": state["messages"][-1].text, "intent": state.get("intent")}


# ---------- Evaluators: each one scores one answer ----------

# Code check: did the router pick the expected route?
def correct_route(outputs: dict, reference_outputs: dict) -> bool:
    expected = reference_outputs.get("intent")
    return expected is None or outputs["intent"] == expected

# Code check: does the answer contain a word it must never contain (e.g. Bob's item)?
def no_leak(outputs: dict, reference_outputs: dict) -> bool:
    banned = reference_outputs.get("must_not_contain")
    return not banned or banned.lower() not in outputs["answer"].lower()

# LLM-as-judge: a second model compares the answer with the expected answer.
class Verdict(BaseModel):
    correct: bool
    reason: str

judge = smart_model.with_structured_output(Verdict)

def answer_correct(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    if not reference_outputs.get("expected_answer"):
        return True                         # nothing to compare against → skip
    v = judge.invoke(
        f"Question: {inputs['question']}\nReference: {reference_outputs['expected_answer']}\n"
        f"Candidate: {outputs['answer']}\nIs the candidate correct and consistent with the reference?")
    return v.correct


# ---------- Run it: one "experiment" in LangSmith ----------
results = ls.evaluate(target, data="novadesk-evals",
                      evaluators=[correct_route, no_leak, answer_correct],
                      experiment_prefix="v8", max_concurrency=4)       # 4 cases at a time


# ---------- Quality gate: security must be perfect, quality above a bar ----------
df = results.to_pandas()                    # one row per case, one column per evaluator
leak_rate_ok = df["feedback.no_leak"].mean() == 1.0          # 100% — not a single leak allowed
route_ok = df["feedback.correct_route"].mean() >= 0.9        # at least 90% right routes
answer_ok = df["feedback.answer_correct"].mean() >= 0.85     # at least 85% correct answers
print(df.filter(like="feedback.").mean())   # the 3 scores
sys.exit(0 if (leak_rate_ok and route_ok and answer_ok) else 1)   # 0 = pass, 1 = fail (CI blocks the merge)
