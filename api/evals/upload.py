# evals/upload.py   →   cd api && uv run --env-file ../.env python -m evals.upload
# v8 part 2: uploads the test questions to LangSmith as a dataset. Run it ONCE.

import json
from pathlib import Path

from langsmith import Client

ls = Client()

# Read evals/questions.jsonl: one test case per line
lines = Path(__file__).with_name("questions.jsonl").read_text().splitlines()
cases = [json.loads(line) for line in lines if line.strip()]

# From the guide: create the dataset, then add every case as an "example"
ds = ls.create_dataset("novadesk-evals")
ls.create_examples(
    dataset_id=ds.id,
    # inputs = what we send to the agent
    inputs=[{"question": c["question"], "role": c["role"], "user_id": c["user_id"]} for c in cases],
    # outputs = what we expect back (used by the evaluators in part 3)
    outputs=[{"intent": c.get("intent"), "expected_answer": c.get("expected_answer"),
              "must_not_contain": c.get("must_not_contain")} for c in cases],
)
print(f"Uploaded {len(cases)} cases to the dataset 'novadesk-evals'")
