# NovaDesk

An AI customer-support agent for **NovaMart**, a made-up online store. Customers can chat or talk to it about policies, orders, returns and refunds. Staff approve refunds, and every answer is traced and tested.

Built step by step, from v0 to v9, following the guide *"NovaDesk — Build an AI Support Agent from v0 to Production"*.

---

## What it can do

| Version | Capability |
|---|---|
| **v1** Chat | Answers stream back word by word over SSE (`POST /chat`) |
| **v2** RAG | Answers policy questions from `data/policies/` using pgvector search, and cites its sources |
| **v3** Tools | Looks up orders, lists your orders, checks return eligibility, starts returns and issues refunds |
| **v4** Auth + RBAC | JWT login. Customers only see their own orders. Staff can look up customers and read internal policies. |
| **v5** LangGraph | A router sends each message to a specialist (policy, orders, returns, handoff or decline). Conversations are remembered in Postgres. Refunds wait for staff approval. |
| **v6** Guardrails | An input guard blocks prompt injection and abuse. Policy answers are fact-checked against the docs. Card numbers are redacted. Runaway loops are stopped. |
| **v7** Vision | Customers upload a damage photo. Metadata such as GPS is stripped, and a vision model assesses the damage. |
| **v8** Evals + observability | LangSmith traces, thumbs up/down feedback, and a 49-question test set with a quality gate |
| **v9** Voice | Talk to the same agent in the terminal, from a browser, or over a real phone call (LiveKit) |

## How it works

```
message ──► guard ──► router ──► policy (RAG + fact check) ──┐
              │                  orders  (tools)             ├──► answer
              ▼                  returns (tools + approval)  │
            decline              handoff / decline ──────────┘

voice:  speech ──► STT ──► same graph ──► TTS ──► speech
```

## Project layout

```
nova-desk/
├── .env                  secrets (not committed)
├── docker-compose.yml    Postgres + pgvector
├── db/seed.sql           tables + demo users and orders
├── data/policies/        the store's policy docs (internal-* = staff only)
└── api/
    ├── app/
    │   ├── main.py       API routes
    │   ├── graph.py      the LangGraph: router + specialists
    │   ├── policy_rag.py policy answers: search → grade → answer → fact-check
    │   ├── guards.py     input guard
    │   ├── tools.py      the agent's tools + permission checks
    │   ├── rag.py        vector store
    │   ├── vision.py     damage-photo assessment
    │   ├── auth.py       JWT login + roles
    │   ├── db.py         all SQL
    │   ├── models.py     which OpenAI models are used
    │   ├── config.py     settings from .env
    │   └── storage.py    saves uploaded photos to api/uploads/
    ├── scripts/          seed_users.py, ingest.py
    ├── evals/            questions.jsonl, upload.py, run.py
    └── voice/            agent.py (voice worker), dispatch-rule.json
```

---

## Getting started

### 1. Prerequisites
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Docker Desktop (running)
- An OpenAI API key
- A LangSmith API key (for traces, feedback and evals)
- A LiveKit Cloud project (only for browser and phone voice)

### 2. Create `.env` in the project root
```bash
OPENAI_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=novadesk
DATABASE_URL=postgresql+psycopg://nova:nova@localhost:5432/nova   # for the vector store
PG_URI=postgresql://nova:nova@localhost:5432/nova                 # for psycopg + memory
JWT_SECRET=any-long-random-string

# voice (v9)
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
TRUSTED_CALLER_NUMBERS=+15551234567     # demo only: calls from these numbers skip the SMS code

# optional: SMS codes on phone calls (needs a paid Twilio account)
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_VERIFY_SID=...
```

### 3. Install and set up the database
```bash
uv sync                                            # install dependencies
docker compose up -d                               # start Postgres (runs db/seed.sql the first time)
uv run --env-file .env python api/scripts/seed_users.py   # give the demo users passwords
cd api
uv run --env-file ../.env python -m scripts.ingest      # load the policies into pgvector
```

> To reset the database, run `docker compose down -v` and then `docker compose up -d`. The seed runs again only on a fresh volume.

### 4. Run the API
```bash
cd api
uv run --env-file ../.env uvicorn app.main:app --reload
```
Check that it's up: `curl localhost:8000/health`

### 5. Try it
```bash
# log in
TOKEN=$(curl -s localhost:8000/auth/login -H 'Content-Type: application/json' \
  -d '{"email":"alice@example.com","password":"alice123"}' | jq -r .access_token)

# chat (streams the answer)
curl -N localhost:8000/chat -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"message":"Where is my order 1042?","conversation_id":"demo"}'
```

**Demo users**

| Email | Password | Role |
|---|---|---|
| alice@example.com | alice123 | customer |
| bob@example.com | bob123 | customer |
| sara@novamart.com | sara123 | agent (staff) |
| omar@novamart.com | omar123 | admin |

**API routes**

| Route | Who | What |
|---|---|---|
| `GET /health` | anyone | is the server alive |
| `POST /auth/login` | anyone | get a token |
| `POST /chat` | logged in | chat with the agent (SSE stream) |
| `POST /feedback` | logged in | thumbs up/down on an answer (`run_id` from `/chat`) |
| `GET /approvals` | staff | refunds waiting for a decision |
| `POST /approvals/{thread_id}` | staff | `{"decision": "approve"}` or `"reject"` |
| `POST /returns/{order_id}/photo` | logged in | upload a damage photo (JPEG/PNG/WebP, max 5 MB) |
| `POST /voice/token` | logged in | ticket for a browser voice call |

### 6. Evals (v8)
```bash
cd api
uv run --env-file ../.env python -m evals.upload   # once: creates the "novadesk-evals" dataset in LangSmith
uv run --env-file ../.env python -m evals.run      # runs all cases, scores them, fails if quality is too low
```

### 7. Voice (v9)
The voice worker doesn't need the API running. It only needs Postgres.

**In the terminal (no phone needed).** You talk as Alice:
```bash
cd api
uv run --env-file ../.env python -m voice.agent console
```

**Over a phone call:**
1. Rent a number in LiveKit Cloud.
2. Create a dispatch rule for it: type **Individual**, room prefix `call-`, agent name **empty**. This is the same rule as `voice/dispatch-rule.json`.
3. Link your phone to Alice:
   ```bash
   docker compose exec db psql -U nova -d nova -c "UPDATE users SET phone = '+<your number>' WHERE id = 'alice';"
   ```
4. Put the same number in `TRUSTED_CALLER_NUMBERS`.
5. Start the worker:
   ```bash
   cd api
   uv run --env-file ../.env python -m voice.agent dev
   ```
6. Call the number.

Callers whose number isn't linked to an account can only ask policy questions.

---

## Notes
- Demo passwords and the trusted-number shortcut are for practice only. Caller ID can be faked.
- The guide's phone setup uses Twilio and a SIP trunk. This project uses a LiveKit phone number instead.
