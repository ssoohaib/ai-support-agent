# app/main.py
# The API: every URL the outside world can call lives here.
# The AI logic lives in graph.py, the tools in tools.py, the SQL in db.py.

import json
import os
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Literal
from uuid import uuid4

import bcrypt
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from langsmith import Client
from livekit import api
from pydantic import BaseModel

from app import db, storage
from app.auth import User, current_user, issue_token, require_role
from app.config import settings
from app.graph import build_graph
from app.tools import UserContext
from app.vision import assess_photo


# ---------- 1. Startup: runs once when the server starts ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_tables()                                    # our approvals table
    # The "saver" (checkpointer) stores every conversation in Postgres = the AI's memory
    async with AsyncPostgresSaver.from_conn_string(settings.pg_uri) as saver:
        await saver.setup()                             # LangGraph's memory tables
        app.state.graph = build_graph(saver)            # build the graph once, with memory
        yield                                           # ← the server runs here
        # (code after yield would run when the server stops)

app = FastAPI(title="NovaDesk", lifespan=lifespan)
ls = Client()                                           # v8: talks to LangSmith (uses LANGSMITH_API_KEY from .env)


# ---------- 2. What each request body looks like ----------
# FastAPI checks incoming JSON against these and rejects anything that doesn't match.

class LoginIn(BaseModel):
    email: str
    password: str

class ChatIn(BaseModel):
    message: str
    conversation_id: str = "default"                    # same id = same conversation (memory)

class DecisionIn(BaseModel):
    decision: Literal["approve", "reject"]              # only these two values are allowed

class FeedbackIn(BaseModel):                            # v8: thumbs up / down on one answer
    run_id: str                                         # which answer (the id /chat sent first)
    thumbs_up: bool
    comment: str | None = None                          # optional: "it got the date wrong"


# ---------- 3. Health + login ----------

@app.get("/health")
def health():
    return {"status": "ok"}                             # "is the server alive?"

@app.post("/auth/login")
def login(body: LoginIn):
    user = db.fetch_user_by_email(body.email)
    # checkpw hashes the typed password and compares it with the stored hash
    ok = (user is not None and user["password_hash"] is not None
          and bcrypt.checkpw(body.password.encode(), user["password_hash"].encode()))
    if not ok:
        raise HTTPException(401, "Wrong email or password")   # same message: don't reveal which emails exist
    # The token proves "I am alice, a customer" on every later request
    return {"access_token": issue_token(user["id"], user["role"]), "token_type": "bearer"}


# ---------- 4. Chat ----------

def visible(chunk, meta) -> bool:
    """Only send real answer text to the user: no tool calls.
    (Router, guard and policy helpers are tagged "nostream" since v9, so they never get here.)"""
    return isinstance(chunk, AIMessageChunk) and bool(chunk.text)

# Depends(current_user) = "this route needs a valid token"; FastAPI checks it before running the code
@app.post("/chat")
async def chat(body: ChatIn, user: User = Depends(current_user)):
    graph = app.state.graph
    thread_id = f"{user.id}:{body.conversation_id}"     # "alice:default": each user has their own threads
    run_id = uuid4()                                    # v8: our own id for this answer's trace
    config = {
        "configurable": {"thread_id": thread_id},       # which memory to load and save
        "recursion_limit": 25,                          # max 25 steps per message (v6: stops loops)
        "run_id": run_id,                               # v8: lets feedback attach to this trace
        "metadata": {"user_role": user.role, "channel": "chat", "app_version": "v8"},  # v8: filters in LangSmith
        "tags": ["chat"],
    }
    ctx = UserContext(user_id=user.id, role=user.role)  # from the token, never from the model

    # A generator: each `yield` sends one piece of text to the user right away (streaming)
    async def events():
        # v8: first event = the run_id, so the browser can send thumbs up/down for this answer later
        yield f"event: run\ndata: {json.dumps({'run_id': str(run_id)})}\n\n"

        # subgraphs=True: also stream text from the specialist agents inside the graph
        # Two kinds of output (v6):
        #   "messages" = text the agents type out word by word (orders, returns)
        #   "custom"   = finished text sent with get_stream_writer() (policy_rag's checked answer)
        async for _ns, mode, payload in graph.astream(
            {"messages": [("user", body.message)]}, config,
            context=ctx, stream_mode=["messages", "custom"], subgraphs=True,
        ):
            if mode == "messages" and visible(*payload):        # payload = (chunk, meta)
                yield f"data: {json.dumps({'delta': payload[0].text})}\n\n"
            elif mode == "custom":                              # payload = the text itself
                yield f"data: {json.dumps({'delta': payload})}\n\n"

        # Did the graph stop to wait for a human? (a refund needs approval)
        snapshot = await graph.aget_state(config)       # the conversation's current saved state
        if snapshot.interrupts:
            # .value = what the AI wanted to do, e.g. issue_refund for order 1047, $449.99
            db.save_pending_approval(thread_id, user.id, snapshot.interrupts[0].value)
            yield 'event: pending_approval\ndata: {"message": "Your refund is waiting for approval."}\n\n'

        yield "event: done\ndata: {}\n\n"               # tells the client the answer is finished

    return StreamingResponse(events(), media_type="text/event-stream")


# v8: the user rates one answer; LangSmith stores it next to that answer's full trace
@app.post("/feedback")
def feedback(body: FeedbackIn, user: User = Depends(current_user)):
    ls.create_feedback(body.run_id, key="user_score", score=1 if body.thumbs_up else 0,
                       comment=body.comment)
    return {"ok": True}


# ---------- 5. Approvals: staff only ----------
# require_role(...) = valid token AND the right role, otherwise 403 Forbidden

@app.get("/approvals", dependencies=[Depends(require_role("agent", "admin"))])
def list_approvals():
    """Refunds waiting for a decision (the guide's dashboard shows this in v10)."""
    return db.list_pending_approvals()

@app.post("/approvals/{thread_id}")
async def decide(thread_id: str, body: DecisionIn,
                 staff: User = Depends(require_role("agent", "admin"))):
    pending = db.get_pending_approval(thread_id)
    if not pending:
        raise HTTPException(404, "Nothing waiting for approval in this conversation")

    # Wake the paused graph up with the staff member's answer.
    # Tools run as the customer, so their ownership checks still apply.
    ctx = UserContext(user_id=pending["customer_id"], role="customer")
    await app.state.graph.ainvoke(
        Command(resume={"decisions": [{"type": body.decision}]}),   # approve → issue_refund runs; reject → it doesn't
        {"configurable": {"thread_id": thread_id}},                 # continue THIS conversation
        context=ctx,
    )
    db.close_pending_approval(thread_id, decided_by=staff.id, decision=body.decision)
    return {"ok": True}


# ---------- 6. Return photos (v7) ----------

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES = 5 * 1024 * 1024                             # 5 MB

def clean_image(data: bytes) -> tuple[bytes, str]:
    """Upload safety (from the guide):
    - the browser's content type can lie, so open the file to prove it's a real image
    - re-save it with pixels only, so metadata like the customer's GPS location is gone"""
    try:
        img = Image.open(BytesIO(data))
        img.load()                                      # actually read the pixels: fake/broken files fail here
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "That file isn't a valid image")
    if img.format not in ("JPEG", "PNG", "WEBP"):
        raise HTTPException(415, "Use a JPEG, PNG or WebP photo")

    fmt = img.format
    img = ImageOps.exif_transpose(img)                  # keep the photo the right way up before metadata is dropped
    if img.mode == "P":
        img = img.convert("RGBA")                       # palette images: keep their colours
    clean = Image.frombytes(img.mode, img.size, img.tobytes())   # a fresh image: pixels only, no metadata
    out = BytesIO()
    clean.save(out, format=fmt)
    return out.getvalue(), Image.MIME[fmt]              # the cleaned bytes + the REAL type, e.g. "image/jpeg"

# UploadFile = the request carries a file (sent as a form upload, not JSON)
@app.post("/returns/{order_id}/photo")
async def upload_photo(order_id: str, file: UploadFile, user: User = Depends(current_user)):
    # 1. Checks: is it your order? is it an image? is it small enough?
    order = db.fetch_order(order_id)
    if not order or (user.role == "customer" and order["user_id"] != user.id):
        raise HTTPException(404, "Order not found")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Use a JPEG, PNG or WebP photo")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Photo too large (max 5 MB)")
    data, mime_type = clean_image(data)                 # real image? strip GPS & other metadata

    # 2. Keep the photo, then let the vision model look at it
    path = storage.save(f"returns/{order_id}/{uuid4()}", data)
    items = [item["name"] for item in order["items"]]   # what the customer ordered, e.g. ["Headphones X2"]
    # assess_photo is a normal (slow) function → run it in a thread so the server isn't blocked
    result = await run_in_threadpool(assess_photo, data, mime_type, items)
    db.save_assessment(order_id, path, result.model_dump())

    # 3. YOUR CODE decides the next step, not the model:
    #    right product + major damage → return label straight away; anything else → a person looks at it
    auto_ok = result.shows_a_product and result.product_matches_order and result.severity == "major"
    return {"assessment": result, "next_step": "return_label" if auto_ok else "staff_review"}


# ---------- 7. Voice (v9): the browser asks for a ticket to join a call ----------

@app.post("/voice/token")
def voice_token(user: User = Depends(current_user)):
    room = f"web-{user.id}-{uuid4().hex[:8]}"           # a new private room for this call
    token = (api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
             .with_identity(user.id)                      # the worker trusts this identity
             .with_grants(api.VideoGrants(room_join=True, room=room))   # may join THIS room only
             .to_jwt())
    return {"token": token, "url": os.environ["LIVEKIT_URL"]}


# ---------- 8. Admin ----------

@app.post("/admin/reindex", dependencies=[Depends(require_role("admin"))])
def reindex():
    # The guide leaves this empty for now: later it re-runs the ingest steps.
    ...
