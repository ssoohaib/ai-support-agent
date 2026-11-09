# app/db.py
import json
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import settings

pool = ConnectionPool(settings.pg_uri, kwargs={"row_factory": dict_row}, open=True)

def fetch_order(order_id: str) -> dict | None:
    with pool.connection() as conn:
        return conn.execute("SELECT * FROM orders WHERE id = %s", (order_id,)).fetchone()

def create_return(order_id: str, reason: str) -> str:
    with pool.connection() as conn:
        conn.execute("UPDATE orders SET status = 'return_requested' WHERE id = %s", (order_id,))
    return f"Return started for order {order_id}."


def orders_for_user(user_id: str) -> list[dict]:
    with pool.connection() as conn:
        return conn.execute(
            "SELECT id, items, total_cents, status, delivered_at FROM orders "
            "WHERE user_id = %s ORDER BY id DESC LIMIT 10",
            (user_id,),
        ).fetchall()


def fetch_user_by_email(email: str) -> dict | None:
    with pool.connection() as conn:
        return conn.execute("SELECT * FROM users WHERE email = %s", (email,)).fetchone()


def create_refund(order_id: str, amount_cents: int, reason: str) -> bool:
    """Returns False if this order already has a refund."""
    with pool.connection() as conn:
        cur = conn.execute(
            "INSERT INTO refunds (order_id, amount_cents, reason, status) "
            "VALUES (%s, %s, %s, 'approved') "
            "ON CONFLICT (order_id) DO NOTHING",      # UNIQUE (order_id) → no double refunds
            (order_id, amount_cents, reason),
        )
        return cur.rowcount == 1


# ---------- Approvals (v5): refunds waiting for a staff member ----------

def init_tables() -> None:
    """Creates our extra tables if they aren't there yet. Runs once at startup."""
    with pool.connection() as conn:
        # v7: what the vision model saw in each return photo
        conn.execute("""
            CREATE TABLE IF NOT EXISTS photo_assessments (
              id           serial PRIMARY KEY,
              order_id     text NOT NULL REFERENCES orders(id),
              photo_path   text NOT NULL,                       -- where storage.save() put the file
              assessment   jsonb NOT NULL,                      -- the DamageAssessment fields
              created_at   timestamptz DEFAULT now()
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_approvals (
              id           serial PRIMARY KEY,
              thread_id    text NOT NULL,                       -- which conversation is paused
              customer_id  text NOT NULL REFERENCES users(id),
              request      jsonb NOT NULL,                      -- what the AI wants to do (tool + arguments)
              decision     text,                                -- empty while waiting, then approve | reject
              decided_by   text REFERENCES users(id),
              created_at   timestamptz DEFAULT now(),
              decided_at   timestamptz
            )""")
        # v9: each customer's phone number (for caller ID + SMS codes)
        conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS phone text UNIQUE")
        # v9: phone calls whose caller has read back the right SMS code
        conn.execute("""
            CREATE TABLE IF NOT EXISTS call_verifications (
              call_id      text PRIMARY KEY,                    -- the LiveKit room name
              verified_at  timestamptz DEFAULT now()
            )""")


def save_pending_approval(thread_id: str, customer_id: str, request: dict) -> None:
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO pending_approvals (thread_id, customer_id, request) VALUES (%s, %s, %s::jsonb)",
            (thread_id, customer_id, json.dumps(request, default=str)),
        )


def list_pending_approvals() -> list[dict]:
    with pool.connection() as conn:
        return conn.execute(
            "SELECT thread_id, customer_id, request, created_at FROM pending_approvals "
            "WHERE decision IS NULL ORDER BY created_at"
        ).fetchall()


def get_pending_approval(thread_id: str) -> dict | None:
    with pool.connection() as conn:
        return conn.execute(
            "SELECT * FROM pending_approvals WHERE thread_id = %s AND decision IS NULL "
            "ORDER BY id DESC LIMIT 1",
            (thread_id,),
        ).fetchone()


def close_pending_approval(thread_id: str, decided_by: str, decision: str) -> None:
    with pool.connection() as conn:
        conn.execute(
            "UPDATE pending_approvals SET decision = %s, decided_by = %s, decided_at = now() "
            "WHERE thread_id = %s AND decision IS NULL",
            (decision, decided_by, thread_id),
        )


# ---------- Photo assessments (v7) ----------

def save_assessment(order_id: str, photo_path: str, assessment: dict) -> None:
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO photo_assessments (order_id, photo_path, assessment) VALUES (%s, %s, %s::jsonb)",
            (order_id, photo_path, json.dumps(assessment)),
        )


def latest_assessment(order_id: str) -> dict | None:
    """The newest photo assessment for this order, or None."""
    with pool.connection() as conn:
        return conn.execute(
            "SELECT assessment, created_at FROM photo_assessments WHERE order_id = %s "
            "ORDER BY id DESC LIMIT 1",
            (order_id,),
        ).fetchone()


# ---------- Voice (v9) ----------

def fetch_user(user_id: str) -> dict | None:
    with pool.connection() as conn:
        return conn.execute("SELECT * FROM users WHERE id = %s", (user_id,)).fetchone()


def fetch_user_by_phone(phone: str) -> dict | None:
    with pool.connection() as conn:
        return conn.execute("SELECT * FROM users WHERE phone = %s", (phone,)).fetchone()


def phone_for_user(user_id: str) -> str | None:
    user = fetch_user(user_id)
    return user["phone"] if user else None


def mark_call_verified(call_id: str) -> None:
    with pool.connection() as conn:
        conn.execute("INSERT INTO call_verifications (call_id) VALUES (%s) ON CONFLICT DO NOTHING", (call_id,))


def call_verified(call_id: str | None) -> bool:
    if not call_id:
        return False
    with pool.connection() as conn:
        return conn.execute("SELECT 1 FROM call_verifications WHERE call_id = %s", (call_id,)).fetchone() is not None


def queue_refund(order_id: str, amount_cents: int, reason: str) -> bool:
    """A refund asked for on a call: saved as 'pending' for staff. False if the order already has one."""
    with pool.connection() as conn:
        cur = conn.execute(
            "INSERT INTO refunds (order_id, amount_cents, reason, status) "
            "VALUES (%s, %s, %s, 'pending') "
            "ON CONFLICT (order_id) DO NOTHING",
            (order_id, amount_cents, reason),
        )
        return cur.rowcount == 1
