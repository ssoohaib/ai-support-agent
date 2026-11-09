# app/tools.py
from datetime import date
from functools import cache
import json
import os
from langchain.tools import tool, ToolRuntime
from twilio.rest import Client as Twilio

from app import db
from app.rag import get_store, format_docs

from dataclasses import dataclass

# v9: the context grows for voice
@dataclass
class UserContext:
    user_id: str
    role: str                      # customer | agent | admin
    channel: str = "chat"          # chat | web_voice | phone
    call_id: str | None = None     # the LiveKit room name, for per-call state

ALLOWED_AUDIENCE = {"customer": ["public"], "agent": ["public", "internal"], "admin": ["public", "internal"]}

store = get_store()


def _needs_verification(user: UserContext) -> str | None:
    """v9: caller ID can be faked, so on a phone call no order data until the SMS code is checked."""
    if user.channel == "phone" and not db.call_verified(user.call_id):
        return ("The caller isn't verified yet. Use send_code to text them a code, "
                "then check_code with the code they read out.")
    return None


@tool
def search_policies(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """Search NovaMart's policies. Returns passages tagged with [source]."""
    allowed = ALLOWED_AUDIENCE[runtime.context.role]
    docs = store.similarity_search(query, k=4, filter={"audience": {"$in": allowed}})
    return format_docs(docs)

@tool
def get_order(order_id: str, runtime: ToolRuntime[UserContext]) -> str:
    """Get an order's status, items, total and delivery date. order_id looks like '1042'."""
    user = runtime.context                     # comes from the JWT, never from the model
    if problem := _needs_verification(user):
        return problem
    order = db.fetch_order(order_id)
    if not order or (user.role == "customer" and order["user_id"] != user.user_id):
        return "Order not found."              # same reply: don't reveal that it exists
    return json.dumps(order, default=str)

@tool
def get_photo_assessment(order_id: str, runtime: ToolRuntime[UserContext]) -> str:
    """Get what the customer's damage photo for this order shows (damage type, severity, notes).
    Use it for damaged-item returns."""
    user = runtime.context
    if problem := _needs_verification(user):
        return problem
    order = db.fetch_order(order_id)
    if not order or (user.role == "customer" and order["user_id"] != user.user_id):
        return "Order not found."              # same ownership check as get_order
    found = db.latest_assessment(order_id)
    if not found:
        return "No photo has been uploaded for this order."
    return json.dumps(found, default=str)

@tool
def list_my_orders(runtime: ToolRuntime[UserContext]) -> str:
    """List the logged-in customer's recent orders."""
    if problem := _needs_verification(runtime.context):
        return problem
    return json.dumps(db.orders_for_user(runtime.context.user_id), default=str)

@tool
def create_return_request(order_id: str, reason: str, runtime: ToolRuntime[UserContext]) -> str:
    """Start a return for a delivered order.
    Only call after confirming the order and the reason with the customer."""
    if problem := _needs_verification(runtime.context):
        return problem
    order = db.fetch_order(order_id)
    problem = _return_problem(order, runtime.context)
    if problem:
        return problem
    return db.create_return(order_id, reason)

def _return_problem(order, user) -> str | None:
    """Not a tool: a helper shared by the two return tools.
    Returns why this order can't be returned, or None if it can."""
    if not order or (user.role == "customer" and order["user_id"] != user.user_id):
        return "Order not found."
    if order["status"] != "delivered":
        return "Only delivered orders can be returned."
    days = (date.today() - order["delivered_at"]).days
    if days > 30:
        return f"This order was delivered {days} days ago, past the 30-day return window."
    return None

@tool
def check_return_eligibility(order_id: str, runtime: ToolRuntime[UserContext]) -> str:
    """Check whether an order can be returned. Changes nothing.
    Call this before create_return_request."""
    if problem := _needs_verification(runtime.context):
        return problem
    order = db.fetch_order(order_id)
    problem = _return_problem(order, runtime.context)
    if problem:
        return f"Not eligible: {problem}"
    days = (date.today() - order["delivered_at"]).days
    return f"Eligible: delivered {days} days ago (limit is 30). Order total ${order['total_cents'] / 100:.2f}."

@tool
def issue_refund(order_id: str, amount: float, reason: str, runtime: ToolRuntime[UserContext]) -> str:
    """Refund money for an order to the customer's original payment method.
    amount is in dollars, e.g. 89.99. A staff member approves every refund before it runs."""
    if problem := _needs_verification(runtime.context):
        return problem
    order = db.fetch_order(order_id)
    user = runtime.context
    if not order or (user.role == "customer" and order["user_id"] != user.user_id):
        return "Order not found."
    amount_cents = round(amount * 100)
    if amount_cents <= 0 or amount_cents > order["total_cents"]:
        return f"Refund must be between $0.01 and the order total (${order['total_cents'] / 100:.2f})."
    if not db.create_refund(order_id, amount_cents, reason):
        return "This order has already been refunded."
    return f"Refund of ${amount_cents / 100:.2f} issued for order {order_id}."

@tool
def lookup_customer(email: str, runtime: ToolRuntime[UserContext]) -> str:
    """Staff only. Find a customer by email and list their recent orders."""
    if runtime.context.role == "customer":
        return "Not allowed."
    user = db.fetch_user_by_email(email)
    if not user or user["role"] != "customer":
        return "Customer not found."
    orders = db.orders_for_user(user["id"])
    return json.dumps({"id": user["id"], "name": user["name"],
                       "email": user["email"], "orders": orders}, default=str)


# ---------- v9: voice only ----------

@cache
def _verify():
    """Twilio Verify. Created on first use (not at import), so chat still works without Twilio keys."""
    tw = Twilio(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    return tw.verify.v2.services(os.environ["TWILIO_VERIFY_SID"])

@tool
def send_code(runtime: ToolRuntime[UserContext]) -> str:
    """Text a 6-digit code to the caller's phone number on file. Required before any order or return action on a phone call."""
    phone = db.phone_for_user(runtime.context.user_id)
    if not phone:                              # unknown number → policy answers only
        return "There's no account for this phone number, so I can only help with policy questions."
    if not os.environ.get("TWILIO_VERIFY_SID"):  # Twilio Verify not set up (e.g. trial account)
        return "Verification by text message isn't available, so I can only help with policy questions on this call."
    _verify().verifications.create(to=phone, channel="sms")
    return "I've sent a code by text message."

@tool
def check_code(code: str, runtime: ToolRuntime[UserContext]) -> str:
    """Check the code the caller reads out."""
    phone = db.phone_for_user(runtime.context.user_id)
    if not phone:
        return "There's no account for this phone number."
    result = _verify().verification_checks.create(to=phone, code=code)
    if result.status == "approved":
        db.mark_call_verified(runtime.context.call_id)
        return "Verified."
    return "That code didn't match."

@tool
def request_refund(order_id: str, amount: float, reason: str, runtime: ToolRuntime[UserContext]) -> str:
    """Ask for a refund during a call. amount is in dollars, e.g. 89.99.
    The call can't wait for a staff member, so the refund is queued for staff review."""
    if problem := _needs_verification(runtime.context):
        return problem
    order = db.fetch_order(order_id)
    user = runtime.context
    if not order or (user.role == "customer" and order["user_id"] != user.user_id):
        return "Order not found."
    amount_cents = round(amount * 100)
    if amount_cents <= 0 or amount_cents > order["total_cents"]:
        return f"Refund must be between $0.01 and the order total (${order['total_cents'] / 100:.2f})."
    if not db.queue_refund(order_id, amount_cents, reason):
        return "This order already has a refund."
    return "Refund request saved. A staff member will review it, and the customer will get a text with the decision."
