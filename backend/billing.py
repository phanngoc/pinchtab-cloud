"""Stripe webhook + tier sync.

Idempotency: every Stripe event_id recorded in stripe_events table (S4 from
CEO review). Duplicate webhooks become no-ops.

VNPay is intentionally absent — CEO review fix 'Stripe-only at launch,
VNPay v1.1'.
"""
from __future__ import annotations

import json

import stripe
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session as DbSession

from backend.config import get_settings
from backend.db import get_db
from backend.models import StripeEvent, Tier, User

router = APIRouter(prefix="/billing", tags=["billing"])


def _settings_or_400():
    s = get_settings()
    if not s.stripe_secret_key or not s.stripe_webhook_secret:
        raise HTTPException(status_code=503, detail="stripe_not_configured")
    return s


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str | None = Header(default=None, alias="Stripe-Signature"),
    db: DbSession = Depends(get_db),
) -> dict:
    settings = _settings_or_400()
    if not stripe_signature:
        raise HTTPException(status_code=400, detail="missing_signature")

    payload = await request.body()
    try:
        event = stripe.Webhook.construct_event(
            payload, stripe_signature, settings.stripe_webhook_secret
        )
    except stripe.error.SignatureVerificationError as e:  # type: ignore[attr-defined]
        raise HTTPException(status_code=400, detail="bad_signature") from e

    event_id = event["id"]
    # Idempotency check
    if db.get(StripeEvent, event_id) is not None:
        return {"status": "duplicate"}
    db.add(StripeEvent(id=event_id, event_type=event["type"]))

    obj = event["data"]["object"]
    handler = _DISPATCH.get(event["type"])
    if handler is None:
        # Unknown events are persisted (idempotency) but not acted on.
        db.commit()
        return {"status": "ignored"}

    handler(obj, db)
    db.commit()
    return {"status": "ok"}


def _on_subscription_active(sub: dict, db: DbSession) -> None:
    customer_id = sub.get("customer")
    price_id = sub.get("items", {}).get("data", [{}])[0].get("price", {}).get("id", "")
    settings = get_settings()

    new_tier = Tier.free
    if price_id == settings.stripe_price_starter:
        new_tier = Tier.starter
    elif price_id == settings.stripe_price_pro:
        new_tier = Tier.pro

    _apply_tier(customer_id, new_tier, db)


def _on_subscription_cancelled(sub: dict, db: DbSession) -> None:
    _apply_tier(sub.get("customer"), Tier.free, db)


def _apply_tier(customer_id: str | None, tier: Tier, db: DbSession) -> None:
    if not customer_id:
        return
    user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
    if user:
        user.tier = tier


_DISPATCH = {
    "customer.subscription.created": _on_subscription_active,
    "customer.subscription.updated": _on_subscription_active,
    "customer.subscription.deleted": _on_subscription_cancelled,
}
