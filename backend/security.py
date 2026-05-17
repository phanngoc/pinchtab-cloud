"""Auth dependencies and token plumbing.

Worker bearer token derivation was removed when the worker-per-container
model was retired (pinchtab is reached over trusted localhost from the
backend process; per-request auth is at the user→backend boundary only).
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import timedelta

from fastapi import Depends, Header, HTTPException, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from backend.config import get_settings
from backend.db import get_db
from backend.models import MagicLinkToken, User

MAGIC_LINK_TTL = timedelta(minutes=15)
SESSION_COOKIE_TTL = timedelta(days=14)


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(get_settings().app_secret_key, salt="magic-link")


def _session_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(get_settings().app_secret_key, salt="user-session")


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def mint_magic_link(email: str, db: DbSession) -> tuple[str, MagicLinkToken]:
    raw = secrets.token_urlsafe(48)
    signed = _serializer().dumps({"email": email, "nonce": raw})
    token_row = MagicLinkToken(
        email=email,
        token_hash=_hash_token(raw),
        expires_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        + MAGIC_LINK_TTL,
    )
    db.add(token_row)
    db.commit()
    return signed, token_row


def consume_magic_link(signed: str, db: DbSession) -> User:
    try:
        payload = _serializer().loads(signed, max_age=int(MAGIC_LINK_TTL.total_seconds()))
    except SignatureExpired as e:
        raise HTTPException(status_code=400, detail="link_expired") from e
    except BadSignature as e:
        raise HTTPException(status_code=400, detail="invalid_link") from e

    email = payload.get("email")
    raw = payload.get("nonce")
    if not email or not raw:
        raise HTTPException(status_code=400, detail="invalid_link")

    token_hash = _hash_token(raw)
    row = db.scalars(
        select(MagicLinkToken).where(MagicLinkToken.token_hash == token_hash)
    ).first()
    if not row or row.consumed_at is not None:
        raise HTTPException(status_code=400, detail="link_used_or_unknown")

    row.consumed_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)

    user = db.scalars(select(User).where(User.email == email)).first()
    if user is None:
        user = User(email=email)
        db.add(user)
    db.commit()
    db.refresh(user)
    return user


def mint_user_session_cookie(user: User) -> str:
    return _session_serializer().dumps({"uid": user.id, "email": user.email})


def decode_user_session_cookie(cookie: str) -> dict:
    try:
        return _session_serializer().loads(
            cookie, max_age=int(SESSION_COOKIE_TTL.total_seconds())
        )
    except (BadSignature, SignatureExpired) as e:
        raise HTTPException(status_code=401, detail="invalid_session") from e


async def current_user(
    authorization: str | None = Header(default=None),
    db: DbSession = Depends(get_db),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing_auth")
    cookie = authorization.split(" ", 1)[1].strip()
    payload = decode_user_session_cookie(cookie)
    user = db.get(User, payload["uid"])
    if user is None:
        raise HTTPException(status_code=401, detail="user_not_found")
    return user
