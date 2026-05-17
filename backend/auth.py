"""Magic link auth router."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session as DbSession

from backend.config import get_settings
from backend.db import get_db
from backend.security import (
    consume_magic_link,
    mint_magic_link,
    mint_user_session_cookie,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class RequestLinkBody(BaseModel):
    email: EmailStr


class VerifyLinkBody(BaseModel):
    token: str


class LoginResponse(BaseModel):
    bearer: str
    email: str


@router.post("/request-link")
async def request_link(body: RequestLinkBody, db: DbSession = Depends(get_db)) -> dict:
    signed, _row = mint_magic_link(body.email, db)
    base = get_settings().app_base_url.rstrip("/")
    link = f"{base}/auth/verify?token={signed}"
    # Resend integration is stubbed; in production this enqueues an email send.
    # For dev we return the link directly so the operator can paste it.
    if get_settings().app_env == "production":
        # TODO: enqueue Resend send job; do NOT return link in prod responses
        return {"status": "sent"}
    return {"status": "sent", "dev_link": link}


@router.post("/verify")
async def verify(body: VerifyLinkBody, db: DbSession = Depends(get_db)) -> LoginResponse:
    user = consume_magic_link(body.token, db)
    cookie = mint_user_session_cookie(user)
    return LoginResponse(bearer=cookie, email=user.email)


@router.get("/verify")
async def verify_get(token: str, request: Request, db: DbSession = Depends(get_db)) -> LoginResponse:
    """GET form for click-from-email flow; returns the same JSON shape."""
    user = consume_magic_link(token, db)
    cookie = mint_user_session_cookie(user)
    return LoginResponse(bearer=cookie, email=user.email)
