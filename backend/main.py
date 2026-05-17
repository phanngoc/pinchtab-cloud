"""FastAPI app entrypoint.

Day-30 minimum architecture (CEO review HOLD-mode):
  - Single VPS in Singapore
  - SQLite + Alembic migrations
  - Stripe at launch; VNPay deferred to v1.1
  - Per-user 60 req/min rate limit
  - Browser worker pool: separate process(es) on same host

Run:
  uvicorn backend.main:app --reload
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from backend.auth import router as auth_router
from backend.billing import router as billing_router
from backend.config import get_settings
from backend.db import Base, engine
from backend.pinchtab_client import PinchtabClient
from backend.profiles import router as profiles_router
from backend.rate_limit import check as rate_check
from backend.tasks import router as tasks_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("pinchtab")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Dev-mode auto-create. Production uses Alembic migrations.
    if settings.app_env != "production":
        Base.metadata.create_all(bind=engine)

    # Pinchtab client lives for the app's lifetime; reused across requests.
    # Tests override via `app.state.pinchtab = FakeClient()` before requests.
    if not hasattr(app.state, "pinchtab"):
        app.state.pinchtab = PinchtabClient(base_url=settings.worker_base_url)

    try:
        yield
    finally:
        client = getattr(app.state, "pinchtab", None)
        if client is not None and isinstance(client, PinchtabClient):
            await client.aclose()


app = FastAPI(
    title="Pinchtab Cloud SG",
    version="0.1.0",
    description="Hosted vision-LLM browser automation. VN-first.",
    lifespan=lifespan,
)


@app.middleware("http")
async def per_user_rate_limit(request: Request, call_next):
    """Apply per-user rate limit only on authenticated endpoints. Auth and
    webhook routes bypass."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        # Cheap-ish: parse user id from cookie token without DB round-trip.
        try:
            from backend.security import decode_user_session_cookie

            payload = decode_user_session_cookie(auth_header.split(" ", 1)[1].strip())
            uid = payload.get("uid")
            if uid and not rate_check(uid):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "rate_limited"},
                )
        except HTTPException:
            # Invalid token is handled by the actual endpoint dependency.
            pass

    response = await call_next(request)
    return response


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """Prometheus-friendly text output (CEO review O1).

    Counters/gauges populated by worker callbacks and session lifecycle hooks.
    Stub for now — wire into a real collector in subsequent turns.
    """
    return (
        "# HELP pinchtab_sessions_total Total sessions created\n"
        "# TYPE pinchtab_sessions_total counter\n"
        "pinchtab_sessions_total 0\n"
        "# HELP pinchtab_sessions_active Current active sessions\n"
        "# TYPE pinchtab_sessions_active gauge\n"
        "pinchtab_sessions_active 0\n"
        "# HELP pinchtab_browser_minutes_total Cumulative browser-minutes consumed\n"
        "# TYPE pinchtab_browser_minutes_total counter\n"
        "pinchtab_browser_minutes_total 0\n"
    )


app.include_router(auth_router)
app.include_router(tasks_router)
app.include_router(profiles_router)
app.include_router(billing_router)
