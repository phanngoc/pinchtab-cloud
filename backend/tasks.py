"""Tasks router — natural-language automation requests bound to a user's profile."""
from __future__ import annotations

import asyncio
import json
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, HttpUrl, SecretStr
from sqlalchemy import func, select
from sqlalchemy.orm import Session as DbSession

from backend.agent_runner import run_task
from backend.config import get_settings
from backend.db import get_db
from backend.denylist import evaluate
from backend.models import (
    InvalidTaskTransition,
    Profile,
    Task,
    TaskEvent,
    TaskStatus,
    User,
)
from backend.security import current_user
from backend.task_bus import END_SENTINEL, bus, registry
from backend.task_input import hint_box, registry as input_registry

router = APIRouter(prefix="/tasks", tags=["tasks"])


class CreateTaskBody(BaseModel):
    task_description: str = Field(min_length=10, max_length=2000)
    start_url: HttpUrl | None = None
    # User's Claude API key. Optional: when omitted, the backend uses the
    # operator's local `claude` CLI subscription instead (only the operator
    # email may submit without a key). When provided, must look like a real
    # Anthropic API key.
    anthropic_api_key: SecretStr | None = None


class TaskOut(BaseModel):
    id: str
    status: TaskStatus
    task_description: str
    start_url: str | None
    final_summary: str | None
    error_message: str | None
    minutes_consumed: float
    created_at: datetime
    started_at: datetime | None
    ended_at: datetime | None
    profile_id: str


def _to_out(t: Task) -> TaskOut:
    return TaskOut(
        id=t.id,
        status=t.status,
        task_description=t.task_description,
        start_url=t.start_url,
        final_summary=t.final_summary,
        error_message=t.error_message,
        minutes_consumed=t.minutes_consumed,
        created_at=t.created_at,
        started_at=t.started_at,
        ended_at=t.ended_at,
        profile_id=t.profile_id,
    )


def _get_or_create_profile(db: DbSession, user: User) -> Profile:
    """Lazy profile creation. One default profile per user; pinchtab profile
    name is a server-generated UUID to avoid user-controlled filesystem paths."""
    profile = db.scalars(select(Profile).where(Profile.user_id == user.id)).first()
    if profile is not None:
        return profile

    profile = Profile(
        user_id=user.id,
        pinchtab_profile_name=f"u_{secrets.token_hex(8)}",
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


def _ensure_under_cap(db: DbSession) -> None:
    live = db.scalar(
        select(func.count(Task.id)).where(
            Task.status.in_((TaskStatus.pending, TaskStatus.running, TaskStatus.halted))
        )
    )
    if (live or 0) >= get_settings().max_concurrent_sessions:
        raise HTTPException(status_code=503, detail="capacity_exhausted")


# ---- Dispatcher hook (test-injectable via app.state) ----


def _dispatch(task_id: str, api_key: str) -> None:
    """Spawn the agent runner as a background asyncio task.

    Held in the module-level `registry` so we can cancel on user halt.
    The api_key is captured in the closure of run_task only; it is never
    written to disk or logged.

    A done_callback consumes any exception (already persisted + published
    by run_task) to suppress asyncio's "exception was never retrieved" GC
    warning. Cancellation is not treated as an error.
    """
    import logging

    coro = run_task(task_id, anthropic_api_key=api_key)
    asyncio_task = asyncio.create_task(coro, name=f"agent:{task_id}")
    registry.register(task_id, asyncio_task)

    def _consume(fut: asyncio.Task):
        if fut.cancelled():
            return
        exc = fut.exception()
        if exc is not None:
            logging.getLogger("tasks").info(
                "task %s ended with %s: %s", task_id, type(exc).__name__, exc
            )

    asyncio_task.add_done_callback(_consume)


def dispatch_for(request: Request):
    """FastAPI dependency that returns the dispatch callable. Tests override
    this on app.state.dispatcher to inject a stub."""
    return getattr(request.app.state, "dispatcher", _dispatch)


# ---- Endpoints ----


@router.post("", response_model=TaskOut, status_code=201)
async def create_task(
    body: CreateTaskBody,
    request: Request,
    user: User = Depends(current_user),
    db: DbSession = Depends(get_db),
) -> TaskOut:
    if body.start_url is not None:
        decision = evaluate(str(body.start_url))
        if decision.blocked:
            raise HTTPException(
                status_code=400,
                detail={"error": "start_url_denied", "rule": decision.matched_rule},
            )

    _ensure_under_cap(db)

    profile = _get_or_create_profile(db, user)
    task = Task(
        user_id=user.id,
        profile_id=profile.id,
        task_description=body.task_description,
        start_url=str(body.start_url) if body.start_url else None,
        status=TaskStatus.pending,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    # Per-request LLM-backend selection:
    #   - Key provided  → AsyncAnthropic SDK with that key. Anyone may submit.
    #   - Key omitted   → ClaudeCLIProvider via local `claude` binary.
    #                     Restricted to OPERATOR_EMAIL (subscription resale guard).
    raw_key = ""
    if body.anthropic_api_key is not None:
        raw_key = body.anthropic_api_key.get_secret_value().strip()

    if not raw_key:
        from backend.llm_cli import is_operator

        if not is_operator(user.email):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "api_key_required",
                    "message": (
                        "Provide a Claude API key in `anthropic_api_key`. "
                        "Omitting the key falls back to the operator's CLI "
                        "subscription, which is restricted to the OPERATOR_EMAIL."
                    ),
                },
            )
        # api_key stays "" — signals run_task to construct ClaudeCLIProvider.
    elif len(raw_key) < 10:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "anthropic_api_key_too_short",
                "message": "anthropic_api_key must be at least 10 characters when provided.",
            },
        )

    # Dispatch. The empty string OR the real key lives only inside the
    # runner coroutine's local frame.
    api_key = raw_key
    dispatcher = dispatch_for(request)
    try:
        dispatcher(task.id, api_key)
    except Exception as e:
        task.status = TaskStatus.errored
        task.error_message = f"dispatch_failed: {e}"
        db.commit()
        raise HTTPException(status_code=500, detail="dispatch_failed")

    return _to_out(task)


@router.get("", response_model=list[TaskOut])
async def list_tasks(
    user: User = Depends(current_user), db: DbSession = Depends(get_db)
) -> list[TaskOut]:
    rows = db.scalars(
        select(Task)
        .where(Task.user_id == user.id)
        .order_by(Task.created_at.desc())
        .limit(50)
    ).all()
    return [_to_out(r) for r in rows]


@router.get("/{task_id}", response_model=TaskOut)
async def get_task(
    task_id: str, user: User = Depends(current_user), db: DbSession = Depends(get_db)
) -> TaskOut:
    t = db.get(Task, task_id)
    if t is None or t.user_id != user.id:
        raise HTTPException(status_code=404, detail="not_found")
    return _to_out(t)


# ---- Human-in-the-loop input ----


class AwaitingInputOut(BaseModel):
    prompt: str
    fields: list[dict]


@router.get("/{task_id}/awaiting-input", response_model=AwaitingInputOut | None)
async def get_awaiting_input(
    task_id: str,
    user: User = Depends(current_user),
    db: DbSession = Depends(get_db),
) -> AwaitingInputOut | None:
    """Returns the pending input request for this task, or null if none.

    Used by the dashboard to recover state after page refresh during an
    in-flight `awaiting_input` event.
    """
    t = db.get(Task, task_id)
    if t is None or t.user_id != user.id:
        raise HTTPException(status_code=404, detail="not_found")
    pending = input_registry.get(task_id)
    if pending is None:
        return None
    pv = pending.public_view()
    return AwaitingInputOut(prompt=pv["prompt"], fields=pv["fields"])


@router.post("/{task_id}/provide-input")
async def provide_input(
    task_id: str,
    body: dict,
    user: User = Depends(current_user),
    db: DbSession = Depends(get_db),
) -> dict:
    """Deliver user-supplied values to the paused runner.

    body shape: { <field_name>: <value>, ... } matching the field schema
    the runner registered. Unknown keys are dropped server-side.
    """
    t = db.get(Task, task_id)
    if t is None or t.user_id != user.id:
        raise HTTPException(status_code=404, detail="not_found")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body_must_be_object")
    ok = input_registry.provide(task_id, body)
    if not ok:
        raise HTTPException(status_code=409, detail="no_pending_input")
    return {"status": "received"}


class HintBody(BaseModel):
    message: str = Field(min_length=1, max_length=1000)


@router.post("/{task_id}/hint")
async def push_hint(
    task_id: str,
    body: HintBody,
    user: User = Depends(current_user),
    db: DbSession = Depends(get_db),
) -> dict:
    """Push a human course-correction note to a running task. The agent
    sees the hint as `[HUMAN HINT]: <text>` appended to its next user
    message — used to redirect the agent away from a stuck loop or
    clarify intent. Only accepted while the task is live (pending/running).
    """
    t = db.get(Task, task_id)
    if t is None or t.user_id != user.id:
        raise HTTPException(status_code=404, detail="not_found")
    if t.status not in (TaskStatus.pending, TaskStatus.running, TaskStatus.halted):
        raise HTTPException(
            status_code=409,
            detail=f"task is {t.status.value}; cannot accept hints",
        )
    count = hint_box.push(task_id, body.message)
    return {"status": "queued", "pending_hints": count}


@router.post("/{task_id}/halt", response_model=TaskOut)
async def halt(
    task_id: str, user: User = Depends(current_user), db: DbSession = Depends(get_db)
) -> TaskOut:
    """Request the runner to stop. The runner sees CancelledError and writes
    `halted` itself; this endpoint just signals the cancellation."""
    t = db.get(Task, task_id)
    if t is None or t.user_id != user.id:
        raise HTTPException(status_code=404, detail="not_found")

    cancelled = await registry.cancel(task_id)

    # If runner wasn't running (e.g. already terminal), reject explicit halt
    # so we don't lie about the task state.
    if not cancelled and t.status != TaskStatus.running:
        raise HTTPException(
            status_code=409, detail=f"task not running (status={t.status.value})"
        )

    # Re-read after cancel: runner should have written halted.
    db.refresh(t)
    return _to_out(t)


class HistoryEvent(BaseModel):
    id: int
    step: int | None
    kind: str
    payload: dict
    created_at: datetime


@router.get("/{task_id}/history", response_model=list[HistoryEvent])
async def get_history(
    task_id: str,
    user: User = Depends(current_user),
    db: DbSession = Depends(get_db),
    since_id: int = 0,
    limit: int = 2000,
) -> list[HistoryEvent]:
    """Persisted full event log for a task — used by the run-detail view
    to render the timeline after the SSE stream is gone. Pass `since_id`
    to fetch only events newer than a known id (cheap incremental load)."""
    t = db.get(Task, task_id)
    if t is None or t.user_id != user.id:
        raise HTTPException(status_code=404, detail="not_found")
    limit = max(1, min(int(limit), 5000))
    rows = db.execute(
        select(TaskEvent)
        .where(TaskEvent.task_id == task_id, TaskEvent.id > int(since_id))
        .order_by(TaskEvent.id.asc())
        .limit(limit)
    ).scalars().all()
    out: list[HistoryEvent] = []
    for r in rows:
        try:
            p = json.loads(r.payload_json or "{}")
            if not isinstance(p, dict):
                p = {"value": p}
        except json.JSONDecodeError:
            p = {"_raw": (r.payload_json or "")[:500]}
        out.append(HistoryEvent(
            id=r.id, step=r.step, kind=r.kind, payload=p, created_at=r.created_at,
        ))
    return out


# ---- Artifacts: per-step screenshots / snapshots from log dir ----

# Same layout as agent_runner: Path("logs") / YYYYMMDD / task_id[:8] / step-NNN.{png,snap.txt,response.json}
_LOGS_ROOT = Path("logs")
_STEP_FILE_RE = re.compile(r"^step-(\d{3})\.(png|jpg|jpeg|snap\.txt|response\.json)$")


def _task_log_dir(task_id: str) -> Path | None:
    """Find the log dir for a task. The runner creates a dated dir; we walk
    today + yesterday to handle midnight-rolled tasks."""
    short = task_id[:8]
    for date_dir in sorted(_LOGS_ROOT.glob("*"), reverse=True):
        cand = date_dir / short
        if cand.is_dir():
            return cand
    return None


class StepArtifact(BaseModel):
    step: int
    screenshot_url: str | None
    snap_url: str | None


@router.get("/{task_id}/steps", response_model=list[StepArtifact])
async def list_steps(
    task_id: str,
    user: User = Depends(current_user),
    db: DbSession = Depends(get_db),
) -> list[StepArtifact]:
    t = db.get(Task, task_id)
    if t is None or t.user_id != user.id:
        raise HTTPException(status_code=404, detail="not_found")

    log_dir = _task_log_dir(task_id)
    if log_dir is None:
        return []

    steps: dict[int, dict[str, str]] = {}
    for p in log_dir.iterdir():
        m = _STEP_FILE_RE.match(p.name)
        if not m:
            continue
        n = int(m.group(1))
        ext = m.group(2)
        d = steps.setdefault(n, {})
        if ext in ("png", "jpg", "jpeg"):
            d["screenshot"] = f"/tasks/{task_id}/steps/{n}/screenshot"
        elif ext == "snap.txt":
            d["snap"] = f"/tasks/{task_id}/steps/{n}/snap"
    return [
        StepArtifact(
            step=n,
            screenshot_url=d.get("screenshot"),
            snap_url=d.get("snap"),
        )
        for n, d in sorted(steps.items())
    ]


def _resolve_artifact(task_id: str, step: int, kind: str) -> Path:
    log_dir = _task_log_dir(task_id)
    if log_dir is None:
        raise HTTPException(status_code=404, detail="task_log_dir_missing")
    name_map = {
        "screenshot_png": f"step-{step:03d}.png",
        "screenshot_jpg": f"step-{step:03d}.jpg",
        "snap": f"step-{step:03d}.snap.txt",
    }
    # Try preferred name then alternates.
    candidates = [name_map[kind]] if kind in name_map else []
    if kind == "screenshot":
        candidates = [f"step-{step:03d}.png", f"step-{step:03d}.jpg"]
    for fname in candidates:
        p = log_dir / fname
        if p.exists():
            return p
    raise HTTPException(status_code=404, detail="artifact_not_found")


@router.get("/{task_id}/steps/{step}/screenshot")
async def get_step_screenshot(
    task_id: str,
    step: int,
    user: User = Depends(current_user),
    db: DbSession = Depends(get_db),
):
    t = db.get(Task, task_id)
    if t is None or t.user_id != user.id:
        raise HTTPException(status_code=404, detail="not_found")
    path = _resolve_artifact(task_id, step, "screenshot")
    # Magic-byte sniff for content type.
    head = path.read_bytes()[:8]
    media = "image/png" if head.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg"
    return FileResponse(str(path), media_type=media)


@router.get("/{task_id}/steps/{step}/snap")
async def get_step_snap(
    task_id: str,
    step: int,
    user: User = Depends(current_user),
    db: DbSession = Depends(get_db),
):
    t = db.get(Task, task_id)
    if t is None or t.user_id != user.id:
        raise HTTPException(status_code=404, detail="not_found")
    path = _resolve_artifact(task_id, step, "snap")
    return FileResponse(str(path), media_type="text/plain; charset=utf-8")


@router.get("/{task_id}/stream")
async def stream(
    task_id: str,
    request: Request,
    user: User = Depends(current_user),
    db: DbSession = Depends(get_db),
):
    """SSE stream of progress events for a task. Closes when the runner
    reaches a terminal state (END_SENTINEL) or the client disconnects."""
    t = db.get(Task, task_id)
    if t is None or t.user_id != user.id:
        raise HTTPException(status_code=404, detail="not_found")

    async def event_source() -> AsyncIterator[str]:
        async with bus.subscribe(task_id) as q:
            # Send an initial 'open' event so the client knows the stream
            # is alive even if no events arrive for a while.
            yield f"event: open\ndata: {json.dumps({'task_id': task_id})}\n\n"
            while True:
                if await request.is_disconnected():
                    return
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield "event: keepalive\ndata: {}\n\n"
                    continue
                if event.get("type") == END_SENTINEL:
                    yield "event: end\ndata: {}\n\n"
                    return
                evt_type = event.get("type", "message")
                yield f"event: {evt_type}\ndata: {json.dumps(event)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
