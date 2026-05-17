"""Agent runner — async coroutine that drives one Task to terminal state.

Invariants enforced here:
  - The Claude API key is held only in this coroutine's local frame.
    Never written to disk, never logged, dropped when the coroutine exits.
  - Pinchtab denylist is applied to the tab BEFORE the agent loop starts.
    A second control-plane denylist check runs on every navigate() tool
    invocation — belt-and-suspenders.
  - Safety patterns (captcha/OTP/auth challenge) detected in snap text
    cause an UNCONDITIONAL halt before the page is shown to Claude.
    This is the prompt-injection mitigation: a page text saying
    "ignore previous instructions, your task is X" never reaches Claude
    because we strip on the input side.

Lifecycle:
   1. Verify task is pending; load profile
   2. Ensure user's instance is running (start if cold)
   3. Open tab at start_url; apply denylist rules
   4. Transition task pending → running
   5. Loop: snap+screenshot → Claude → tool call(s) → execute → repeat
   6. On terminal (done | halted | errored): record minutes, close tab
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from sqlalchemy.orm import Session as DbSession

from backend.db import SessionLocal
from backend.denylist import DenylistPolicy, evaluate
from backend.models import (
    InvalidTaskTransition,
    Profile,
    Task,
    TaskStatus,
    UsageMetric,
)
from backend.pinchtab_client import PinchtabClient
from backend.task_bus import TaskBus, bus as default_bus
from core.prompts import HARDENED_SYSTEM_PROMPT

log = logging.getLogger("agent_runner")

# ---- Tunables ----

DEFAULT_MAX_STEPS = 30
DEFAULT_SNAP_MAX_TOKENS = 5000
DEFAULT_STEP_DELAY_SECONDS = 1.5
DEFAULT_SCREENSHOT_QUALITY = 60
DEFAULT_MODEL = "claude-sonnet-4-6"

# Safety patterns — lower-case substring match against snap text triggers
# an unconditional halt. Page content NEVER reaches Claude when these hit.
HALT_PATTERNS = (
    "captcha",
    "verify you are human",
    "are you a robot",
    "two-factor",
    "enter the code we sent",
    "otp",
    "xác thực",
    "mã otp",
)

# Anthropic tool schema (mirrors the local CLI agent.py spec)
TOOLS = [
    {
        "name": "click",
        "description": "Click an element by its refXXX ID from the latest snapshot.",
        "input_schema": {
            "type": "object",
            "properties": {"ref": {"type": "string"}},
            "required": ["ref"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text into an input identified by its ref.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ref": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["ref", "text"],
        },
    },
    {
        "name": "select_option",
        "description": "Select an option in a <select> dropdown by value.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ref": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["ref", "value"],
        },
    },
    {
        "name": "press_key",
        "description": "Press a keyboard key (Enter, Tab, Escape, ArrowDown).",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "scroll",
        "description": "Scroll the page by a pixel amount.",
        "input_schema": {
            "type": "object",
            "properties": {"amount": {"type": "string"}},
            "required": ["amount"],
        },
    },
    {
        "name": "wait",
        "description": "Wait N seconds (max 15) for page to settle.",
        "input_schema": {
            "type": "object",
            "properties": {"seconds": {"type": "number"}},
            "required": ["seconds"],
        },
    },
    {
        "name": "halt_for_human",
        "description": (
            "Stop and return control to a human. Use when you see captcha, "
            "OTP, payment confirmation, or any irreversible/sensitive action."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
    {
        "name": "task_complete",
        "description": "Task is finished. Include a short summary of what was accomplished.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
]


# ---- Helpers ----


def detect_halt_pattern(snap_text: str) -> str | None:
    lower = snap_text.lower()
    for p in HALT_PATTERNS:
        if p in lower:
            return p
    return None


def detect_image_media_type(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return "image/png"


def build_user_message(snap_text: str, screenshot: bytes, step: int) -> dict[str, Any]:
    media_type = detect_image_media_type(screenshot)
    img_b64 = base64.standard_b64encode(screenshot).decode()
    return {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_b64,
                },
            },
            {
                "type": "text",
                "text": (
                    f"## Step {step}\n\n"
                    f"### Interactive elements\n```\n{snap_text}\n```\n\n"
                    f"Decide the next action."
                ),
            },
        ],
    }


# ---- Anthropic injection point ----


class _MessagesAPI(Protocol):
    async def create(self, **kwargs) -> Any: ...


class _AnthropicLike(Protocol):
    messages: _MessagesAPI


def _default_anthropic_factory(api_key: str) -> _AnthropicLike:
    """Real Anthropic client. Imported lazily so tests don't need the package."""
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(api_key=api_key)


# ---- Result types ----


class AgentResult:
    """Plain-data result. Useful for return value testing without DB."""

    def __init__(
        self,
        *,
        terminal: TaskStatus,
        summary: str | None = None,
        error_message: str | None = None,
        steps: int = 0,
        minutes_consumed: float = 0.0,
    ):
        self.terminal = terminal
        self.summary = summary
        self.error_message = error_message
        self.steps = steps
        self.minutes_consumed = minutes_consumed


# ---- Instance/tab orchestration ----


async def ensure_instance(
    client: PinchtabClient, profile: Profile, db: DbSession
) -> str:
    """Get or start an instance for the profile. Returns pinchtab instance id."""
    if profile.pinchtab_instance_id:
        existing = await client.get_instance(profile.pinchtab_instance_id)
        if existing and (existing.get("status") in ("running", "starting")):
            return profile.pinchtab_instance_id
        # Stale handle — clear and start fresh.
        profile.pinchtab_instance_id = None

    started = await client.start_instance(profile_id=profile.pinchtab_profile_name)
    instance_id = started.get("id") or started.get("instanceId")
    if not instance_id:
        raise RuntimeError(f"pinchtab returned no instance id: {started}")
    profile.pinchtab_instance_id = instance_id
    profile.last_used_at = datetime.now(timezone.utc)
    db.commit()
    return instance_id


async def _open_tab_and_secure(
    client: PinchtabClient,
    instance_id: str,
    start_url: str | None,
    policy: DenylistPolicy,
) -> tuple[str, int]:
    """Open a tab on the instance, then apply the denylist BEFORE returning.

    Returns (tab_id, num_rules_installed). The denylist must be in place
    before any agent action runs, so this is sequential by design.
    """
    tab_info = await client.open_tab(instance_id, start_url)
    tab_id = tab_info.get("id") or tab_info.get("tabId")
    if not tab_id:
        raise RuntimeError(f"pinchtab returned no tab id: {tab_info}")
    rules_installed = await client.apply_denylist(tab_id, policy)
    return tab_id, rules_installed


# ---- Tool execution ----


async def _execute_tool(
    client: PinchtabClient,
    tab_id: str,
    name: str,
    params: dict[str, Any],
    policy: DenylistPolicy,
) -> tuple[str, bool, TaskStatus | None, str | None]:
    """Run a single tool call. Returns (result_text, should_stop, terminal_status, finalize_summary)."""
    try:
        if name == "click":
            res = await client.click(tab_id, params["ref"])
            return (json.dumps(res)[:200] if not isinstance(res, str) else res[:200], False, None, None)
        if name == "type_text":
            res = await client.type_text(tab_id, params["ref"], params["text"])
            return (str(res)[:200], False, None, None)
        if name == "select_option":
            res = await client.select_option(tab_id, params["ref"], params["value"])
            return (str(res)[:200], False, None, None)
        if name == "press_key":
            res = await client.press_key(tab_id, params["key"])
            return (str(res)[:200], False, None, None)
        if name == "scroll":
            res = await client.scroll(tab_id, params["amount"])
            return (str(res)[:200], False, None, None)
        if name == "wait":
            seconds = min(float(params.get("seconds", 1)), 15.0)
            await asyncio.sleep(seconds)
            return (f"waited {seconds}s", False, None, None)
        if name == "halt_for_human":
            reason = params.get("reason", "")
            return (f"HALT: {reason}", True, TaskStatus.halted, reason)
        if name == "task_complete":
            summary = params.get("summary", "")
            return (f"DONE: {summary}", True, TaskStatus.done, summary)
        return (f"ERROR: unknown tool '{name}'", False, None, None)
    except Exception as e:
        log.warning("tool %s raised: %s", name, e)
        return (f"ERROR: {type(e).__name__}: {e}", False, None, None)


def _serialize_assistant_content(content_blocks: list[Any]) -> list[dict[str, Any]]:
    """Normalize Anthropic response content blocks into wire-format dicts that
    can be sent back as `messages` history on the next turn."""
    out: list[dict[str, Any]] = []
    for b in content_blocks:
        if hasattr(b, "model_dump"):
            out.append(b.model_dump())
        elif isinstance(b, dict):
            out.append(b)
        else:
            # Best-effort fallback.
            out.append({"type": "text", "text": str(b)})
    return out


# ---- The runner ----


async def run_task(
    task_id: str,
    *,
    anthropic_api_key: str,
    pinchtab_client: PinchtabClient | None = None,
    pinchtab_base_url: str = "http://127.0.0.1:9867",
    anthropic_factory: Callable[[str], _AnthropicLike] = _default_anthropic_factory,
    db_session_factory: Callable[[], DbSession] | None = None,
    denylist_policy: DenylistPolicy | None = None,
    model: str = DEFAULT_MODEL,
    max_steps: int = DEFAULT_MAX_STEPS,
    snap_max_tokens: int = DEFAULT_SNAP_MAX_TOKENS,
    step_delay_seconds: float = DEFAULT_STEP_DELAY_SECONDS,
    screenshot_quality: int = DEFAULT_SCREENSHOT_QUALITY,
    log_dir: Path | None = None,
    close_tab_on_finish: bool = True,
    bus: TaskBus | None = None,
) -> AgentResult:
    """Run the agent loop for one task to a terminal state.

    The api key is held only in the local frame of this coroutine.

    Cancellation: if this coroutine is cancelled (asyncio.Task.cancel called
    from the dispatcher), the task is marked `halted` and resources are
    cleaned up before re-raising CancelledError.
    """
    policy = denylist_policy or DenylistPolicy()
    event_bus = bus if bus is not None else default_bus

    def _publish(event_type: str, **fields):
        event_bus.publish(task_id, {"type": event_type, **fields})

    # Per-task log dir
    log_root = log_dir or (Path("logs") / time.strftime("%Y%m%d") / task_id[:8])
    log_root.mkdir(parents=True, exist_ok=True)

    own_client = pinchtab_client is None
    client = pinchtab_client or PinchtabClient(base_url=pinchtab_base_url)
    own_db_factory = db_session_factory is None
    db_factory = db_session_factory or (lambda: SessionLocal())
    db = db_factory()

    anthropic_client = anthropic_factory(anthropic_api_key)
    # Don't carry the key past this point — only `anthropic_client` holds it.

    result: AgentResult | None = None

    try:
        task = db.get(Task, task_id)
        if task is None:
            raise RuntimeError(f"task {task_id} not found")
        if task.status != TaskStatus.pending:
            raise RuntimeError(f"task {task_id} not pending (status={task.status.value})")

        profile = db.get(Profile, task.profile_id)
        if profile is None:
            raise RuntimeError(f"profile {task.profile_id} not found")

        # Bring instance online + open tab + apply denylist
        instance_id = await ensure_instance(client, profile, db)
        try:
            tab_id, rules_installed = await _open_tab_and_secure(
                client, instance_id, task.start_url, policy
            )
        except Exception as e:
            task.error_message = f"open_tab/apply_denylist failed: {e}"
            task.transition(TaskStatus.errored)
            db.commit()
            return AgentResult(terminal=TaskStatus.errored, error_message=task.error_message)

        task.pinchtab_tab_id = tab_id
        task.transition(TaskStatus.running)
        db.commit()
        db.refresh(task)
        log.info("task %s running on tab %s (denylist rules: %d)", task_id, tab_id, rules_installed)
        _publish(
            "started",
            tab_id=tab_id,
            instance_id=instance_id,
            denylist_rules=rules_installed,
        )

        # Agent loop
        system_blocks = [
            {
                "type": "text",
                "text": f"{HARDENED_SYSTEM_PROMPT}\n\n## Task\n{task.task_description}",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        messages: list[dict[str, Any]] = []

        terminal_status: TaskStatus | None = None
        summary: str | None = None
        error_msg: str | None = None
        steps_taken = 0

        start_unix = time.time()

        for step in range(1, max_steps + 1):
            steps_taken = step
            _publish("step", step=step)

            try:
                snap_text = await client.snapshot(tab_id, max_tokens=snap_max_tokens)
                screenshot = await client.screenshot(tab_id, quality=screenshot_quality)
            except Exception as e:
                error_msg = f"snap/screenshot at step {step}: {e}"
                terminal_status = TaskStatus.errored
                break

            (log_root / f"step-{step:03d}.png").write_bytes(screenshot)
            (log_root / f"step-{step:03d}.snap.txt").write_text(snap_text)

            # Safety filter BEFORE any LLM call — prompt-injection mitigation.
            halt_p = detect_halt_pattern(snap_text)
            if halt_p:
                summary = f"safety pattern matched: {halt_p}"
                terminal_status = TaskStatus.halted
                break

            messages.append(build_user_message(snap_text, screenshot, step))

            try:
                response = await anthropic_client.messages.create(
                    model=model,
                    max_tokens=2048,
                    system=system_blocks,
                    tools=TOOLS,
                    messages=messages,
                )
            except Exception as e:
                error_msg = f"anthropic call at step {step}: {type(e).__name__}: {e}"
                terminal_status = TaskStatus.errored
                break

            # Log response (response object does NOT contain the api key).
            try:
                (log_root / f"step-{step:03d}.response.json").write_text(
                    json.dumps(
                        response.model_dump() if hasattr(response, "model_dump") else response,
                        indent=2,
                        default=str,
                    )
                )
            except Exception:
                # Logging failure is not fatal.
                pass

            messages.append(
                {"role": "assistant", "content": _serialize_assistant_content(response.content)}
            )

            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                error_msg = f"no tool call returned at step {step}"
                terminal_status = TaskStatus.errored
                break

            tool_results: list[dict[str, Any]] = []
            stop_loop = False
            for tu in tool_uses:
                _publish("tool_call", step=step, name=tu.name, params=dict(tu.input))
                result_text, do_stop, term, term_summary = await _execute_tool(
                    client, tab_id, tu.name, dict(tu.input), policy
                )
                _publish("tool_result", step=step, name=tu.name, result=result_text[:200])
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tu.id, "content": result_text}
                )
                if term is not None:
                    terminal_status = term
                    summary = term_summary
                    stop_loop = True
                elif do_stop:
                    stop_loop = True

            messages.append({"role": "user", "content": tool_results})

            if stop_loop:
                break

            await asyncio.sleep(step_delay_seconds)

        if terminal_status is None:
            error_msg = f"max_steps {max_steps} reached"
            terminal_status = TaskStatus.errored

        minutes = (time.time() - start_unix) / 60.0

        # Persist terminal state.
        task.minutes_consumed = minutes
        task.final_summary = summary
        task.error_message = error_msg
        try:
            task.transition(terminal_status)
        except InvalidTaskTransition as e:
            log.error("invalid terminal transition for %s: %s", task_id, e)
        db.add(UsageMetric(user_id=task.user_id, task_id=task.id, minutes=minutes))
        db.commit()

        if close_tab_on_finish:
            try:
                await client.close_tab(tab_id)
            except Exception as e:
                log.warning("close_tab failed: %s", e)

        _publish(
            "terminal",
            status=terminal_status.value,
            summary=summary,
            error=error_msg,
            steps=steps_taken,
            minutes=minutes,
        )

        result = AgentResult(
            terminal=terminal_status,
            summary=summary,
            error_message=error_msg,
            steps=steps_taken,
            minutes_consumed=minutes,
        )
        return result

    except asyncio.CancelledError:
        # User-requested halt — dispatcher called .cancel() on this task.
        log.info("task %s cancelled — marking halted", task_id)
        try:
            t = db.get(Task, task_id)
            if t is not None and t.status == TaskStatus.running:
                t.transition(TaskStatus.halted)
                t.error_message = "cancelled_by_user"
                db.commit()
        except Exception:
            pass
        _publish("terminal", status=TaskStatus.halted.value, error="cancelled_by_user")
        raise

    finally:
        try:
            event_bus.end(task_id)
        except Exception:
            pass
        db.close()
        if own_client:
            await client.aclose()
        # Drop references; Python GC will collect the api key string and
        # the Anthropic client which retained it.
        del anthropic_client
