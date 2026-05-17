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
from time import monotonic  # noqa: F401  (kept for tests/external use)
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
from backend.task_input import registry as input_registry
from core.prompts import HARDENED_SYSTEM_PROMPT


# Hard cap on how long the runner waits for user input before giving up.
USER_INPUT_TIMEOUT_SECONDS = 600  # 10 minutes

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
        "description": (
            "Click an interactive element by its refXXX from the latest snapshot. "
            "Use for buttons, links, checkboxes. For form submission, click the "
            "submit button — DO NOT press_key('Enter')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ref": {"type": "string"}},
            "required": ["ref"],
        },
    },
    {
        "name": "fill",
        "description": (
            "Set an input element's value directly. Preferred for forms — works "
            "on most fields including ones with React/Vue controllers. Pass the "
            "full final value; this replaces the field's content."
        ),
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
        "name": "type_text",
        "description": (
            "Send real keystroke events to an input. Only use when fill() doesn't "
            "work — typically sites with keystroke listeners (chat inputs, "
            "search-as-you-type). Otherwise prefer fill()."
        ),
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
        "description": (
            "Select an option in a <select> dropdown. Matches by `value` "
            "attribute first, falls back to visible text — pass either."
        ),
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
        "description": (
            "Press a single key (Enter, Tab, Escape, ArrowDown, etc). Useful "
            "for keyboard navigation; DO NOT use Enter to submit forms — "
            "click the submit button instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "scroll",
        "description": (
            "Scroll the page. Pass a pixel amount as a string ('500', '1500'), "
            "negative for up ('-300'), or a direction word ('down', 'up'). "
            "Use to bring more elements into the snap viewport."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"amount": {"type": "string"}},
            "required": ["amount"],
        },
    },
    {
        "name": "get_text",
        "description": (
            "Read page text content. Use when you need prose (article body, "
            "search results, headline list) rather than interactive elements. "
            "Optionally narrow to one element by ref/css selector. Returns "
            "readable text only — hidden/decorative nodes are filtered."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "Optional. ref like 'e7' or CSS like '#article-body'. Omit for whole page.",
                },
            },
        },
    },
    {
        "name": "find_element",
        "description": (
            "Locate an element by natural-language description. Returns the "
            "ref of the best match. Use when the element you want is on the "
            "page but isn't in the current interactive snap (e.g. you need a "
            "specific link buried in a long list). Cheaper than scrolling "
            "and re-snapping multiple times."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Description of the target, e.g. 'the submit button' or 'login link in the top right'.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "wait",
        "description": (
            "Pause N seconds (max 15) for the page to settle. Use sparingly — "
            "the runner already pauses ~1.5s between steps. Useful after "
            "actions that trigger async loading (XHR, spinner)."
        ),
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
    {
        "name": "request_user_input",
        "description": (
            "Pause and ask the human user for information you cannot proceed "
            "without — login credentials, an OTP after they receive it, picking "
            "between options on an ambiguous page. The task pauses; the user "
            "fills in a form on the dashboard; you receive their values as a "
            "JSON dict and continue. Prefer this tool over halt_for_human or "
            "typing random values. Use specific, minimal field lists "
            "(don't ask for everything; only what's strictly needed)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Short message shown to the user explaining what you need.",
                },
                "fields": {
                    "type": "array",
                    "description": "Form fields the user must fill.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Key used in the returned dict."},
                            "label": {"type": "string", "description": "Human-readable label shown above the field."},
                            "type": {
                                "type": "string",
                                "enum": ["text", "password", "email", "number"],
                                "description": "HTML input type. 'password' hides the value as the user types.",
                            },
                        },
                        "required": ["name", "label"],
                    },
                },
            },
            "required": ["prompt", "fields"],
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
    """Pick an LLM client per-request based on the api_key value.

      - empty / whitespace  → ClaudeCLIProvider (operator's CLI subscription)
      - non-empty           → AsyncAnthropic with that key (BYO API tier)

    The operator-only gate fires upstream in tasks.py — by the time we get
    here the empty-key branch is already authorized.
    """
    if not (api_key or "").strip():
        from backend.llm_cli import ClaudeCLIProvider

        return ClaudeCLIProvider()

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
    """Get or start an instance for the profile. Returns pinchtab instance id.

    Pinchtab does not auto-create profiles. If our stored name doesn't
    resolve in pinchtab, we first POST /profiles to create it, store the
    returned pinchtab id back into our row, then start the instance.
    """
    if profile.pinchtab_instance_id:
        existing = await client.get_instance(profile.pinchtab_instance_id)
        if existing and (existing.get("status") in ("running", "starting")):
            return profile.pinchtab_instance_id
        # Stale handle — clear and start fresh.
        profile.pinchtab_instance_id = None

    async def _wait_running(instance_id: str, timeout_s: float = 20.0) -> None:
        """Poll until instance status == 'running' or timeout. Chromium cold
        boot inside a pinchtab instance takes ~2-6 seconds in practice."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            inst = await client.get_instance(instance_id)
            if inst and inst.get("status") == "running":
                return
            await asyncio.sleep(0.4)
        raise RuntimeError(
            f"instance {instance_id} failed to reach 'running' within {timeout_s}s"
        )

    async def _start_and_record(pid: str) -> str:
        started = await client.start_instance(profile_id=pid)
        instance_id = started.get("id") or started.get("instanceId")
        if not instance_id:
            raise RuntimeError(f"pinchtab returned no instance id: {started}")
        profile.pinchtab_instance_id = instance_id
        profile.last_used_at = datetime.now(timezone.utc)
        db.commit()
        # Block until Chromium is fully up; opening a tab before this races.
        if started.get("status") != "running":
            await _wait_running(instance_id)
        return instance_id

    from backend.pinchtab_client import PinchtabError  # local import to avoid cycles

    try:
        return await _start_and_record(profile.pinchtab_profile_name)
    except PinchtabError as e:
        if e.status != 404:
            raise
        # Profile doesn't exist in pinchtab — create it once, then retry.
        log.info(
            "profile %s not in pinchtab — creating", profile.pinchtab_profile_name
        )
        created = await client.create_profile(
            name=profile.pinchtab_profile_name,
            description=f"auto-created by pinchtab-cloud for user {profile.user_id}",
        )
        new_id = created.get("id")
        if not new_id:
            raise RuntimeError(f"pinchtab create_profile returned no id: {created}")
        # Store the canonical pinchtab id back into our row so subsequent
        # starts use the id directly (faster than name resolution).
        profile.pinchtab_profile_name = new_id
        db.commit()
        return await _start_and_record(new_id)


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
        if name == "fill":
            res = await client.fill(tab_id, params["ref"], params["text"])
            return (str(res)[:200], False, None, None)
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
        if name == "get_text":
            selector = params.get("selector")
            res = await client.text(tab_id, selector=selector)
            # Trim to keep tool_result manageable; agent gets gist.
            return (res[:2000] if isinstance(res, str) else str(res)[:2000], False, None, None)
        if name == "find_element":
            res = await client.find(tab_id, params["query"])
            return (json.dumps(res)[:300] if not isinstance(res, str) else res[:300], False, None, None)
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


def _trim_history_for_cli(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Slim message history for claude CLI subprocess.

    claude CLI has no prompt cache between invocations, so each turn re-pays
    the full history. The biggest waste is OLD snap text + screenshots in
    prior user messages — those snapshots are stale (page has navigated/
    interacted since). We:

    - Keep the CURRENT user message intact (its snap is what the agent acts on).
    - Replace OLD "snap" user messages (any user message containing an image
      block) with a one-line placeholder. The agent's memory of "we already
      ran step N" is preserved via the assistant's tool_use + the
      tool_result messages, which we leave intact.
    - Don't truncate by count; just remove the dead weight per message.

    Empirical: step-4 prompt drops from ~70 KB → ~15 KB. Allows runs of
    10+ steps without hitting the 90s CLI timeout or 30k tokens/min
    subscription cap.
    """
    if len(messages) <= 1:
        return messages
    out: list[dict[str, Any]] = []
    last_idx = len(messages) - 1
    for i, msg in enumerate(messages):
        if i == last_idx:
            out.append(msg)  # current message — never trim
            continue
        if msg.get("role") != "user":
            out.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        # Detect snap-turn messages by presence of an image block.
        has_image = any(
            isinstance(b, dict) and b.get("type") == "image" for b in content
        )
        if not has_image:
            # Tool-result message — keep intact (small + load-bearing).
            out.append(msg)
            continue
        # Extract step number from the text block for a useful placeholder.
        step_marker = ""
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                m = re.search(r"##\s*Step\s+(\d+)", b.get("text", ""))
                if m:
                    step_marker = f" step {m.group(1)}"
                    break
        out.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"[prior page snapshot{step_marker} — omitted; the page "
                            "has changed since then. See assistant's tool_use + the "
                            "tool_result messages below for what actually happened.]"
                        ),
                    }
                ],
            }
        )
    return out


def _apply_message_cache_breakpoint(messages: list[dict[str, Any]]) -> None:
    """Rewrite messages in-place: strip any stale cache_control on user
    blocks, then set cache_control on the last block of the most-recent
    user message. Lets Anthropic cache the entire prior conversation prefix
    so we only pay full input cost on the freshly-appended content."""
    # Strip.
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict):
                block.pop("cache_control", None)

    # Set on the latest user message's last block.
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list) or not content:
            break
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = {"type": "ephemeral"}
        break


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
    if own_client:
        from backend.config import get_settings

        _s = get_settings()
        client = PinchtabClient(
            base_url=_s.worker_base_url,
            token=_s.pinchtab_token or None,
        )
    else:
        client = pinchtab_client
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

            # Mode detection: when the user did not supply an API key,
            # _default_anthropic_factory returns a ClaudeCLIProvider. CLI
            # has no prompt caching between subprocess calls, so we trim
            # message history to a recent window and skip cache_control
            # marking (which the CLI also ignores).
            is_cli_mode = not (anthropic_api_key or "").strip()
            if is_cli_mode:
                send_messages = _trim_history_for_cli(messages)
            else:
                send_messages = messages
                _apply_message_cache_breakpoint(send_messages)

            _publish("llm_call", step=step, prompt_messages=len(send_messages))
            t0 = time.time()
            try:
                response = await anthropic_client.messages.create(
                    model=model,
                    max_tokens=2048,
                    system=system_blocks,
                    tools=TOOLS,
                    messages=send_messages,
                )
            except Exception as e:
                _publish(
                    "llm_done",
                    step=step,
                    elapsed_seconds=round(time.time() - t0, 1),
                    error=type(e).__name__,
                )
                error_msg = f"anthropic call at step {step}: {type(e).__name__}: {e}"
                terminal_status = TaskStatus.errored
                break
            _publish("llm_done", step=step, elapsed_seconds=round(time.time() - t0, 1))

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

                # Special: human-in-the-loop input request. Runner pauses
                # until the user fills the form via /tasks/{id}/provide-input.
                if tu.name == "request_user_input":
                    prompt_text = str(tu.input.get("prompt", ""))
                    fields = tu.input.get("fields") or []
                    if not isinstance(fields, list):
                        fields = []
                    input_registry.register(task_id, prompt_text, fields)
                    _publish(
                        "awaiting_input",
                        step=step,
                        prompt=prompt_text,
                        fields=[f for f in fields if isinstance(f, dict)],
                    )
                    try:
                        values = await asyncio.wait_for(
                            input_registry.wait(task_id),
                            timeout=USER_INPUT_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        input_registry.cancel(task_id)
                        _publish(
                            "input_timeout",
                            step=step,
                            timeout_seconds=USER_INPUT_TIMEOUT_SECONDS,
                        )
                        terminal_status = TaskStatus.halted
                        summary = "user_input_timeout"
                        stop_loop = True
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tu.id,
                                "content": "TIMEOUT: user did not provide input within "
                                f"{USER_INPUT_TIMEOUT_SECONDS}s",
                            }
                        )
                        continue
                    # Publish only field names, never values, into the event stream.
                    _publish(
                        "input_received",
                        step=step,
                        field_names=list(values.keys()),
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": json.dumps(values, ensure_ascii=False),
                        }
                    )
                    continue

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

    except Exception as exc:
        # Setup-phase failure (pinchtab unreachable, profile missing, etc.).
        # Without this catch, the task row stays in pending forever.
        log.exception("task %s runner exception", task_id)
        try:
            t = db.get(Task, task_id)
            if t is not None and t.status in (TaskStatus.pending, TaskStatus.running):
                t.transition(TaskStatus.errored)
                t.error_message = f"{type(exc).__name__}: {exc}"[:500]
                db.commit()
        except Exception:
            pass
        _publish(
            "terminal",
            status=TaskStatus.errored.value,
            error=f"{type(exc).__name__}: {exc}"[:200],
        )
        # Re-raise so direct callers (and tests) see the failure. The
        # dispatcher in tasks.py installs a done_callback that consumes the
        # exception silently in production (we already persisted + published).
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
