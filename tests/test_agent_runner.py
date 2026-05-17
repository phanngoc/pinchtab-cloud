"""Tests for the agent runner.

Strategy: dependency injection. Provide a FakePinchtabClient and a fake
Anthropic factory that return scripted responses. No real network, no real
LLM, no real pinchtab daemon.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.agent_runner import (
    AgentResult,
    _execute_tool,
    _extract_refs,
    _humanize_pinchtab_error,
    run_task,
)
from backend.pinchtab_client import PinchtabError
from backend.db import Base
from backend.models import Profile, Task, TaskEvent, TaskStatus, User


# ---------------------------------------------------------------------------
# Fixtures: isolated in-memory DB per test
# ---------------------------------------------------------------------------

@pytest.fixture
def db_factory(tmp_path):
    """Each test gets its own in-memory SQLite with shared connection
    (StaticPool keeps the same `:memory:` db visible across sessions)."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return Session


@pytest.fixture
def seed(db_factory):
    """Create a user, profile, and pending task. Returns (user_id, profile_id, task_id)."""
    with db_factory() as db:
        user = User(email="alice@example.com")
        db.add(user)
        db.flush()
        profile = Profile(
            user_id=user.id, pinchtab_profile_name=f"u_alice_test"
        )
        db.add(profile)
        db.flush()
        task = Task(
            user_id=user.id,
            profile_id=profile.id,
            task_description="find the main news headlines on news.example.com",
            start_url="https://news.example.com/",
            status=TaskStatus.pending,
        )
        db.add(task)
        db.commit()
        return (user.id, profile.id, task.id)


# ---------------------------------------------------------------------------
# Fake pinchtab client
# ---------------------------------------------------------------------------

@dataclass
class FakePinchtabClient:
    """Records every call; returns scripted responses.

    Defaults are valid for the happy path. Tests override per-test for failure
    paths. snap_text_sequence cycles through snapshots step-by-step.
    """

    snap_text_sequence: list[str] = field(default_factory=list)
    raise_on_open_tab: Exception | None = None
    raise_on_snapshot: Exception | None = None
    instance_response: dict = field(
        default_factory=lambda: {"id": "inst_fake1", "status": "running"}
    )
    open_tab_response: dict = field(
        default_factory=lambda: {"id": "tab_fake1"}
    )
    calls: list[tuple[str, dict]] = field(default_factory=list)
    snapshot_idx: int = 0
    closed: bool = False

    # ---- record helper ----
    def _record(self, name: str, **kwargs):
        self.calls.append((name, kwargs))

    # ---- instance ----
    async def get_instance(self, instance_id: str):
        self._record("get_instance", instance_id=instance_id)
        return None  # force start_instance path; happy path uses fresh start

    async def start_instance(self, *, profile_id: str, mode: str = "headless"):
        self._record("start_instance", profile_id=profile_id, mode=mode)
        return self.instance_response

    async def stop_instance(self, instance_id: str):
        self._record("stop_instance", instance_id=instance_id)
        return {"ok": True}

    # ---- tabs ----
    async def open_tab(self, instance_id: str, url: str | None = None):
        self._record("open_tab", instance_id=instance_id, url=url)
        if self.raise_on_open_tab:
            raise self.raise_on_open_tab
        return self.open_tab_response

    async def close_tab(self, tab_id: str):
        self._record("close_tab", tab_id=tab_id)
        return {"ok": True}

    async def navigate(self, tab_id: str, url: str):
        self._record("navigate", tab_id=tab_id, url=url)
        return {"ok": True}

    # ---- inspection ----
    async def snapshot(self, tab_id: str, *, interactive=True, compact=True, max_tokens=None):
        self._record("snapshot", tab_id=tab_id, max_tokens=max_tokens)
        if self.raise_on_snapshot:
            raise self.raise_on_snapshot
        if not self.snap_text_sequence:
            return "e1:button \"placeholder\""
        idx = min(self.snapshot_idx, len(self.snap_text_sequence) - 1)
        self.snapshot_idx += 1
        return self.snap_text_sequence[idx]

    async def screenshot(self, tab_id: str, *, quality=70):
        self._record("screenshot", tab_id=tab_id, quality=quality)
        # 1-byte fake PNG — runtime only checks magic bytes to pick media_type
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 8

    async def text(self, tab_id: str):
        return "page text"

    # ---- actions ----
    async def click(self, tab_id: str, ref: str):
        self._record("click", tab_id=tab_id, ref=ref)
        return {"clicked": True, "ref": ref}

    async def type_text(self, tab_id: str, ref: str, text: str):
        self._record("type_text", tab_id=tab_id, ref=ref, text=text)
        return {"typed": True}

    async def press_key(self, tab_id: str, key: str):
        self._record("press_key", tab_id=tab_id, key=key)
        return {"pressed": key}

    async def scroll(self, tab_id: str, amount):
        self._record("scroll", tab_id=tab_id, amount=amount)
        return {"scrolled": amount}

    async def select_option(self, tab_id: str, ref, value):
        self._record("select_option", tab_id=tab_id, ref=ref, value=value)
        return {"selected": value}

    # ---- denylist ----
    async def apply_denylist(self, tab_id: str, policy):
        self._record("apply_denylist", tab_id=tab_id, deny_count=len(policy.deny))
        return len(policy.deny) * 2  # 2 rules per domain

    async def add_route_rule(self, tab_id, rule):
        self._record("add_route_rule", tab_id=tab_id, rule=rule)
        return {"ok": True}

    async def aclose(self):
        self.closed = True


# ---------------------------------------------------------------------------
# Fake Anthropic client
# ---------------------------------------------------------------------------

class _Block:
    """Mimic an Anthropic content block (tool_use or text)."""

    def __init__(self, type: str, **fields):
        self.type = type
        for k, v in fields.items():
            setattr(self, k, v)

    def model_dump(self):
        d = {"type": self.type}
        for attr in ("id", "name", "input", "text"):
            if hasattr(self, attr):
                d[attr] = getattr(self, attr)
        return d


class _Response:
    def __init__(self, blocks: list[_Block]):
        self.content = blocks

    def model_dump(self):
        return {"content": [b.model_dump() for b in self.content]}


class _Messages:
    def __init__(self, scripted: list[list[_Block]]):
        self._scripted = scripted
        self._idx = 0
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._scripted:
            return _Response([])
        blocks = self._scripted[min(self._idx, len(self._scripted) - 1)]
        self._idx += 1
        return _Response(blocks)


class FakeAnthropic:
    def __init__(self, scripted: list[list[_Block]]):
        self.messages = _Messages(scripted)


def anthropic_factory_for(scripted: list[list[_Block]]):
    captured: list[FakeAnthropic] = []

    def factory(api_key: str) -> FakeAnthropic:
        a = FakeAnthropic(scripted)
        captured.append(a)
        return a

    factory.captured = captured  # type: ignore[attr-defined]
    return factory


def _tool_use_block(name: str, **inputs) -> _Block:
    import secrets

    return _Block(
        type="tool_use",
        id="toolu_" + secrets.token_hex(6),
        name=name,
        input=inputs,
    )


def _text_block(text: str) -> _Block:
    return _Block(type="text", text=text)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_task_complete(db_factory, seed, tmp_path):
    """Step 1: scroll. Step 2: task_complete with summary."""
    _, _, task_id = seed
    snaps = [
        "e1:link \"Headline A\"\ne2:link \"Headline B\"",
        "e1:link \"Headline A\"\ne2:link \"Headline B\"\ne3:link \"Headline C\"\ne4:link \"Headline D\"",
    ]
    scripted = [
        [_text_block("Need to scroll for more headlines."), _tool_use_block("scroll", amount="1000")],
        [_text_block("Got enough headlines."), _tool_use_block("task_complete", summary="Found 4 headlines")],
    ]
    fake_pt = FakePinchtabClient(snap_text_sequence=snaps)

    res = await run_task(
        task_id,
        anthropic_api_key="sk-fake",
        pinchtab_client=fake_pt,
        anthropic_factory=anthropic_factory_for(scripted),
        db_session_factory=db_factory,
        log_dir=tmp_path,
        step_delay_seconds=0.0,
    )

    assert res.terminal == TaskStatus.done
    assert res.summary == "Found 4 headlines"
    assert res.steps == 2

    # Verify DB state.
    with db_factory() as db:
        t = db.get(Task, task_id)
        assert t.status == TaskStatus.done
        assert t.final_summary == "Found 4 headlines"
        assert t.pinchtab_tab_id == "tab_fake1"
        assert t.minutes_consumed >= 0
    # Apply denylist before any agent step ran.
    call_names = [c[0] for c in fake_pt.calls]
    assert call_names.index("apply_denylist") < call_names.index("snapshot")
    assert "close_tab" in call_names


@pytest.mark.asyncio
async def test_halt_for_human(db_factory, seed, tmp_path):
    _, _, task_id = seed
    scripted = [
        [_tool_use_block("halt_for_human", reason="payment confirmation required")],
    ]
    fake_pt = FakePinchtabClient(snap_text_sequence=["e1:button \"Pay now\""])

    res = await run_task(
        task_id,
        anthropic_api_key="sk-fake",
        pinchtab_client=fake_pt,
        anthropic_factory=anthropic_factory_for(scripted),
        db_session_factory=db_factory,
        log_dir=tmp_path,
        step_delay_seconds=0.0,
    )

    assert res.terminal == TaskStatus.halted
    assert "payment confirmation" in res.summary
    with db_factory() as db:
        t = db.get(Task, task_id)
        assert t.status == TaskStatus.halted


@pytest.mark.asyncio
async def test_safety_pattern_halts_before_claude(db_factory, seed, tmp_path):
    """Captcha page text triggers unconditional halt — Anthropic is never called."""
    _, _, task_id = seed
    # The snap text contains 'captcha' (lowercased substring match).
    captcha_snap = "e1:textbox \"Please solve the CAPTCHA below\"\ne2:button \"Verify\""
    fake_pt = FakePinchtabClient(snap_text_sequence=[captcha_snap])
    factory = anthropic_factory_for([])  # empty — would error if reached

    res = await run_task(
        task_id,
        anthropic_api_key="sk-fake",
        pinchtab_client=fake_pt,
        anthropic_factory=factory,
        db_session_factory=db_factory,
        log_dir=tmp_path,
        step_delay_seconds=0.0,
    )

    assert res.terminal == TaskStatus.halted
    assert "captcha" in res.summary.lower()
    # Crucial: Anthropic was NEVER called (prompt-injection mitigation).
    assert factory.captured[0].messages.calls == []
    with db_factory() as db:
        t = db.get(Task, task_id)
        assert t.status == TaskStatus.halted


@pytest.mark.asyncio
async def test_pinchtab_snapshot_failure_errors_task(db_factory, seed, tmp_path):
    _, _, task_id = seed
    fake_pt = FakePinchtabClient(raise_on_snapshot=RuntimeError("CDP disconnected"))

    res = await run_task(
        task_id,
        anthropic_api_key="sk-fake",
        pinchtab_client=fake_pt,
        anthropic_factory=anthropic_factory_for([]),
        db_session_factory=db_factory,
        log_dir=tmp_path,
        step_delay_seconds=0.0,
    )

    assert res.terminal == TaskStatus.errored
    assert "CDP disconnected" in res.error_message
    with db_factory() as db:
        t = db.get(Task, task_id)
        assert t.status == TaskStatus.errored


@pytest.mark.asyncio
async def test_open_tab_failure_errors_task_at_setup(db_factory, seed, tmp_path):
    """Failure before the run loop starts — task errored without ever transitioning to running."""
    _, _, task_id = seed
    fake_pt = FakePinchtabClient(raise_on_open_tab=RuntimeError("instance unhealthy"))

    res = await run_task(
        task_id,
        anthropic_api_key="sk-fake",
        pinchtab_client=fake_pt,
        anthropic_factory=anthropic_factory_for([]),
        db_session_factory=db_factory,
        log_dir=tmp_path,
        step_delay_seconds=0.0,
    )

    assert res.terminal == TaskStatus.errored
    assert "instance unhealthy" in res.error_message
    with db_factory() as db:
        t = db.get(Task, task_id)
        assert t.status == TaskStatus.errored
        # Tab was never assigned.
        assert t.pinchtab_tab_id is None


@pytest.mark.asyncio
async def test_max_steps_reached_errors(db_factory, seed, tmp_path):
    _, _, task_id = seed
    # Claude keeps scrolling forever with VARYING args — never task_complete.
    # Args vary each step so the (tool, args) loop detector does NOT fire;
    # this isolates the max_steps safety net.
    fake_pt = FakePinchtabClient(snap_text_sequence=["e1:link \"X\""])
    varied_responses = [
        [_tool_use_block("scroll", amount=str(100 * (i + 1)))] for i in range(10)
    ]

    res = await run_task(
        task_id,
        anthropic_api_key="sk-fake",
        pinchtab_client=fake_pt,
        anthropic_factory=anthropic_factory_for(varied_responses),
        db_session_factory=db_factory,
        max_steps=3,
        log_dir=tmp_path,
        step_delay_seconds=0.0,
    )

    assert res.terminal == TaskStatus.errored
    assert "max_steps" in res.error_message
    assert res.steps == 3


def test_extract_refs_handles_real_snap_format():
    snap = (
        "# Page title | https://example.com | 12 nodes\n"
        'e2:link "Home"\n'
        'e5:button "Login"\n'
        "e42:textbox \"Search\"\n"
        # noise lines that should NOT match
        "Just some prose with e99 in the middle (no colon).\n"
        "  e7:not-anchored (leading space, must not match)\n"
    )
    assert _extract_refs(snap) == {"e2", "e5", "e42"}
    assert _extract_refs("") == set()


def test_humanize_pinchtab_error_detached_node():
    """Real Viblo failure: 'Node is detached from document' should map to
    an actionable hint about DOM mutation, not the raw JSON."""
    exc = PinchtabError(
        500,
        '{"code":"action_failed","error":"action click: scroll into view: '
        'Node is detached from document (-32000)","retryable":true}',
        "POST /tabs/X/action",
    )
    msg = _humanize_pinchtab_error(exc)
    assert "DOM" in msg or "removed" in msg
    assert "next snap" in msg.lower()
    # Original JSON noise should NOT be in the actionable part.
    assert '"code":"action_failed"' not in msg


def test_humanize_pinchtab_error_navigation_changed():
    exc = PinchtabError(
        409,
        '{"code":"navigation_changed","error":"unexpected page navigation: '
        'https://a.com -> https://b.com"}',
        "POST /tabs/X/action",
    )
    msg = _humanize_pinchtab_error(exc)
    assert "navigated" in msg.lower()
    assert "stale" in msg.lower()


def test_humanize_pinchtab_error_not_focusable():
    exc = PinchtabError(
        500,
        '{"code":"action_failed","error":"action fill: Element is not focusable (-32000)"}',
        "POST /tabs/X/action",
    )
    msg = _humanize_pinchtab_error(exc)
    assert "focusable" in msg.lower()
    assert "find_element" in msg.lower()


def test_humanize_pinchtab_error_unknown_falls_back_cleanly():
    """An unrecognized error still gets sanitized — no raw JSON braces."""
    exc = PinchtabError(
        500,
        '{"code":"weird_thing","error":"something broke deeply"}',
        "POST /tabs/X/action",
    )
    msg = _humanize_pinchtab_error(exc)
    assert "action failed" in msg
    assert "something broke deeply" in msg
    assert "{" not in msg  # no JSON noise


@pytest.mark.asyncio
async def test_pinchtab_error_translated_in_tool_result():
    """End-to-end: a click that raises PinchtabError returns a humanized
    error string to the agent, not the raw HTTP body."""
    class FlakyClient:
        async def click(self, tab_id, ref):
            raise PinchtabError(
                500,
                '{"code":"action_failed","error":"action click: Node is detached from document"}',
                "POST /tabs/X/action",
            )

    res, _, _, _ = await _execute_tool(
        FlakyClient(),  # type: ignore[arg-type]
        "tab-x",
        "click",
        {"ref": "e5"},
        None,  # type: ignore[arg-type]
        current_refs={"e5"},
    )
    assert "action failed:" in res
    assert "DOM" in res or "removed" in res
    assert "HTTP 500" not in res
    assert '"code"' not in res


@pytest.mark.asyncio
async def test_stale_ref_rejected_before_pinchtab_call():
    """Clicking a ref that's not in the current snap returns a helpful error
    WITHOUT calling pinchtab — saves wasted roundtrips on detached/old refs."""
    calls: list[tuple[str, dict[str, Any]]] = []

    class TrackingClient:
        async def click(self, tab_id, ref):
            calls.append(("click", {"ref": ref}))
            return {"clicked": True}

    res, stop, term, _ = await _execute_tool(
        TrackingClient(),  # type: ignore[arg-type]
        "tab-x",
        "click",
        {"ref": "e999"},
        None,  # type: ignore[arg-type]
        current_refs={"e2", "e5"},
    )
    assert calls == []  # pinchtab was NEVER hit
    assert "ref 'e999' is not in the current snap" in res
    assert "e2" in res and "e5" in res  # suggestion includes valid refs
    assert stop is False and term is None


@pytest.mark.asyncio
async def test_valid_ref_passes_through_to_pinchtab():
    """Sanity: a ref present in current_refs is forwarded to the client."""
    class TrackingClient:
        async def click(self, tab_id, ref):
            return {"clicked": True, "ref": ref}

    res, _, _, _ = await _execute_tool(
        TrackingClient(),  # type: ignore[arg-type]
        "tab-x",
        "click",
        {"ref": "e5"},
        None,  # type: ignore[arg-type]
        current_refs={"e2", "e5"},
    )
    assert '"clicked": true' in res


@pytest.mark.asyncio
async def test_ref_guard_skipped_when_current_refs_is_none():
    """Tests / callers that don't pass current_refs (legacy behavior) keep working."""
    class TrackingClient:
        async def click(self, tab_id, ref):
            return {"clicked": True}

    res, _, _, _ = await _execute_tool(
        TrackingClient(),  # type: ignore[arg-type]
        "tab-x",
        "click",
        {"ref": "e999"},
        None,  # type: ignore[arg-type]
        current_refs=None,
    )
    assert '"clicked": true' in res


@pytest.mark.asyncio
async def test_llm_call_retries_once_on_transient_failure(db_factory, seed, tmp_path):
    """A transient exception on the first messages.create() attempt should
    trigger one retry; second attempt succeeding lets the task continue.
    Mirrors the CLI 120s-timeout failure mode observed in production."""
    _, _, task_id = seed
    success_blocks = [_tool_use_block("task_complete", summary="recovered after retry")]
    fake_pt = FakePinchtabClient(snap_text_sequence=["e1:link \"X\""])

    # Anthropic-like client: first call raises, subsequent calls return success.
    class FlakyMessages:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated CLI timeout after 180.0s")
            return _Response(success_blocks)

    class FlakyAnthropic:
        def __init__(self):
            self.messages = FlakyMessages()
            # Presence of session_id makes the runner treat this as CLI mode
            # and reset on retry — exercises that code path.
            self.session_id = "stale-session-uuid"
            self._messages_sent_count = 999

    flaky = FlakyAnthropic()

    def factory(_key):
        return flaky

    res = await run_task(
        task_id,
        anthropic_api_key="sk-fake",
        pinchtab_client=fake_pt,
        anthropic_factory=factory,
        db_session_factory=db_factory,
        max_steps=3,
        log_dir=tmp_path,
        step_delay_seconds=0.0,
    )

    # Task completed because attempt 2 succeeded.
    assert res.terminal == TaskStatus.done
    assert flaky.messages.calls == 2  # exactly one retry
    # CLI session was reset before the retry — proves the reset path fired.
    assert flaky.session_id is None
    assert flaky._messages_sent_count == 0


@pytest.mark.asyncio
async def test_llm_call_fails_after_second_retry(db_factory, seed, tmp_path):
    """If both attempts fail, task errors with last exception's message."""
    _, _, task_id = seed
    fake_pt = FakePinchtabClient(snap_text_sequence=["e1:link \"X\""])

    class AlwaysFailMessages:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            raise RuntimeError("permanent failure")

    class AlwaysFailAnthropic:
        def __init__(self):
            self.messages = AlwaysFailMessages()

    af = AlwaysFailAnthropic()
    res = await run_task(
        task_id,
        anthropic_api_key="sk-fake",
        pinchtab_client=fake_pt,
        anthropic_factory=lambda _k: af,
        db_session_factory=db_factory,
        max_steps=3,
        log_dir=tmp_path,
        step_delay_seconds=0.0,
    )
    assert res.terminal == TaskStatus.errored
    assert af.messages.calls == 2  # 1 attempt + 1 retry, then give up
    assert "permanent failure" in (res.error_message or "")


@pytest.mark.asyncio
async def test_events_persisted_to_db(db_factory, seed, tmp_path):
    """Runner writes a TaskEvent row for every event it publishes —
    enabling after-the-fact history view in the dashboard."""
    _, _, task_id = seed
    scripted = [[_tool_use_block("task_complete", summary="all done")]]
    fake_pt = FakePinchtabClient(snap_text_sequence=["e1:link \"X\""])

    await run_task(
        task_id,
        anthropic_api_key="sk-fake",
        pinchtab_client=fake_pt,
        anthropic_factory=anthropic_factory_for(scripted),
        db_session_factory=db_factory,
        max_steps=3,
        log_dir=tmp_path,
        step_delay_seconds=0.0,
    )

    with db_factory() as db:
        events = db.query(TaskEvent).filter(TaskEvent.task_id == task_id).order_by(TaskEvent.id).all()
        kinds = [e.kind for e in events]
        # We don't pin the exact event sequence (it would couple the test to
        # publish-call ordering), but we require: started, ≥1 step, ≥1
        # tool_call, terminal.
        assert "started" in kinds
        assert "step" in kinds
        assert "tool_call" in kinds
        assert "terminal" in kinds
        # Every event has parseable JSON payload.
        import json as _json
        for e in events:
            _json.loads(e.payload_json)


@pytest.mark.asyncio
async def test_tool_loop_detector_halts(db_factory, seed, tmp_path):
    """Agent calling same (tool, args) 3+ times in last 5 turns triggers halt."""
    _, _, task_id = seed
    # Identical scroll(500) every turn — loop detector should fire by step 3.
    looping_response = [_tool_use_block("scroll", amount="500")]
    fake_pt = FakePinchtabClient(snap_text_sequence=["e1:link \"X\""])

    res = await run_task(
        task_id,
        anthropic_api_key="sk-fake",
        pinchtab_client=fake_pt,
        anthropic_factory=anthropic_factory_for([looping_response] * 10),
        db_session_factory=db_factory,
        max_steps=10,
        log_dir=tmp_path,
        step_delay_seconds=0.0,
    )

    assert res.terminal == TaskStatus.halted
    assert "loop" in (res.error_message or "").lower() or res.steps <= 5


@pytest.mark.asyncio
async def test_denylist_installed_before_agent_loop(db_factory, seed, tmp_path):
    """The first action against the tab must be apply_denylist, not snapshot."""
    _, _, task_id = seed
    scripted = [[_tool_use_block("task_complete", summary="trivial")]]
    fake_pt = FakePinchtabClient(snap_text_sequence=["e1:link"])

    await run_task(
        task_id,
        anthropic_api_key="sk-fake",
        pinchtab_client=fake_pt,
        anthropic_factory=anthropic_factory_for(scripted),
        db_session_factory=db_factory,
        log_dir=tmp_path,
        step_delay_seconds=0.0,
    )

    names = [c[0] for c in fake_pt.calls]
    assert "apply_denylist" in names
    apply_idx = names.index("apply_denylist")
    first_snap_idx = names.index("snapshot")
    assert apply_idx < first_snap_idx


@pytest.mark.asyncio
async def test_api_key_not_in_log_files(db_factory, seed, tmp_path):
    """No log file should ever contain the Claude API key."""
    _, _, task_id = seed
    api_key = "sk-ant-fake-DO-NOT-LEAK-12345"
    scripted = [[_tool_use_block("task_complete", summary="ok")]]
    fake_pt = FakePinchtabClient(snap_text_sequence=["e1:link"])

    await run_task(
        task_id,
        anthropic_api_key=api_key,
        pinchtab_client=fake_pt,
        anthropic_factory=anthropic_factory_for(scripted),
        db_session_factory=db_factory,
        log_dir=tmp_path,
        step_delay_seconds=0.0,
    )

    for f in tmp_path.rglob("*"):
        if f.is_file():
            content = f.read_bytes()
            assert api_key.encode() not in content, f"api key leaked into {f}"


@pytest.mark.asyncio
async def test_runner_rejects_non_pending_task(db_factory, seed, tmp_path):
    """A task already in `running` state must not be re-driven (idempotency)."""
    _, _, task_id = seed
    # Force task into running state out-of-band.
    with db_factory() as db:
        t = db.get(Task, task_id)
        t.status = TaskStatus.running
        db.commit()

    fake_pt = FakePinchtabClient()
    with pytest.raises(RuntimeError, match="not pending"):
        await run_task(
            task_id,
            anthropic_api_key="sk-fake",
            pinchtab_client=fake_pt,
            anthropic_factory=anthropic_factory_for([]),
            db_session_factory=db_factory,
            log_dir=tmp_path,
            step_delay_seconds=0.0,
        )
