"""Task state machine invariants.

Replaces the prior test_session_state_machine.py (BrowserSession concept
collapsed when the worker-per-container model was retired in favor of
pinchtab tabs).
"""
import pytest

from backend.models import InvalidTaskTransition, Task, TaskStatus


def _new_task(status: TaskStatus = TaskStatus.pending) -> Task:
    return Task(
        user_id="u_test",
        profile_id="p_test",
        task_description="some test task description",
        status=status,
    )


def test_happy_path():
    t = _new_task()
    t.transition(TaskStatus.running)
    assert t.started_at is not None
    t.transition(TaskStatus.done)
    assert t.ended_at is not None


def test_halted_then_resumed():
    """Captcha → halt → human solves → resume back to running."""
    t = _new_task(TaskStatus.running)
    t.transition(TaskStatus.halted)
    t.transition(TaskStatus.running)
    assert t.status == TaskStatus.running


def test_halted_then_done():
    """User abandons after halt — close out as done."""
    t = _new_task(TaskStatus.halted)
    t.transition(TaskStatus.done)
    assert t.status == TaskStatus.done


def test_pending_to_errored_allowed():
    """Worker fails to start — never reached running."""
    t = _new_task()
    t.transition(TaskStatus.errored)
    assert t.status == TaskStatus.errored
    assert t.ended_at is not None


def test_cannot_revive_done():
    t = _new_task(TaskStatus.done)
    with pytest.raises(InvalidTaskTransition):
        t.transition(TaskStatus.running)


def test_cannot_revive_errored():
    t = _new_task(TaskStatus.errored)
    with pytest.raises(InvalidTaskTransition):
        t.transition(TaskStatus.running)


def test_cannot_skip_states():
    t = _new_task()
    with pytest.raises(InvalidTaskTransition):
        t.transition(TaskStatus.halted)  # pending → halted not allowed
    with pytest.raises(InvalidTaskTransition):
        t.transition(TaskStatus.done)    # pending → done not allowed
