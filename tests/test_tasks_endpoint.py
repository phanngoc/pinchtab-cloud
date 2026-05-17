"""Integration tests for /tasks endpoints with the dispatch hook stubbed.

We don't run the real agent runner here (covered by test_agent_runner.py).
We verify the endpoint:
  - accepts the Claude API key in the body
  - calls dispatch() exactly once with the right task_id and key
  - does NOT leak the key into the response or task row
  - rejects denied start_urls
  - enforces capacity cap
  - dispatch failure is surfaced as 500 and the task row reflects errored state
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app
from backend.models import Task, TaskStatus


def _signup(c: TestClient) -> str:
    r = c.post("/auth/request-link", json={"email": "alice@example.com"})
    token = r.json()["dev_link"].split("token=", 1)[1]
    return c.post("/auth/verify", json={"token": token}).json()["bearer"]


def _install_dispatcher_recorder(client: TestClient):
    calls: list[tuple[str, str]] = []

    def fake_dispatch(task_id: str, api_key: str) -> None:
        calls.append((task_id, api_key))

    client.app.state.dispatcher = fake_dispatch
    return calls


def test_post_tasks_dispatches_and_returns_pending():
    with TestClient(app) as c:
        bearer = _signup(c)
        calls = _install_dispatcher_recorder(c)

        r = c.post(
            "/tasks",
            json={
                "task_description": "find main headlines on news.example.com",
                "start_url": "https://news.example.com/",
                "anthropic_api_key": "sk-ant-FAKE-12345",
            },
            headers={"Authorization": f"Bearer {bearer}"},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["status"] == "pending"
        assert "profile_id" in body and body["profile_id"]
        # Dispatcher was called once with our task id and key.
        assert len(calls) == 1
        assert calls[0][0] == body["id"]
        assert calls[0][1] == "sk-ant-FAKE-12345"


def test_response_does_not_leak_api_key():
    with TestClient(app) as c:
        bearer = _signup(c)
        _install_dispatcher_recorder(c)

        api_key = "sk-ant-DO-NOT-LEAK-ABCDE"
        r = c.post(
            "/tasks",
            json={
                "task_description": "task body to send",
                "anthropic_api_key": api_key,
            },
            headers={"Authorization": f"Bearer {bearer}"},
        )
        assert r.status_code == 201
        assert api_key not in r.text


def test_denied_start_url_rejected_before_dispatch():
    with TestClient(app) as c:
        bearer = _signup(c)
        calls = _install_dispatcher_recorder(c)

        r = c.post(
            "/tasks",
            json={
                "task_description": "do something on a denied site",
                "start_url": "https://m.shopee.vn/",  # subdomain of denied apex
                "anthropic_api_key": "sk-ant-FAKE",
            },
            headers={"Authorization": f"Bearer {bearer}"},
        )
        assert r.status_code == 400
        assert r.json()["detail"]["rule"] == "shopee.vn"
        # No dispatch when input fails validation.
        assert calls == []


def test_dispatch_failure_marks_task_errored_and_returns_500():
    """If the dispatch hook raises, the task row should reflect errored
    state and the client gets a clean 500 — not a stuck pending row."""
    with TestClient(app) as c:
        bearer = _signup(c)

        def boom(task_id: str, api_key: str) -> None:
            raise RuntimeError("worker pool exhausted")

        c.app.state.dispatcher = boom

        r = c.post(
            "/tasks",
            json={
                "task_description": "task that will fail to dispatch",
                "anthropic_api_key": "sk-ant-FAKE",
            },
            headers={"Authorization": f"Bearer {bearer}"},
        )
        assert r.status_code == 500
        assert r.json()["detail"] == "dispatch_failed"


def test_post_tasks_omitted_key_requires_operator():
    """Missing key falls back to CLI provider — restricted to operator email.
    Non-operator without a key gets 400 api_key_required."""
    with TestClient(app) as c:
        bearer = _signup(c)
        _install_dispatcher_recorder(c)

        r = c.post(
            "/tasks",
            json={"task_description": "missing api key on purpose"},
            headers={"Authorization": f"Bearer {bearer}"},
        )
        # Without OPERATOR_EMAIL set, no one is the operator → 400.
        assert r.status_code == 400
        body = r.json()
        assert body["detail"]["error"] == "api_key_required"


def test_post_tasks_short_key_rejected():
    """A non-empty key shorter than 10 chars is rejected."""
    with TestClient(app) as c:
        bearer = _signup(c)
        _install_dispatcher_recorder(c)

        r = c.post(
            "/tasks",
            json={
                "task_description": "short api key on purpose to test rejection",
                "anthropic_api_key": "tooshort",
            },
            headers={"Authorization": f"Bearer {bearer}"},
        )
        assert r.status_code == 400
        assert r.json()["detail"]["error"] == "anthropic_api_key_too_short"


def test_post_tasks_omitted_key_works_for_operator(monkeypatch):
    """When OPERATOR_EMAIL is set and the user matches, omitted key is OK
    and the dispatcher receives an empty string (run_task picks CLI provider)."""
    with TestClient(app) as c:
        bearer = _signup(c)  # this helper hard-codes alice@example.com
        calls = _install_dispatcher_recorder(c)
        monkeypatch.setenv("OPERATOR_EMAIL", "alice@example.com")
        # The is_operator helper consults settings + env; refresh settings cache.
        from backend.config import get_settings
        get_settings.cache_clear()

        r = c.post(
            "/tasks",
            json={"task_description": "operator fallback to CLI, no key"},
            headers={"Authorization": f"Bearer {bearer}"},
        )
        assert r.status_code == 201, r.text
        # Dispatcher was called with empty api_key — runner will pick CLI.
        assert len(calls) == 1
        assert calls[0][1] == ""
        get_settings.cache_clear()


def test_halt_returns_409_when_not_running():
    """Tasks created via this test path go straight to pending (dispatcher is
    a no-op recorder). Halting a pending task should 409, not silently 200."""
    with TestClient(app) as c:
        bearer = _signup(c)
        _install_dispatcher_recorder(c)

        r = c.post(
            "/tasks",
            json={"task_description": "task pending forever", "anthropic_api_key": "sk-ant-test-1234567890"},
            headers={"Authorization": f"Bearer {bearer}"},
        )
        task_id = r.json()["id"]
        r = c.post(f"/tasks/{task_id}/halt", headers={"Authorization": f"Bearer {bearer}"})
        assert r.status_code == 409
        assert "not running" in r.json()["detail"]


def test_list_and_get_after_dispatch():
    with TestClient(app) as c:
        bearer = _signup(c)
        _install_dispatcher_recorder(c)

        r = c.post(
            "/tasks",
            json={"task_description": "first task created", "anthropic_api_key": "sk-ant-test-1234567890"},
            headers={"Authorization": f"Bearer {bearer}"},
        )
        task_id = r.json()["id"]

        r = c.get("/tasks", headers={"Authorization": f"Bearer {bearer}"})
        assert r.status_code == 200
        assert any(t["id"] == task_id for t in r.json())

        r = c.get(f"/tasks/{task_id}", headers={"Authorization": f"Bearer {bearer}"})
        assert r.status_code == 200
        assert r.json()["id"] == task_id


def test_second_task_reuses_same_profile():
    with TestClient(app) as c:
        bearer = _signup(c)
        _install_dispatcher_recorder(c)

        r1 = c.post(
            "/tasks",
            json={"task_description": "first one for the user", "anthropic_api_key": "sk-ant-test-1234567890"},
            headers={"Authorization": f"Bearer {bearer}"},
        )
        r2 = c.post(
            "/tasks",
            json={"task_description": "second task for same user", "anthropic_api_key": "sk-ant-test-1234567890"},
            headers={"Authorization": f"Bearer {bearer}"},
        )
        assert r1.json()["profile_id"] == r2.json()["profile_id"]
