"""Integration tests for /profiles endpoints. FakePinchtabClient is injected
via app.state.pinchtab — no real pinchtab daemon needed.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


def _signup(c: TestClient, email: str = "alice@example.com") -> str:
    r = c.post("/auth/request-link", json={"email": email})
    token = r.json()["dev_link"].split("token=", 1)[1]
    return c.post("/auth/verify", json={"token": token}).json()["bearer"]


class StubPinchtab:
    """Records every call. Default behavior = success."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.stop_should_404 = False

    async def stop_instance(self, instance_id: str):
        self.calls.append(("stop_instance", {"instance_id": instance_id}))
        if self.stop_should_404:
            from backend.pinchtab_client import PinchtabError
            raise PinchtabError(404, "not found", f"POST /instances/{instance_id}/stop")
        return {"ok": True}

    async def aclose(self):
        pass


def _install_pinchtab(c: TestClient) -> StubPinchtab:
    stub = StubPinchtab()
    c.app.state.pinchtab = stub
    return stub


def _create_task(c: TestClient, bearer: str, description: str = "do something useful here") -> dict:
    """Create a task via the /tasks endpoint (lazy profile creation)."""
    # Stub dispatcher so the task stays in pending state without firing the runner.
    c.app.state.dispatcher = lambda task_id, api_key: None
    r = c.post(
        "/tasks",
        json={
            "task_description": description,
            "anthropic_api_key": "sk-ant-test-1234567890",
        },
        headers={"Authorization": f"Bearer {bearer}"},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _mark_task_done(profile_id: str):
    """Force any tasks on this profile to `done` so reset/delete aren't blocked."""
    from backend.db import SessionLocal
    from backend.models import Task, TaskStatus
    from sqlalchemy import select
    with SessionLocal() as db:
        rows = db.scalars(select(Task).where(Task.profile_id == profile_id)).all()
        for r in rows:
            r.status = TaskStatus.done
        db.commit()


# ---- tests ----


def test_list_profiles_empty_for_fresh_user():
    with TestClient(app) as c:
        bearer = _signup(c)
        r = c.get("/profiles", headers={"Authorization": f"Bearer {bearer}"})
        assert r.status_code == 200
        assert r.json() == []


def test_profile_appears_after_first_task():
    with TestClient(app) as c:
        bearer = _signup(c)
        _install_pinchtab(c)
        task = _create_task(c, bearer)
        r = c.get("/profiles", headers={"Authorization": f"Bearer {bearer}"})
        assert r.status_code == 200
        profiles = r.json()
        assert len(profiles) == 1
        assert profiles[0]["id"] == task["profile_id"]
        assert profiles[0]["instance_live"] is False
        assert profiles[0]["pinchtab_profile_name"].startswith("u_")


def test_reset_profile_rotates_name_and_stops_instance():
    with TestClient(app) as c:
        bearer = _signup(c)
        stub = _install_pinchtab(c)
        task = _create_task(c, bearer)
        profile_id = task["profile_id"]

        # Simulate a live instance attached to this profile.
        from backend.db import SessionLocal
        from backend.models import Profile
        with SessionLocal() as db:
            p = db.get(Profile, profile_id)
            p.pinchtab_instance_id = "inst_live_X"
            old_name = p.pinchtab_profile_name
            db.commit()

        _mark_task_done(profile_id)

        r = c.post(f"/profiles/{profile_id}/reset", headers={"Authorization": f"Bearer {bearer}"})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["instance_live"] is False
        assert out["pinchtab_profile_name"] != old_name
        assert out["pinchtab_profile_name"].startswith("u_")
        # stop_instance was called.
        assert any(name == "stop_instance" for name, _ in stub.calls)


def test_reset_rejects_when_live_tasks_exist():
    with TestClient(app) as c:
        bearer = _signup(c)
        _install_pinchtab(c)
        task = _create_task(c, bearer)
        # task stays in pending state since dispatcher is a no-op
        r = c.post(f"/profiles/{task['profile_id']}/reset", headers={"Authorization": f"Bearer {bearer}"})
        assert r.status_code == 409
        assert "live_tasks" in r.json()["detail"]


def test_reset_handles_stale_instance_id_gracefully():
    """If pinchtab returns 404 (instance already gone), reset proceeds."""
    with TestClient(app) as c:
        bearer = _signup(c)
        stub = _install_pinchtab(c)
        stub.stop_should_404 = True
        task = _create_task(c, bearer)
        profile_id = task["profile_id"]

        from backend.db import SessionLocal
        from backend.models import Profile
        with SessionLocal() as db:
            p = db.get(Profile, profile_id)
            p.pinchtab_instance_id = "inst_stale"
            db.commit()

        _mark_task_done(profile_id)

        r = c.post(f"/profiles/{profile_id}/reset", headers={"Authorization": f"Bearer {bearer}"})
        assert r.status_code == 200


def test_delete_profile_removes_row():
    with TestClient(app) as c:
        bearer = _signup(c)
        _install_pinchtab(c)
        task = _create_task(c, bearer)
        profile_id = task["profile_id"]
        _mark_task_done(profile_id)

        r = c.delete(f"/profiles/{profile_id}", headers={"Authorization": f"Bearer {bearer}"})
        assert r.status_code == 204

        r = c.get("/profiles", headers={"Authorization": f"Bearer {bearer}"})
        assert r.json() == []


def test_delete_rejects_when_live_tasks_exist():
    with TestClient(app) as c:
        bearer = _signup(c)
        _install_pinchtab(c)
        task = _create_task(c, bearer)
        r = c.delete(
            f"/profiles/{task['profile_id']}", headers={"Authorization": f"Bearer {bearer}"}
        )
        assert r.status_code == 409


def test_cannot_access_another_users_profile():
    """Alice's profile is invisible to Bob."""
    with TestClient(app) as c:
        alice = _signup(c, "alice@example.com")
        _install_pinchtab(c)
        alice_task = _create_task(c, alice)
        bob = _signup(c, "bob@example.com")

        r = c.get(
            f"/profiles/{alice_task['profile_id']}", headers={"Authorization": f"Bearer {bob}"}
        )
        # No GET /profiles/{id} singular endpoint, but reset/delete check ownership.
        r = c.post(
            f"/profiles/{alice_task['profile_id']}/reset",
            headers={"Authorization": f"Bearer {bob}"},
        )
        assert r.status_code == 404
        r = c.delete(
            f"/profiles/{alice_task['profile_id']}",
            headers={"Authorization": f"Bearer {bob}"},
        )
        assert r.status_code == 404


def test_pinchtab_unavailable_returns_503_on_reset():
    """Reset depends on stop_instance; if pinchtab not configured, 503."""
    with TestClient(app) as c:
        bearer = _signup(c)
        # Do NOT install pinchtab — lifespan creates a real one but it can't connect.
        # Direct way: delete it.
        c.app.state.pinchtab = None
        task = _create_task(c, bearer)
        # Wait — _create_task already set up dispatcher, but we removed pinchtab.
        # Tasks endpoint doesn't depend on pinchtab; this works.
        _mark_task_done(task["profile_id"])

        r = c.post(
            f"/profiles/{task['profile_id']}/reset",
            headers={"Authorization": f"Bearer {bearer}"},
        )
        assert r.status_code == 503
        assert r.json()["detail"] == "pinchtab_unavailable"
