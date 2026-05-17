"""Test env bootstrap. Loaded by pytest before any test module imports.

Backend modules (db.py, config.py) read settings at import time, so env
vars must be set in os.environ before the first `from backend...` import
in any test module.
"""
import os

os.environ.setdefault("APP_SECRET_KEY", "test-secret-test-secret-test-secret-32b")
os.environ.setdefault("WORKER_SHARED_SECRET", "test-worker-test-worker-test-worker-32b")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("APP_ENV", "test")
# Tests assume the default Anthropic SDK provider (no operator gate).
# Force override even if developer's local .env sets LLM_PROVIDER=claude_cli.
os.environ["LLM_PROVIDER"] = "anthropic_sdk"
os.environ["OPERATOR_EMAIL"] = ""

import pytest


@pytest.fixture(autouse=True)
def _reset_db_between_tests():
    """Drop and recreate all tables before each test that touches the global
    engine. Tests using their own engine (db_factory fixture in
    test_agent_runner.py) are unaffected — Base.metadata.drop_all on the
    global engine doesn't touch other engines.
    """
    # Lazy import to ensure env vars are read first.
    from backend.db import Base, engine

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    # No-op on teardown; next test will drop and recreate.


@pytest.fixture(autouse=True)
def _reset_app_state_between_tests():
    """Clear injected dispatcher / pinchtab / registry so cross-test
    state doesn't leak."""
    yield
    try:
        from backend.main import app
        from backend.task_bus import registry

        for attr in ("dispatcher", "pinchtab"):
            if hasattr(app.state, attr):
                delattr(app.state, attr)
        # Cancel any leftover registered tasks.
        for tid in list(registry._tasks.keys()):
            t = registry._tasks.pop(tid)
            if not t.done():
                t.cancel()
    except Exception:
        pass
