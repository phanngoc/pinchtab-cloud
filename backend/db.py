from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.config import get_settings


class Base(DeclarativeBase):
    pass


_settings = get_settings()
_url = _settings.database_url

_engine_kwargs: dict = {"future": True}
if _url.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
    # In-memory SQLite is per-connection; StaticPool ensures all sessions
    # see the same database (required for tests + dev with :memory: URL).
    if ":memory:" in _url:
        _engine_kwargs["poolclass"] = StaticPool

engine = create_engine(_url, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
