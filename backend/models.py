"""SQLAlchemy models.

Schema reflects the multi-tenant architecture: one pinchtab server, one
profile-per-user, instances launched on demand, tasks are short-lived tab
sessions on a user's instance.

State machines:

  Task (a single user-issued automation request):
      pending → running → done
                      ↘ halted ↘ errored
                              ↗ running (resumable from halted, e.g. after captcha)

  Profile (a user's persistent pinchtab profile):
      created → active (instance live) → idle (instance evicted) → active (cold start)
      created → deleted (user-requested wipe)
      No row deletion; soft-delete via status when needed.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Tier(str, enum.Enum):
    free = "free"
    starter = "starter"
    pro = "pro"


class TaskStatus(str, enum.Enum):
    pending = "pending"   # row created, agent loop not yet started
    running = "running"   # agent loop in flight on the bound tab
    halted = "halted"     # halt_for_human called (captcha, OTP, safety pattern)
    done = "done"         # task_complete called by agent
    errored = "errored"   # unrecoverable failure


TASK_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.pending: {TaskStatus.running, TaskStatus.errored},
    TaskStatus.running: {TaskStatus.halted, TaskStatus.done, TaskStatus.errored},
    TaskStatus.halted: {TaskStatus.running, TaskStatus.errored, TaskStatus.done},
    TaskStatus.done: set(),
    TaskStatus.errored: set(),
}


class InvalidTaskTransition(ValueError):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    tier: Mapped[Tier] = mapped_column(Enum(Tier), default=Tier.free, nullable=False)
    stripe_customer_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    profiles: Mapped[list["Profile"]] = relationship(back_populates="user")
    tasks: Mapped[list["Task"]] = relationship(back_populates="user")


class MagicLinkToken(Base):
    __tablename__ = "magic_link_tokens"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)


class Profile(Base):
    """A user's persistent pinchtab profile.

    The pinchtab_profile_name is what we pass to pinchtab as profileId on
    instance start. The instance_id is set when the instance is live and
    cleared when evicted. Profile data (cookies, extensions, localStorage)
    lives in pinchtab's profile dir — we never reach into that.
    """

    __tablename__ = "profiles"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id"), index=True, nullable=False
    )
    # pinchtab profile identifier. We generate this; never user-controlled
    # input (path-traversal protection).
    pinchtab_profile_name: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    pinchtab_instance_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    user: Mapped[User] = relationship(back_populates="profiles")
    # passive_deletes="all": when a Profile is deleted, do NOT touch related
    # tasks (no auto SET NULL of profile_id). Tasks become orphaned rows that
    # still serve as historical record. SQLite with default FK enforcement
    # off permits the dangling FK; the API never joins through it.
    tasks: Mapped[list["Task"]] = relationship(
        back_populates="profile", passive_deletes="all"
    )


class Task(Base):
    """One user-issued automation request, bound to a single tab."""

    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id"), index=True, nullable=False
    )
    profile_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("profiles.id"), index=True, nullable=False
    )
    status: Mapped[TaskStatus] = mapped_column(
        Enum(TaskStatus), default=TaskStatus.pending, nullable=False
    )

    # Pinchtab handles for this task's tab. Both set when the tab is opened.
    pinchtab_tab_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    task_description: Mapped[str] = mapped_column(Text, nullable=False)
    start_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    final_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    minutes_consumed: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    user: Mapped[User] = relationship(back_populates="tasks")
    profile: Mapped[Profile] = relationship(back_populates="tasks")

    __table_args__ = (Index("ix_task_user_status", "user_id", "status"),)

    def transition(self, new: TaskStatus) -> None:
        allowed = TASK_TRANSITIONS.get(self.status, set())
        if new not in allowed:
            raise InvalidTaskTransition(
                f"cannot transition task {self.id} from {self.status.value} to {new.value}"
            )
        self.status = new
        now = _utcnow()
        if new == TaskStatus.running and self.started_at is None:
            self.started_at = now
        if new in (TaskStatus.done, TaskStatus.errored):
            self.ended_at = now


class TaskEvent(Base):
    """Append-only timeline of agent events per task.

    Mirrors every SSE event the runner emits (step, llm_call, llm_done,
    tool_call, tool_result, loop_detected, hint_delivered, awaiting_input,
    terminal, etc.) so the dashboard can render full run history after
    the live stream is gone. payload_json is the event dict minus `type`,
    truncated at 8 KB per row.
    """

    __tablename__ = "task_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("tasks.id"), index=True, nullable=False
    )
    step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False, index=True
    )

    __table_args__ = (Index("ix_task_event_task_created", "task_id", "created_at"),)


class StripeEvent(Base):
    """Idempotency table for Stripe webhook events (CEO review S4)."""

    __tablename__ = "stripe_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    received_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)


class UsageMetric(Base):
    """Per-customer minute accounting (CEO review O2)."""

    __tablename__ = "usage_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id"), index=True)
    task_id: Mapped[str] = mapped_column(String(32), ForeignKey("tasks.id"))
    minutes: Mapped[float] = mapped_column(Float, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
