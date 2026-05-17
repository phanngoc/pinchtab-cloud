"""Profile management endpoints.

Reset strategy: rotate the pinchtab profile_name to a fresh UUID so the next
session uses a clean profile directory. We keep the Profile row (and its id)
intact — Task history still resolves. The old pinchtab dir is orphaned;
a scheduled cleanup job sweeps ~/.pinchtab/profiles/ for dirs not referenced
by any Profile.pinchtab_profile_name.

Delete strategy: stop instance, delete row. Next task creation re-creates
a fresh Profile via the lazy-create path in tasks.py.
"""
from __future__ import annotations

import logging
import secrets
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from backend.db import get_db
from backend.models import Profile, Task, TaskStatus, User
from backend.pinchtab_client import PinchtabClient, PinchtabError
from backend.security import current_user

log = logging.getLogger("profiles")

router = APIRouter(prefix="/profiles", tags=["profiles"])


class ProfileOut(BaseModel):
    id: str
    pinchtab_profile_name: str
    instance_live: bool
    last_used_at: datetime | None
    created_at: datetime


def _to_out(p: Profile) -> ProfileOut:
    return ProfileOut(
        id=p.id,
        pinchtab_profile_name=p.pinchtab_profile_name,
        instance_live=bool(p.pinchtab_instance_id),
        last_used_at=p.last_used_at,
        created_at=p.created_at,
    )


def get_pinchtab(request: Request) -> PinchtabClient:
    """Resolve the pinchtab client from app state. Tests inject via
    `app.state.pinchtab = FakePinchtabClient()`."""
    client = getattr(request.app.state, "pinchtab", None)
    if client is None:
        raise HTTPException(status_code=503, detail="pinchtab_unavailable")
    return client


def _user_owns(profile: Profile | None, user: User) -> Profile:
    if profile is None or profile.user_id != user.id:
        raise HTTPException(status_code=404, detail="not_found")
    return profile


async def _try_stop_instance(client: PinchtabClient, instance_id: str) -> None:
    try:
        await client.stop_instance(instance_id)
    except PinchtabError as e:
        if e.status == 404:
            return
        log.warning("stop_instance(%s) failed: %s", instance_id, e)


def _has_live_tasks(db: DbSession, profile_id: str) -> bool:
    live = db.scalar(
        select(Task.id)
        .where(Task.profile_id == profile_id)
        .where(Task.status.in_((TaskStatus.pending, TaskStatus.running)))
        .limit(1)
    )
    return live is not None


@router.get("", response_model=list[ProfileOut])
async def list_profiles(
    user: User = Depends(current_user), db: DbSession = Depends(get_db)
) -> list[ProfileOut]:
    rows = db.scalars(
        select(Profile).where(Profile.user_id == user.id).order_by(Profile.created_at)
    ).all()
    return [_to_out(p) for p in rows]


@router.post("/{profile_id}/reset", response_model=ProfileOut)
async def reset_profile(
    profile_id: str,
    user: User = Depends(current_user),
    db: DbSession = Depends(get_db),
    pinchtab: PinchtabClient = Depends(get_pinchtab),
) -> ProfileOut:
    """Wipe the user's pinchtab profile (cookies, localStorage, extensions
    state). Implemented by rotating to a fresh profile_name; the next task
    starts a fresh pinchtab profile directory.

    Rejects if any task is currently live on this profile (pending/running).
    """
    profile = _user_owns(db.get(Profile, profile_id), user)

    if _has_live_tasks(db, profile.id):
        raise HTTPException(
            status_code=409,
            detail="profile_has_live_tasks",
        )

    if profile.pinchtab_instance_id:
        await _try_stop_instance(pinchtab, profile.pinchtab_instance_id)

    profile.pinchtab_profile_name = f"u_{secrets.token_hex(8)}"
    profile.pinchtab_instance_id = None
    profile.last_used_at = None
    db.commit()
    db.refresh(profile)
    return _to_out(profile)


@router.delete("/{profile_id}", status_code=204)
async def delete_profile(
    profile_id: str,
    user: User = Depends(current_user),
    db: DbSession = Depends(get_db),
    pinchtab: PinchtabClient = Depends(get_pinchtab),
) -> None:
    """Delete the user's pinchtab profile. Their next task lazily creates
    a brand-new Profile row.

    Tasks history persists (Task.profile_id FK is preserved by SQLAlchemy
    default behavior — those rows remain readable but reference a deleted
    profile). Rejects if any task is live.
    """
    profile = _user_owns(db.get(Profile, profile_id), user)

    if _has_live_tasks(db, profile.id):
        raise HTTPException(
            status_code=409,
            detail="profile_has_live_tasks",
        )

    if profile.pinchtab_instance_id:
        await _try_stop_instance(pinchtab, profile.pinchtab_instance_id)

    db.delete(profile)
    db.commit()
