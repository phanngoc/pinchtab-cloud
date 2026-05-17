"""Per-user rate limiting (CEO review S2: per-user 60 req/min).

Lightweight token bucket keyed by user id, stored in process memory. For
multi-instance deployments, swap to Redis-backed limits.RedisStorage.
"""
from __future__ import annotations

from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter

from backend.config import get_settings

_storage = MemoryStorage()
_limiter = MovingWindowRateLimiter(_storage)


def check(user_id: str) -> bool:
    """Returns True if the request is within the limit, False if it should be rejected."""
    limit = parse(get_settings().rate_limit_per_user)
    return _limiter.hit(limit, "user", user_id)
