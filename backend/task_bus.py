"""In-process pub/sub bus + task registry for live agent progress.

Both are intentionally in-process and unbounded by feature flags or message
brokers — MVP scope. When we move to multi-worker uvicorn or out-of-process
runners, swap TaskBus for Redis pub/sub and TaskRegistry for a distributed
lock service.

Concurrency assumption: single uvicorn worker. With multiple workers the
in-memory state is per-process and breaks. Add MAX_CONCURRENT_SESSIONS in
config to keep cluster headroom in line with single-process capacity.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

log = logging.getLogger("task_bus")

# A sentinel that subscribers see when a task reaches terminal state.
END_SENTINEL = "__END__"


class TaskBus:
    """Per-task fan-out queue. Subscribers each get their own queue; publish
    enqueues on all subscriber queues. Slow consumers drop events rather than
    block the publisher (visibility > correctness for progress events)."""

    def __init__(self, subscriber_buffer: int = 256):
        self._subs: dict[str, list[asyncio.Queue]] = {}
        self._buffer = subscriber_buffer

    def publish(self, task_id: str, event: dict[str, Any]) -> int:
        """Send event to all current subscribers of task_id. Returns count
        of subscribers that received the event."""
        delivered = 0
        for q in self._subs.get(task_id, ()):
            try:
                q.put_nowait(event)
                delivered += 1
            except asyncio.QueueFull:
                # Drop the event for this consumer. Logged once at info.
                log.info("dropped event for slow consumer on task=%s", task_id)
        return delivered

    def end(self, task_id: str) -> None:
        """Signal end-of-stream. Subscribers receive END_SENTINEL and may exit."""
        for q in self._subs.get(task_id, ()):
            try:
                q.put_nowait({"type": END_SENTINEL})
            except asyncio.QueueFull:
                pass

    @asynccontextmanager
    async def subscribe(self, task_id: str) -> AsyncIterator[asyncio.Queue]:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._buffer)
        self._subs.setdefault(task_id, []).append(q)
        try:
            yield q
        finally:
            try:
                self._subs[task_id].remove(q)
                if not self._subs[task_id]:
                    del self._subs[task_id]
            except (KeyError, ValueError):
                pass

    def subscriber_count(self, task_id: str) -> int:
        return len(self._subs.get(task_id, ()))


class TaskRegistry:
    """Holds asyncio.Task handles for running agent loops. Allows cancellation
    when the user halts a task. Auto-removes entries on task completion."""

    def __init__(self):
        self._tasks: dict[str, asyncio.Task] = {}

    def register(self, task_id: str, t: asyncio.Task) -> None:
        if task_id in self._tasks and not self._tasks[task_id].done():
            raise ValueError(f"task {task_id} already registered and not done")
        self._tasks[task_id] = t
        t.add_done_callback(lambda _f, tid=task_id: self._cleanup(tid))

    def _cleanup(self, task_id: str) -> None:
        t = self._tasks.get(task_id)
        if t is not None and t.done():
            self._tasks.pop(task_id, None)

    def is_running(self, task_id: str) -> bool:
        t = self._tasks.get(task_id)
        return t is not None and not t.done()

    async def cancel(self, task_id: str) -> bool:
        """Request cancellation. Returns True if a task was running."""
        t = self._tasks.get(task_id)
        if t is None or t.done():
            return False
        t.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(t), timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass
        return True

    def running_ids(self) -> list[str]:
        return [tid for tid, t in self._tasks.items() if not t.done()]


# Module singletons. Imported by tasks router and runner.
bus = TaskBus()
registry = TaskRegistry()
