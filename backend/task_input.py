"""In-memory registry for human-in-the-loop input requests.

Lifecycle:
   1. Agent calls request_user_input(prompt, fields). Runner calls
      registry.register(task_id, prompt, fields) and publishes an
      `awaiting_input` SSE event.
   2. Runner awaits registry.wait(task_id) with a hard timeout.
   3. User submits via POST /tasks/{id}/provide-input → endpoint calls
      registry.provide(task_id, values) which sets the event.
   4. Runner wakes, receives the values dict, returns it as the
      tool_result to Claude, continues the loop.

Security caveat (documented for the operator):
   Values pass through Claude's message history. Do NOT use this for
   production-grade secrets without first adding placeholder substitution
   at the runner's tool-call layer (a follow-up enhancement). For dev,
   it's fine: user explicitly types each value per task.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FieldSpec:
    name: str
    label: str
    type: str = "text"  # text | password | email | number

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FieldSpec":
        return cls(
            name=str(d.get("name", "")),
            label=str(d.get("label", d.get("name", ""))),
            type=str(d.get("type", "text")),
        )


@dataclass
class PendingInput:
    prompt: str
    fields: list[FieldSpec]
    event: asyncio.Event = field(default_factory=asyncio.Event)
    values: dict[str, str] | None = None

    def public_view(self) -> dict[str, Any]:
        """Dict shape sent to dashboard via SSE / poll."""
        return {
            "prompt": self.prompt,
            "fields": [
                {"name": f.name, "label": f.label, "type": f.type}
                for f in self.fields
            ],
        }


class InputRegistry:
    """In-process registry. One pending input per task at any time.

    Concurrency: a task only ever has one outstanding request at a time
    (the runner is sequential within a task's coroutine). Registering a
    second request for the same task replaces the previous one's event
    (caller should not do this; agent should resolve current input first).
    """

    def __init__(self):
        self._pending: dict[str, PendingInput] = {}

    def register(self, task_id: str, prompt: str, fields: list[dict[str, Any]]) -> PendingInput:
        specs = [FieldSpec.from_dict(f) for f in fields if isinstance(f, dict)]
        # Keep only sane field types.
        ALLOWED = {"text", "password", "email", "number"}
        for s in specs:
            if s.type not in ALLOWED:
                s.type = "text"
        p = PendingInput(prompt=prompt, fields=specs)
        self._pending[task_id] = p
        return p

    def get(self, task_id: str) -> PendingInput | None:
        return self._pending.get(task_id)

    async def wait(self, task_id: str) -> dict[str, str]:
        p = self._pending.get(task_id)
        if p is None:
            raise RuntimeError(f"no pending input for task {task_id}")
        await p.event.wait()
        # The PendingInput is still referenced locally here even though
        # provide() / cancel() have already popped it from the registry —
        # we read values from the local reference, no race possible.
        return p.values or {}

    def provide(self, task_id: str, values: dict[str, Any]) -> bool:
        """Returns True iff a pending input existed and values were delivered."""
        p = self._pending.get(task_id)
        if p is None or p.event.is_set():
            return False
        # Restrict to declared field names; drop unexpected keys.
        declared = {f.name for f in p.fields}
        p.values = {k: str(v) for k, v in values.items() if k in declared}
        # Remove from registry BEFORE signaling — so a subsequent
        # provide() / get() observes the absence immediately, no race.
        self._pending.pop(task_id, None)
        p.event.set()
        return True

    def cancel(self, task_id: str) -> None:
        """Discard a pending input without delivering (e.g. timeout)."""
        p = self._pending.pop(task_id, None)
        if p is not None and not p.event.is_set():
            # Signal anyone waiting; they'll see empty values.
            p.values = {}
            p.event.set()


# Module-level singleton, used by runner + tasks router.
registry = InputRegistry()
