"""Async HTTP client for the pinchtab daemon.

Trust model: pinchtab is bound to 127.0.0.1 and reached only by this backend
process. We do NOT layer auth between backend and pinchtab — authorization
happens upstream (backend authenticates the user, then maps user_id to the
specific profile/tab pinchtab handle). The client is intentionally narrow:
only the endpoints we use are wrapped, anti-detection surface is omitted.

Endpoints intentionally NOT wrapped:
  - GET /stealth/status
  - POST /fingerprint/rotate
  - POST /tabs/{id}/solve  (captcha auto-solver — Premise 4)
  - GET /solvers
These exist in pinchtab but are part of the anti-detection / evasion
surface. Exposing them through our control plane would expand the gray-zone
risk (CEO review Premise 4, Gray-Zone Gate). Keep them off the client.

Endpoint reference (extracted from pinchtab/internal/handlers + orchestrator):
  Instance / profile mgmt:
    GET    /instances
    GET    /instances/{id}
    POST   /instances/start              {profileId, mode?, securityPolicy?}
    POST   /instances/{id}/stop
    POST   /instances/{id}/tabs/open     {url}                       → tab info
    POST   /profiles/{id}/start
    GET    /profiles/{id}/instance
  Per-tab:
    POST   /tabs/{id}/navigate           {url}
    GET    /tabs/{id}/snapshot           ?interactive=&compact=&maxTokens=
    GET    /tabs/{id}/screenshot         ?quality=
    GET    /tabs/{id}/text
    POST   /tabs/{id}/action             {type, ref, ...}
    POST   /tabs/{id}/close
    POST   /tabs/{id}/network/route      {pattern, action, ...}      ← denylist
    DELETE /tabs/{id}/network/route      ?pattern=
    GET    /tabs/{id}/network/route
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import httpx

from backend.denylist import DenylistPolicy


class PinchtabError(RuntimeError):
    def __init__(self, status: int, body: str, endpoint: str):
        super().__init__(f"pinchtab {endpoint} → HTTP {status}: {body[:200]}")
        self.status = status
        self.body = body
        self.endpoint = endpoint


@dataclass(frozen=True)
class RouteRule:
    """Mirrors pinchtab's bridge.RouteRule wire format."""

    pattern: str
    action: Literal["block", "fulfill", "continue"] = "block"
    resourceType: str | None = None   # "document" | "xhr" | "fetch" | "image" | ...
    method: str | None = None         # "GET" | "POST" | ...
    status: int | None = None         # for fulfill rules
    body: str | None = None           # for fulfill rules
    contentType: str | None = None    # for fulfill rules

    def to_payload(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class PinchtabClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9867",
        timeout: float = 30.0,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        """`transport` is a hook for tests — pass an httpx.MockTransport to
        intercept calls without spinning up a real network client."""
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=timeout, base_url=self._base, transport=transport
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "PinchtabClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    # ---- Internal helpers ----

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        r = await self._client.request(method, path, **kwargs)
        if r.status_code >= 400:
            raise PinchtabError(r.status_code, r.text, f"{method} {path}")
        return r

    async def _json(self, method: str, path: str, **kwargs) -> Any:
        r = await self._request(method, path, **kwargs)
        if not r.content:
            return None
        ct = r.headers.get("content-type", "")
        if "application/json" in ct:
            return r.json()
        return r.text

    # ---- Health ----

    async def health(self) -> dict[str, Any]:
        return await self._json("GET", "/health")

    # ---- Instances ----

    async def list_instances(self) -> list[dict[str, Any]]:
        return await self._json("GET", "/instances") or []

    async def get_instance(self, instance_id: str) -> dict[str, Any] | None:
        try:
            return await self._json("GET", f"/instances/{instance_id}")
        except PinchtabError as e:
            if e.status == 404:
                return None
            raise

    async def start_instance(
        self, *, profile_id: str, mode: str = "headless"
    ) -> dict[str, Any]:
        """Start (or resume) an instance for the given profile.

        If the profile doesn't exist yet, pinchtab creates the profile directory
        on first launch. profile_id here is pinchtab's profileId — we generate
        this server-side and pass it (never accept user-controlled strings).
        """
        return await self._json(
            "POST",
            "/instances/start",
            json={"profileId": profile_id, "mode": mode},
        )

    async def stop_instance(self, instance_id: str) -> dict[str, Any]:
        return await self._json("POST", f"/instances/{instance_id}/stop")

    async def get_profile_instance(self, profile_id: str) -> dict[str, Any] | None:
        try:
            return await self._json("GET", f"/profiles/{profile_id}/instance")
        except PinchtabError as e:
            if e.status == 404:
                return None
            raise

    # ---- Tabs ----

    async def open_tab(self, instance_id: str, url: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if url is not None:
            body["url"] = url
        return await self._json("POST", f"/instances/{instance_id}/tabs/open", json=body)

    async def close_tab(self, tab_id: str) -> dict[str, Any]:
        return await self._json("POST", f"/tabs/{tab_id}/close")

    async def navigate(self, tab_id: str, url: str) -> dict[str, Any]:
        return await self._json("POST", f"/tabs/{tab_id}/navigate", json={"url": url})

    # ---- Inspection ----

    async def snapshot(
        self,
        tab_id: str,
        *,
        interactive: bool = True,
        compact: bool = True,
        max_tokens: int | None = None,
    ) -> str:
        params: dict[str, Any] = {
            "interactive": "true" if interactive else "false",
            "compact": "true" if compact else "false",
        }
        if max_tokens:
            params["maxTokens"] = str(max_tokens)
        r = await self._request("GET", f"/tabs/{tab_id}/snapshot", params=params)
        return r.text

    async def screenshot(self, tab_id: str, *, quality: int = 70) -> bytes:
        r = await self._request(
            "GET", f"/tabs/{tab_id}/screenshot", params={"quality": str(quality)}
        )
        return r.content

    async def text(self, tab_id: str) -> str:
        r = await self._request("GET", f"/tabs/{tab_id}/text")
        return r.text

    # ---- Actions ----

    async def _action(self, tab_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._json("POST", f"/tabs/{tab_id}/action", json=payload)

    async def click(self, tab_id: str, ref: str) -> dict[str, Any]:
        return await self._action(tab_id, {"type": "click", "ref": ref})

    async def type_text(self, tab_id: str, ref: str, text: str) -> dict[str, Any]:
        return await self._action(tab_id, {"type": "type", "ref": ref, "text": text})

    async def fill(self, tab_id: str, ref: str, text: str) -> dict[str, Any]:
        return await self._action(tab_id, {"type": "fill", "ref": ref, "text": text})

    async def press_key(self, tab_id: str, key: str) -> dict[str, Any]:
        return await self._action(tab_id, {"type": "press", "key": key})

    async def scroll(self, tab_id: str, amount: str | int) -> dict[str, Any]:
        return await self._action(tab_id, {"type": "scroll", "amount": str(amount)})

    async def select_option(self, tab_id: str, ref: str, value: str) -> dict[str, Any]:
        return await self._action(tab_id, {"type": "select", "ref": ref, "value": value})

    # ---- Network interception (denylist enforcement) ----

    async def add_route_rule(self, tab_id: str, rule: RouteRule) -> dict[str, Any]:
        return await self._json(
            "POST", f"/tabs/{tab_id}/network/route", json=rule.to_payload()
        )

    async def remove_route_rule(self, tab_id: str, pattern: str) -> dict[str, Any]:
        return await self._json(
            "DELETE", f"/tabs/{tab_id}/network/route", params={"pattern": pattern}
        )

    async def list_route_rules(self, tab_id: str) -> list[dict[str, Any]]:
        return await self._json("GET", f"/tabs/{tab_id}/network/route") or []

    async def apply_denylist(self, tab_id: str, policy: DenylistPolicy) -> int:
        """Install one block rule per denied registrable domain.

        Pattern shape: `*://*.{domain}/*` to catch all subdomains plus
        `*://{domain}/*` for the apex. Both go in.

        Returns the number of rules installed.
        """
        installed = 0
        for domain in sorted(policy.deny):
            if domain in policy.allow:
                continue
            for pattern in (f"*://*.{domain}/*", f"*://{domain}/*"):
                await self.add_route_rule(
                    tab_id, RouteRule(pattern=pattern, action="block")
                )
                installed += 1
        return installed
