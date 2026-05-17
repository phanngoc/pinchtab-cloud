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
    action: Literal["abort", "fulfill", "continue"] = "abort"
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
        token: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        """`transport` is a hook for tests — pass an httpx.MockTransport to
        intercept calls without spinning up a real network client.

        `token` is the pinchtab API token (server.token in ~/.pinchtab/config.json).
        Sent on every request as `Authorization: Bearer <token>`. Required by
        pinchtab 0.8+; harmless on older versions.
        """
        self._base = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {token}"} if token else None
        self._client = httpx.AsyncClient(
            timeout=timeout,
            base_url=self._base,
            transport=transport,
            headers=headers,
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

    async def create_profile(
        self, *, name: str, description: str = ""
    ) -> dict[str, Any]:
        """Create a new pinchtab profile. Returns the full profile object
        including the `id` (which is what `/instances/start` accepts as
        `profileId`). Pinchtab does not auto-create profiles on instance
        start — POST /profiles is required first."""
        return await self._json(
            "POST",
            "/profiles",
            json={"name": name, "description": description},
        )

    async def list_profiles(self) -> list[dict[str, Any]]:
        return await self._json("GET", "/profiles") or []

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

    # Pinchtab role names worth surfacing to the agent. Everything else
    # (StaticText decorations, generic listitem containers, etc.) gets
    # dropped to keep the snap compact.
    _SNAP_ROLES = frozenset({
        "button", "link", "textbox", "searchbox", "combobox",
        "checkbox", "radio", "switch", "menuitem", "tab",
        "img", "image", "heading", "form",
    })

    async def snapshot(
        self,
        tab_id: str,
        *,
        interactive: bool = True,
        compact: bool = True,
        max_tokens: int | None = None,
    ) -> str:
        """Returns a SHORT human-readable accessibility snapshot.

        Pinchtab's /snapshot endpoint returns verbose JSON even with
        compact=true (each node has nodeId/frameId/frameUrl metadata that
        the agent doesn't need). We parse the JSON and emit one line per
        interactive node: `eN:role "name"`. Empirical: ~5x smaller than
        raw JSON, which makes a huge latency difference when feeding the
        text to the claude CLI subprocess (no prompt caching there).
        """
        import json as _json

        params: dict[str, Any] = {
            "interactive": "true" if interactive else "false",
            "compact": "true" if compact else "false",
        }
        if max_tokens:
            params["maxTokens"] = str(max_tokens)
        r = await self._request("GET", f"/tabs/{tab_id}/snapshot", params=params)
        raw = r.text
        try:
            data = _json.loads(raw)
        except Exception:
            # Old binaries / unexpected payloads — pass through verbatim.
            return raw
        lines: list[str] = []
        title = data.get("title", "")
        url = data.get("url", "")
        nodes = data.get("nodes", []) or []
        lines.append(f"# {title} | {url} | {len(nodes)} nodes")
        for n in nodes:
            role = (n.get("role") or "").lower()
            tag = (n.get("tag") or "").lower()
            name = n.get("name") or n.get("text") or n.get("placeholder") or ""
            ref = n.get("ref", "")
            if role not in self._SNAP_ROLES and tag not in {"a", "button", "input", "textarea", "select"}:
                continue
            if not ref:
                continue
            # Trim each line to keep snap small even on huge pages.
            label = " ".join(name.split())[:80]
            lines.append(f'{ref}:{role or tag} "{label}"')
        if data.get("truncated"):
            lines.append("# (truncated by maxTokens)")
        return "\n".join(lines)

    async def screenshot(self, tab_id: str, *, quality: int = 70) -> bytes:
        """Returns raw image bytes (JPEG or PNG depending on pinchtab config).

        Pinchtab returns JSON {"base64": "...", "encoding": "...", "size": N}
        rather than raw image bytes, so we decode here. Callers detect the
        actual MIME type from magic bytes."""
        import base64 as _b64

        r = await self._request(
            "GET", f"/tabs/{tab_id}/screenshot", params={"quality": str(quality)}
        )
        # Try JSON wrapper first (current pinchtab behavior).
        ct = r.headers.get("content-type", "")
        if "application/json" in ct:
            data = r.json()
            b64 = data.get("base64") or data.get("data") or ""
            return _b64.b64decode(b64) if b64 else b""
        # Fallback: raw image bytes (older or different pinchtab build).
        return r.content

    async def text(self, tab_id: str, *, selector: str | None = None) -> str:
        """Return readable page text. With `selector`, returns text from one
        element (ref like 'e7' or CSS like '#article-body')."""
        params: dict[str, Any] = {}
        if selector:
            params["selector"] = selector
        r = await self._request("GET", f"/tabs/{tab_id}/text", params=params or None)
        # Pinchtab may return either text or JSON {text: "..."}; handle both.
        ct = r.headers.get("content-type", "")
        if "application/json" in ct:
            try:
                data = r.json()
                return data.get("text", "") if isinstance(data, dict) else r.text
            except Exception:
                return r.text
        return r.text

    async def find(self, tab_id: str, query: str) -> dict[str, Any]:
        """Semantic find — natural-language description → ref. Calls
        pinchtab's POST /tabs/{id}/find endpoint."""
        return await self._json(
            "POST", f"/tabs/{tab_id}/find", json={"query": query}
        )

    # ---- Actions ----

    async def _action(self, tab_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._json("POST", f"/tabs/{tab_id}/action", json=payload)

    # Pinchtab's POST /tabs/{id}/action expects `kind` as the action name field
    # (constants in pinchtab/internal/bridge/action_registry.go: click/type/fill/
    # press/select/scroll). Sending `type` returns 400 missing_field 'kind'.
    async def click(self, tab_id: str, ref: str) -> dict[str, Any]:
        return await self._action(tab_id, {"kind": "click", "ref": ref})

    async def type_text(self, tab_id: str, ref: str, text: str) -> dict[str, Any]:
        return await self._action(tab_id, {"kind": "type", "ref": ref, "text": text})

    async def fill(self, tab_id: str, ref: str, text: str) -> dict[str, Any]:
        return await self._action(tab_id, {"kind": "fill", "ref": ref, "text": text})

    async def press_key(self, tab_id: str, key: str) -> dict[str, Any]:
        return await self._action(tab_id, {"kind": "press", "key": key})

    async def scroll(self, tab_id: str, amount: str | int) -> dict[str, Any]:
        return await self._action(tab_id, {"kind": "scroll", "amount": str(amount)})

    async def select_option(self, tab_id: str, ref: str, value: str) -> dict[str, Any]:
        return await self._action(tab_id, {"kind": "select", "ref": ref, "value": value})

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
                    tab_id, RouteRule(pattern=pattern, action="abort")
                )
                installed += 1
        return installed
