"""Tests for the pinchtab HTTP client. httpx is mocked via MockTransport so
no live pinchtab daemon is required.
"""
import httpx
import pytest

from backend.denylist import DenylistPolicy
from backend.pinchtab_client import PinchtabClient, PinchtabError, RouteRule


def make_client(handler) -> PinchtabClient:
    """Wire a PinchtabClient around an httpx MockTransport with the given handler.

    Uses the transport= constructor hook so no real network client is allocated.
    """
    return PinchtabClient(transport=httpx.MockTransport(handler))


def _recorder():
    """Return (handler, calls) — handler records every request into calls list."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        # Default OK JSON; specific tests can replace via closure.
        return httpx.Response(200, json={"ok": True})

    return handler, calls


@pytest.mark.asyncio
async def test_health_calls_health_endpoint():
    handler, calls = _recorder()
    c = make_client(handler)
    await c.health()
    await c.aclose()
    assert calls[0].method == "GET"
    assert calls[0].url.path == "/health"


@pytest.mark.asyncio
async def test_start_instance_passes_profile_id_in_body():
    handler, calls = _recorder()
    c = make_client(handler)
    await c.start_instance(profile_id="u_alice", mode="headless")
    await c.aclose()
    assert calls[0].method == "POST"
    assert calls[0].url.path == "/instances/start"
    body = calls[0].read()
    assert b'"profileId":"u_alice"' in body.replace(b" ", b"")
    assert b'"mode":"headless"' in body.replace(b" ", b"")


@pytest.mark.asyncio
async def test_open_tab_uses_instance_scoped_endpoint():
    handler, calls = _recorder()
    c = make_client(handler)
    await c.open_tab("inst_123", "https://viblo.asia/followings")
    await c.aclose()
    assert calls[0].method == "POST"
    assert calls[0].url.path == "/instances/inst_123/tabs/open"
    assert b"viblo.asia/followings" in calls[0].read()


@pytest.mark.asyncio
async def test_navigate_uses_tab_scoped_endpoint():
    handler, calls = _recorder()
    c = make_client(handler)
    await c.navigate("tab_abc", "https://example.com")
    await c.aclose()
    assert calls[0].method == "POST"
    assert calls[0].url.path == "/tabs/tab_abc/navigate"


@pytest.mark.asyncio
async def test_snapshot_reformats_json_to_compact_text():
    """Pinchtab returns verbose JSON; client reformats to one terse line
    per interactive node."""
    import json

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps({
            "title": "Some Page",
            "url": "https://example.com/",
            "count": 4,
            "nodes": [
                {"ref": "e0", "role": "RootWebArea", "name": "Some Page"},
                {"ref": "e1", "role": "button", "name": "Login", "tag": "button"},
                {"ref": "e2", "role": "textbox", "name": "Email", "tag": "input"},
                {"ref": "e3", "role": "StaticText", "name": "Footer text"},
            ],
        })
        return httpx.Response(200, text=body)

    c = make_client(handler)
    out = await c.snapshot("tab_x", interactive=True, compact=True)
    await c.aclose()
    # Header line
    assert "Some Page" in out
    assert "https://example.com/" in out
    # Interactive elements included
    assert 'e1:button "Login"' in out
    assert 'e2:textbox "Email"' in out
    # Decorative nodes dropped
    assert "Footer text" not in out
    assert "RootWebArea" not in out


@pytest.mark.asyncio
async def test_snapshot_query_includes_max_tokens():
    handler, calls = _recorder()

    def text_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, text="snapshot")

    c = make_client(text_handler)
    await c.snapshot("tab_x", max_tokens=4000)
    await c.aclose()
    q = calls[0].url.params
    assert q["interactive"] == "true"
    assert q["compact"] == "true"
    assert q["maxTokens"] == "4000"


@pytest.mark.asyncio
async def test_click_dispatches_to_action_endpoint():
    handler, calls = _recorder()
    c = make_client(handler)
    await c.click("tab_x", "e7")
    await c.aclose()
    assert calls[0].url.path == "/tabs/tab_x/action"
    payload = calls[0].read()
    assert b'"kind":"click"' in payload.replace(b" ", b"")
    assert b'"ref":"e7"' in payload.replace(b" ", b"")


@pytest.mark.asyncio
async def test_type_text_includes_text_field():
    handler, calls = _recorder()
    c = make_client(handler)
    await c.type_text("tab_x", "e6", "hello world")
    await c.aclose()
    payload = calls[0].read()
    assert b'"kind":"type"' in payload.replace(b" ", b"")
    assert b"hello world" in payload


@pytest.mark.asyncio
async def test_add_route_rule_serializes_correctly():
    handler, calls = _recorder()
    c = make_client(handler)
    rule = RouteRule(pattern="*://*.shopee.vn/*", action="abort")
    await c.add_route_rule("tab_x", rule)
    await c.aclose()
    assert calls[0].url.path == "/tabs/tab_x/network/route"
    body = calls[0].read()
    assert b"shopee.vn" in body
    assert b'"action":"abort"' in body.replace(b" ", b"")
    # Optional fields not set should not be in payload.
    assert b"resourceType" not in body
    assert b"method" not in body


@pytest.mark.asyncio
async def test_apply_denylist_installs_two_patterns_per_domain():
    """Each denied registrable domain produces one apex rule + one wildcard
    subdomain rule. With default policy (10 denied domains), that's 20 calls."""
    handler, calls = _recorder()
    c = make_client(handler)
    policy = DenylistPolicy(deny=frozenset({"shopee.vn", "facebook.com"}))
    n = await c.apply_denylist("tab_x", policy)
    await c.aclose()
    assert n == 4  # 2 domains × 2 patterns each
    assert len(calls) == 4
    patterns = []
    for call in calls:
        body = call.read().decode()
        # crude extract; not parsing JSON
        start = body.find('"pattern":"') + len('"pattern":"')
        end = body.find('"', start)
        patterns.append(body[start:end])
    assert "*://shopee.vn/*" in patterns
    assert "*://*.shopee.vn/*" in patterns
    assert "*://facebook.com/*" in patterns
    assert "*://*.facebook.com/*" in patterns


@pytest.mark.asyncio
async def test_remove_route_rule_uses_query_param():
    handler, calls = _recorder()
    c = make_client(handler)
    await c.remove_route_rule("tab_x", "*://*.shopee.vn/*")
    await c.aclose()
    assert calls[0].method == "DELETE"
    assert calls[0].url.path == "/tabs/tab_x/network/route"
    assert calls[0].url.params["pattern"] == "*://*.shopee.vn/*"


@pytest.mark.asyncio
async def test_close_tab_uses_post_close():
    handler, calls = _recorder()
    c = make_client(handler)
    await c.close_tab("tab_x")
    await c.aclose()
    assert calls[0].method == "POST"
    assert calls[0].url.path == "/tabs/tab_x/close"


@pytest.mark.asyncio
async def test_get_instance_returns_none_on_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not_found"})

    c = make_client(handler)
    out = await c.get_instance("inst_missing")
    await c.aclose()
    assert out is None


@pytest.mark.asyncio
async def test_non_404_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    c = make_client(handler)
    with pytest.raises(PinchtabError) as exc:
        await c.health()
    await c.aclose()
    assert exc.value.status == 500
