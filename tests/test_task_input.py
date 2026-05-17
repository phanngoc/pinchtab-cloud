"""Tests for the human-in-the-loop input registry."""
import asyncio

import pytest

from backend.task_input import InputRegistry


@pytest.mark.asyncio
async def test_register_then_provide_delivers_values():
    reg = InputRegistry()
    reg.register(
        "t1",
        "need login",
        [{"name": "email", "label": "Email"}, {"name": "password", "label": "Password", "type": "password"}],
    )

    async def consumer():
        return await reg.wait("t1")

    waiter = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    assert reg.provide("t1", {"email": "a@b.c", "password": "secret"}) is True
    values = await waiter
    assert values == {"email": "a@b.c", "password": "secret"}


@pytest.mark.asyncio
async def test_provide_drops_unknown_fields():
    reg = InputRegistry()
    reg.register("t1", "p", [{"name": "email", "label": "Email"}])

    async def consumer():
        return await reg.wait("t1")

    waiter = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    reg.provide("t1", {"email": "a@b.c", "secret_token": "leaked"})
    values = await waiter
    assert "email" in values
    assert "secret_token" not in values  # unknown field dropped


@pytest.mark.asyncio
async def test_provide_returns_false_when_no_pending():
    reg = InputRegistry()
    assert reg.provide("nope", {"x": "y"}) is False


@pytest.mark.asyncio
async def test_provide_second_time_returns_false():
    reg = InputRegistry()
    reg.register("t1", "p", [{"name": "x", "label": "x"}])

    async def consumer():
        return await reg.wait("t1")

    asyncio.create_task(consumer())
    await asyncio.sleep(0)
    assert reg.provide("t1", {"x": "1"}) is True
    # After delivery the pending entry is removed.
    assert reg.get("t1") is None
    assert reg.provide("t1", {"x": "2"}) is False


@pytest.mark.asyncio
async def test_cancel_unblocks_with_empty_dict():
    reg = InputRegistry()
    reg.register("t1", "p", [{"name": "x", "label": "x"}])

    async def consumer():
        return await reg.wait("t1")

    waiter = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    reg.cancel("t1")
    values = await waiter
    assert values == {}


@pytest.mark.asyncio
async def test_field_type_sanitization():
    """Reject exotic types; coerce to 'text'."""
    reg = InputRegistry()
    p = reg.register(
        "t1",
        "p",
        [
            {"name": "a", "label": "A", "type": "file"},      # not allowed → text
            {"name": "b", "label": "B", "type": "password"},  # allowed
            {"name": "c", "label": "C", "type": "junk"},      # not allowed → text
        ],
    )
    assert p.fields[0].type == "text"
    assert p.fields[1].type == "password"
    assert p.fields[2].type == "text"


@pytest.mark.asyncio
async def test_public_view_omits_values():
    reg = InputRegistry()
    p = reg.register("t1", "p", [{"name": "x", "label": "X", "type": "text"}])
    p.values = {"x": "PEEKED"}
    pv = p.public_view()
    assert "values" not in pv
    assert pv["prompt"] == "p"
    assert pv["fields"] == [{"name": "x", "label": "X", "type": "text"}]


@pytest.mark.asyncio
async def test_isolation_between_tasks():
    reg = InputRegistry()
    reg.register("t1", "a", [{"name": "x", "label": "x"}])
    reg.register("t2", "b", [{"name": "y", "label": "y"}])

    async def w1():
        return await reg.wait("t1")
    async def w2():
        return await reg.wait("t2")

    waiter1 = asyncio.create_task(w1())
    waiter2 = asyncio.create_task(w2())
    await asyncio.sleep(0)
    reg.provide("t1", {"x": "x1"})
    # t2 must still be pending — providing to t1 doesn't wake t2.
    assert reg.get("t2") is not None
    v1 = await waiter1
    assert v1 == {"x": "x1"}
    reg.provide("t2", {"y": "y2"})
    v2 = await waiter2
    assert v2 == {"y": "y2"}
