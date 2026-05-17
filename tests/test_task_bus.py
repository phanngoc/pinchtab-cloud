"""Tests for TaskBus and TaskRegistry."""
import asyncio

import pytest

from backend.task_bus import END_SENTINEL, TaskBus, TaskRegistry


@pytest.mark.asyncio
async def test_publish_no_subscribers_noop():
    bus = TaskBus()
    delivered = bus.publish("t1", {"type": "step", "n": 1})
    assert delivered == 0


@pytest.mark.asyncio
async def test_single_subscriber_receives_events_in_order():
    bus = TaskBus()
    received = []

    async def consume():
        async with bus.subscribe("t1") as q:
            for _ in range(3):
                received.append(await q.get())

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)  # let subscriber register

    bus.publish("t1", {"type": "step", "n": 1})
    bus.publish("t1", {"type": "step", "n": 2})
    bus.publish("t1", {"type": "step", "n": 3})

    await consumer
    assert [e["n"] for e in received] == [1, 2, 3]


@pytest.mark.asyncio
async def test_two_subscribers_each_get_all_events():
    """Fan-out: every subscriber receives every event."""
    bus = TaskBus()
    a_events: list = []
    b_events: list = []

    async def consume(target):
        async with bus.subscribe("t1") as q:
            for _ in range(2):
                target.append(await q.get())

    ca = asyncio.create_task(consume(a_events))
    cb = asyncio.create_task(consume(b_events))
    await asyncio.sleep(0)

    bus.publish("t1", {"n": 1})
    bus.publish("t1", {"n": 2})

    await asyncio.gather(ca, cb)
    assert [e["n"] for e in a_events] == [1, 2]
    assert [e["n"] for e in b_events] == [1, 2]


@pytest.mark.asyncio
async def test_end_sentinel_delivered():
    bus = TaskBus()
    captured = []

    async def consume():
        async with bus.subscribe("t1") as q:
            while True:
                e = await q.get()
                captured.append(e)
                if e.get("type") == END_SENTINEL:
                    return

    c = asyncio.create_task(consume())
    await asyncio.sleep(0)
    bus.publish("t1", {"type": "step"})
    bus.end("t1")
    await c
    assert captured[-1]["type"] == END_SENTINEL


@pytest.mark.asyncio
async def test_subscriber_isolation_per_task():
    """Events on task A are not visible to subscribers of task B."""
    bus = TaskBus()
    a_events: list = []
    b_events: list = []

    async def consume(task_id, target):
        async with bus.subscribe(task_id) as q:
            try:
                target.append(await asyncio.wait_for(q.get(), timeout=0.1))
            except asyncio.TimeoutError:
                pass

    ca = asyncio.create_task(consume("A", a_events))
    cb = asyncio.create_task(consume("B", b_events))
    await asyncio.sleep(0)
    bus.publish("A", {"n": 1})

    await asyncio.gather(ca, cb)
    assert a_events == [{"n": 1}]
    assert b_events == []


@pytest.mark.asyncio
async def test_slow_consumer_drops_events_rather_than_blocking():
    bus = TaskBus(subscriber_buffer=2)

    async def consume_slow():
        async with bus.subscribe("t1") as q:
            await asyncio.sleep(0.05)  # don't read

    consumer = asyncio.create_task(consume_slow())
    await asyncio.sleep(0)
    bus.publish("t1", {"n": 1})
    bus.publish("t1", {"n": 2})
    # Queue is now full; 3rd publish should drop, NOT block.
    delivered = bus.publish("t1", {"n": 3})
    assert delivered == 0  # dropped
    await consumer


@pytest.mark.asyncio
async def test_subscribe_cleanup_on_exit():
    bus = TaskBus()
    async with bus.subscribe("t1"):
        assert bus.subscriber_count("t1") == 1
    assert bus.subscriber_count("t1") == 0


# ---- TaskRegistry ----


@pytest.mark.asyncio
async def test_registry_basic_register_and_running():
    reg = TaskRegistry()

    async def long_task():
        await asyncio.sleep(0.5)

    t = asyncio.create_task(long_task())
    reg.register("t1", t)
    assert reg.is_running("t1")
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_registry_cancel_running_task():
    reg = TaskRegistry()
    cancelled_flag = []

    async def long_task():
        try:
            await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            cancelled_flag.append(True)
            raise

    t = asyncio.create_task(long_task())
    reg.register("t1", t)
    await asyncio.sleep(0.01)

    result = await reg.cancel("t1")
    assert result is True
    assert cancelled_flag == [True]
    assert not reg.is_running("t1")


@pytest.mark.asyncio
async def test_registry_cancel_unknown_task_returns_false():
    reg = TaskRegistry()
    result = await reg.cancel("nope")
    assert result is False


@pytest.mark.asyncio
async def test_registry_auto_cleans_done_tasks():
    reg = TaskRegistry()

    async def quick():
        return "done"

    t = asyncio.create_task(quick())
    reg.register("t1", t)
    await t
    # done_callback fires; cleanup happens.
    await asyncio.sleep(0)
    assert not reg.is_running("t1")
    assert "t1" not in reg.running_ids()


@pytest.mark.asyncio
async def test_registry_rejects_double_register_of_live_task():
    reg = TaskRegistry()

    async def long():
        await asyncio.sleep(0.5)

    t1 = asyncio.create_task(long())
    reg.register("X", t1)
    t2 = asyncio.create_task(long())
    with pytest.raises(ValueError, match="already registered"):
        reg.register("X", t2)

    t1.cancel()
    t2.cancel()
    for t in (t1, t2):
        try:
            await t
        except asyncio.CancelledError:
            pass
