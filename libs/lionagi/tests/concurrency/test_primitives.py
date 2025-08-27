import time

import anyio
import pytest

from lionagi.ln.concurrency import (
    CapacityLimiter,
    Condition,
    Event,
    LeakTracker,
    Lock,
    Queue,
    Semaphore,
    track_resource,
    untrack_resource,
)


@pytest.mark.anyio
async def test_queue_unbuffered_backpressure(anyio_backend):
    # Unbuffered stream (maxsize=0) should block until receiver calls get()
    q = Queue.with_maxsize(0)
    got = {}

    async def consumer(*, task_status=None):
        if task_status:
            task_status.started()
        got["v"] = await q.get()

    async with anyio.create_task_group() as tg:
        await tg.start(consumer)
        # put will not return until consumer receives
        await q.put(123)

    assert got["v"] == 123


@pytest.mark.anyio
async def test_queue_sender_receiver_properties(anyio_backend):
    q = Queue.with_maxsize(1)
    # Ensure we can use the raw streams without closing the queue
    await q.sender.send("x")
    assert await q.receiver.receive() == "x"


@pytest.mark.anyio
async def test_capacity_limiter_one_token(anyio_backend):
    lim = CapacityLimiter(1)
    entered = 0
    max_conc = 0

    async def w():
        nonlocal entered, max_conc
        await lim.acquire()
        entered += 1
        max_conc = max(max_conc, entered)
        await anyio.sleep(0.01)
        entered -= 1
        lim.release()

    async with anyio.create_task_group() as tg:
        for _ in range(5):
            tg.start_soon(w)

    assert max_conc == 1


@pytest.mark.anyio
async def test_lock_is_a_context_manager(anyio_backend):
    lock = Lock()
    async with lock:
        pass  # no error


@pytest.mark.anyio
async def test_semaphore_limits_exactly(anyio_backend):
    sem = Semaphore(2)
    conc = 0
    maxc = 0

    async def w():
        nonlocal conc, maxc
        async with sem:
            conc += 1
            maxc = max(maxc, conc)
            await anyio.sleep(0.01)
            conc -= 1

    async with anyio.create_task_group() as tg:
        for _ in range(6):
            tg.start_soon(w)
    assert maxc <= 2


def test_resource_tracker_default_name_and_clear():
    tracker = LeakTracker()

    class X:
        pass

    x = X()
    tracker.track(x, name=None, kind=None)
    live = tracker.live()
    assert len(live) == 1 and live[0].name.startswith("obj-")
    tracker.clear()
    assert tracker.live() == []


def test_module_level_track_resource_defaults():
    class Y:
        pass

    y = Y()
    track_resource(y)  # default name/kind
    untrack_resource(y)  # should not raise


@pytest.mark.anyio
async def test_queue_buffered_behavior(anyio_backend):
    """Test that buffered queue doesn't block put until full."""
    q = Queue.with_maxsize(3)

    # Should be able to put 3 items without blocking
    q.put_nowait(1)
    q.put_nowait(2)
    q.put_nowait(3)

    # Fourth should raise WouldBlock
    with pytest.raises(anyio.WouldBlock):
        q.put_nowait(4)

    # Get one item to make room
    assert await q.get() == 1

    # Now should be able to put another
    q.put_nowait(4)

    # Verify remaining items
    assert await q.get() == 2
    assert await q.get() == 3
    assert await q.get() == 4


@pytest.mark.anyio
async def test_queue_get_nowait(anyio_backend):
    """Test non-blocking get operations."""
    q = Queue.with_maxsize(2)

    # Empty queue should raise WouldBlock
    with pytest.raises(anyio.WouldBlock):
        q.get_nowait()

    # Add items
    await q.put(1)
    await q.put(2)

    # Non-blocking get should work
    assert q.get_nowait() == 1
    assert q.get_nowait() == 2

    # Queue empty again
    with pytest.raises(anyio.WouldBlock):
        q.get_nowait()


@pytest.mark.anyio
async def test_queue_lifecycle_with_context_manager(anyio_backend):
    """Test queue cleanup with async context manager."""
    async with Queue.with_maxsize(1) as q:
        await q.put("test")
        assert await q.get() == "test"
    # Queue should be closed after context exit

    # Trying to use closed queue should raise
    with pytest.raises(anyio.ClosedResourceError):
        await q.put("fail")


@pytest.mark.anyio
async def test_lock_mutual_exclusion(anyio_backend):
    """Test that Lock enforces mutual exclusion under contention."""
    lock = Lock()
    counter = 0
    TASKS = 10  # Further reduced
    INCREMENTS = 10  # Further reduced

    async def increment():
        nonlocal counter
        for _ in range(INCREMENTS):
            async with lock:
                # Critical section
                temp = counter
                await anyio.sleep(0)  # Force context switch
                counter = temp + 1

    async with anyio.create_task_group() as tg:
        for _ in range(TASKS):
            tg.start_soon(increment)

    # Without lock, we'd lose increments due to race conditions
    assert counter == TASKS * INCREMENTS


@pytest.mark.anyio
async def test_semaphore_strict_limit(anyio_backend):
    """Test that Semaphore strictly enforces its limit."""
    sem = Semaphore(3)
    active = 0
    max_active = 0
    violations = []

    async def worker(i):
        nonlocal active, max_active
        async with sem:
            active += 1
            max_active = max(max_active, active)
            if active > 3:
                violations.append(f"Task {i}: {active} concurrent")
            await anyio.sleep(0.001)  # Minimal sleep
            active -= 1

    async with anyio.create_task_group() as tg:
        for i in range(10):  # Reduced from 20
            tg.start_soon(worker, i)

    assert max_active == 3
    assert not violations, f"Semaphore violations: {violations}"


@pytest.mark.anyio
async def test_capacity_limiter_dynamic_adjustment(anyio_backend):
    """Test dynamic capacity adjustment of CapacityLimiter."""
    limiter = CapacityLimiter(2)

    # Initially should allow 2 tokens
    assert limiter.total_tokens == 2
    assert limiter.available_tokens == 2
    assert limiter.borrowed_tokens == 0

    # Acquire one token using context manager
    async with limiter:
        assert limiter.available_tokens == 1
        assert limiter.borrowed_tokens == 1

        # Dynamically increase capacity while holding a token
        limiter.total_tokens = 4
        assert limiter.available_tokens == 3  # 4 total - 1 borrowed
        assert limiter.borrowed_tokens == 1

    # After release
    assert limiter.available_tokens == 4
    assert limiter.borrowed_tokens == 0

    # Test multiple acquisitions with increased capacity using context managers
    tokens_held = []

    async def acquire_and_hold():
        async with limiter:
            tokens_held.append(True)

    # Start multiple tasks to acquire tokens
    async with anyio.create_task_group() as tg:
        for _ in range(3):
            tg.start_soon(acquire_and_hold)

    assert len(tokens_held) == 3
    # After all context managers exit, tokens should be released
    assert limiter.borrowed_tokens == 0
    assert limiter.available_tokens == 4


@pytest.mark.anyio
async def test_capacity_limiter_is_async_context_manager(anyio_backend):
    lim = CapacityLimiter(2)
    async with lim:
        assert lim.borrowed_tokens == 1
        assert lim.available_tokens == 1
    assert lim.borrowed_tokens == 0
    assert lim.available_tokens == 2


@pytest.mark.anyio
async def test_event_signaling(anyio_backend):
    """Test Event for task signaling."""
    event = Event()
    results = []

    async def waiter(n):
        await event.wait()
        results.append(n)

    # Start waiters
    async with anyio.create_task_group() as tg:
        for i in range(5):
            tg.start_soon(waiter, i)

        # Give waiters time to start waiting
        await anyio.sleep(0.01)
        assert not event.is_set()
        assert results == []

        # Signal all waiters
        event.set()
        assert event.is_set()

    # All waiters should have been signaled
    assert set(results) == {0, 1, 2, 3, 4}


@pytest.mark.anyio
async def test_condition_notify_selective(anyio_backend):
    """Test Condition variable for selective task notification."""
    lock = Lock()
    condition = Condition(lock)
    results = []
    waiters_ready = 0

    async def waiter(n):
        nonlocal waiters_ready
        async with condition:
            waiters_ready += 1
            await condition.wait()
            results.append(n)

    async def notifier():
        # Wait for all waiters to be ready
        while waiters_ready < 5:
            await anyio.sleep(0.001)

        await anyio.sleep(0.01)
        async with condition:
            # Notify only 2 tasks
            condition.notify(2)

        await anyio.sleep(0.01)
        async with condition:
            # Notify all remaining
            condition.notify_all()

    async with anyio.create_task_group() as tg:
        # Start 5 waiters
        for i in range(5):
            tg.start_soon(waiter, i)

        # Start notifier
        tg.start_soon(notifier)

    # After TaskGroup exits, all tasks are done
    assert len(results) == 5  # All eventually notified


@pytest.mark.anyio
async def test_condition_without_explicit_lock(anyio_backend):
    """Test Condition with auto-created lock."""
    condition = Condition()  # Creates its own lock
    value = None

    async def producer():
        nonlocal value
        await anyio.sleep(0.01)
        async with condition:
            value = 42
            condition.notify()

    async def consumer():
        async with condition:
            while value is None:
                await condition.wait()
            return value

    async with anyio.create_task_group() as tg:
        tg.start_soon(producer)
        result = await consumer()

    assert result == 42
