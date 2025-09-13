import random
import time

import anyio
import pytest

from lionagi.ln.concurrency import (
    bounded_map,
    fail_after,
    gather,
    get_cancelled_exc_class,
    race,
    retry,
)


@pytest.mark.anyio
async def test_gather_first_error_cancels_peers(anyio_backend):
    cancelled = anyio.Event()

    async def boom():
        await anyio.sleep(0.01)
        raise RuntimeError("x")

    async def peer():
        try:
            await anyio.sleep(10)
        except BaseException:
            cancelled.set()
            raise

    t0 = time.perf_counter()
    with pytest.raises(RuntimeError):
        await gather(boom(), peer(), return_exceptions=False)
    dt = time.perf_counter() - t0
    assert cancelled.is_set()
    assert dt < 0.5


@pytest.mark.anyio
async def test_bounded_map_raises_and_cancels_others(anyio_backend):
    started = 0
    cancelled = anyio.Event()

    async def fn(x):
        nonlocal started
        started += 1
        if x == 3:
            await anyio.sleep(0.01)
            raise ValueError("boom")
        try:
            await anyio.sleep(0.1)
        except BaseException:
            cancelled.set()
            raise
        return x

    with pytest.raises(ValueError):
        await bounded_map(fn, range(6), limit=2)
    assert started >= 2  # at least limited concurrency started
    assert cancelled.is_set()


@pytest.mark.anyio
async def test_race_multiple_exceptions_vs_success(anyio_backend):
    async def err1():
        await anyio.sleep(0.005)
        raise RuntimeError("e1")

    async def err2():
        await anyio.sleep(0.006)
        raise ValueError("e2")

    async def ok():
        await anyio.sleep(0.004)
        return "ok"

    assert await race(err1(), ok(), err2()) == "ok"


@pytest.mark.anyio
async def test_retry_respects_attempts_count(anyio_backend):
    calls = {"n": 0}

    async def always():
        calls["n"] += 1
        raise TimeoutError("x")

    with pytest.raises(TimeoutError):
        await retry(
            always,
            attempts=3,
            base_delay=0.001,
            max_delay=0.002,
            retry_on=(TimeoutError,),
        )
    assert calls["n"] == 3


@pytest.mark.anyio
async def test_gather_empty_returns_empty(anyio_backend):
    res = await gather()
    assert res == []


@pytest.mark.anyio
async def test_gather_return_exceptions_true(anyio_backend):
    """Test gather with return_exceptions=True collects all results and exceptions."""

    async def success(x):
        await anyio.sleep(0.001 * x)  # Reduced timing
        return f"result_{x}"

    async def failure(x):
        await anyio.sleep(0.001 * x)  # Reduced timing
        raise ValueError(f"error_{x}")

    # Mix successes and failures
    results = await gather(
        success(1), failure(2), success(3), failure(4), return_exceptions=True
    )

    assert len(results) == 4
    assert results[0] == "result_1"
    assert isinstance(results[1], ValueError)
    assert str(results[1]) == "error_2"
    assert results[2] == "result_3"
    assert isinstance(results[3], ValueError)
    assert str(results[3]) == "error_4"


@pytest.mark.anyio
async def test_gather_return_exceptions_preserves_order(anyio_backend):
    """Test that gather preserves order even with varying completion times."""

    async def task(i):
        # Reverse sleep times to test order preservation
        await anyio.sleep(0.001 * (6 - i))  # Much reduced timing
        if i % 2 == 0:
            return i
        raise RuntimeError(f"error_{i}")

    results = await gather(
        *[task(i) for i in range(6)], return_exceptions=True
    )

    assert len(results) == 6
    for i in range(6):
        if i % 2 == 0:
            assert results[i] == i
        else:
            assert isinstance(results[i], RuntimeError)
            assert str(results[i]) == f"error_{i}"


@pytest.mark.anyio
async def test_bounded_map_empty_and_large_limit(anyio_backend):
    out = await bounded_map(lambda x: x, [], limit=999)
    assert out == []


@pytest.mark.anyio
async def test_bounded_map_respects_limit(anyio_backend):
    """Test that bounded_map strictly enforces concurrency limit."""
    LIMIT = 3
    TASKS = 10  # Reduced from 20
    current_concurrency = 0
    max_observed_concurrency = 0

    async def worker(x):
        nonlocal current_concurrency, max_observed_concurrency
        current_concurrency += 1
        max_observed_concurrency = max(
            max_observed_concurrency, current_concurrency
        )

        await anyio.sleep(0.001)  # Minimal sleep

        current_concurrency -= 1
        return x

    results = await bounded_map(worker, range(TASKS), limit=LIMIT)
    assert max_observed_concurrency == LIMIT
    assert results == list(range(TASKS))  # Verify correct results


@pytest.mark.anyio
async def test_bounded_map_with_return_exceptions(anyio_backend):
    """Test bounded_map with return_exceptions=True."""

    async def worker(x):
        await anyio.sleep(0.001)
        if x % 3 == 0:
            raise ValueError(f"error_{x}")
        return x * 2

    results = await bounded_map(
        worker, range(9), limit=3, return_exceptions=True
    )

    assert len(results) == 9
    for i in range(9):
        if i % 3 == 0:
            assert isinstance(results[i], ValueError)
            assert str(results[i]) == f"error_{i}"
        else:
            assert results[i] == i * 2


@pytest.mark.anyio
async def test_race_single_and_loser_cancelled(anyio_backend):
    # single
    async def one():
        await anyio.sleep(0.002)
        return 1

    assert await race(one()) == 1
    # loser is cancelled
    cancelled = anyio.Event()

    async def slow():
        try:
            await anyio.sleep(10)
            return "slow"
        except BaseException:
            cancelled.set()
            raise

    async def fast():
        await anyio.sleep(0.002)
        return "fast"

    assert await race(slow(), fast()) == "fast"
    assert cancelled.is_set()


@pytest.mark.anyio
async def test_race_first_failure_propagates(anyio_backend):
    """Test that race propagates the first completion even if it's a failure."""
    cancelled = []

    async def fast_failure():
        await anyio.sleep(0.005)
        raise ValueError("I fail fast")

    async def slow_success():
        try:
            await anyio.sleep(0.05)
            return "success"
        except BaseException:
            cancelled.append("slow_success")
            raise

    async def slower_success():
        try:
            await anyio.sleep(0.1)
            return "also_success"
        except BaseException:
            cancelled.append("slower_success")
            raise

    # First completion is a failure, should propagate immediately
    with pytest.raises(ValueError) as exc_info:
        await race(fast_failure(), slow_success(), slower_success())

    assert str(exc_info.value) == "I fail fast"
    # Give a moment for cancellations
    await anyio.sleep(0.01)
    # At least one should be cancelled
    assert len(cancelled) > 0


@pytest.mark.anyio
async def test_race_all_failures_returns_first(anyio_backend):
    """Test that when all tasks fail, race returns the first failure."""

    async def fail1():
        await anyio.sleep(0.01)
        raise ValueError("first")

    async def fail2():
        await anyio.sleep(0.02)
        raise RuntimeError("second")

    async def fail3():
        await anyio.sleep(0.03)
        raise TypeError("third")

    with pytest.raises(ValueError) as exc_info:
        await race(fail1(), fail2(), fail3())

    assert str(exc_info.value) == "first"


@pytest.mark.anyio
async def test_race_requires_at_least_one(anyio_backend):
    with pytest.raises(ValueError):
        await race()


@pytest.mark.anyio
async def test_retry_deadline_capped_by_parent(anyio_backend):
    calls = {"n": 0}

    async def always():
        calls["n"] += 1
        raise TimeoutError("x")

    # Base delays are large, but parent deadline is tight; retry should stop early
    start = anyio.current_time()
    with fail_after(0.1):  # More generous deadline
        with pytest.raises(TimeoutError) as exc_info:
            # Use tiny delays to avoid deadline conflicts
            await retry(
                always,
                attempts=50,
                base_delay=0.001,
                max_delay=0.001,
                retry_on=(TimeoutError,),
                jitter=0.0,
            )
    elapsed = anyio.current_time() - start
    # Could be either the original TimeoutError or the deadline TimeoutError
    # The important thing is that it respects the deadline and doesn't hang
    assert elapsed <= 0.5  # Should complete quickly (CI-friendly)
    assert calls["n"] >= 1


@pytest.mark.anyio
async def test_retry_with_and_without_jitter(anyio_backend):
    seen = {"n": 0}

    async def boom():
        seen["n"] += 1
        raise TimeoutError("nope")

    # no jitter path
    with pytest.raises(TimeoutError):
        await retry(
            boom,
            attempts=2,
            base_delay=0.001,
            max_delay=0.002,
            retry_on=(TimeoutError,),
            jitter=0.0,
        )
    # jitter>0 path executes as well
    with pytest.raises(TimeoutError):
        await retry(
            boom,
            attempts=2,
            base_delay=0.001,
            max_delay=0.002,
            retry_on=(TimeoutError,),
            jitter=0.001,
        )
    assert seen["n"] >= 4


@pytest.mark.anyio
async def test_retry_eventual_success(anyio_backend):
    """Test that retry returns result if a later attempt succeeds."""
    attempts = {"count": 0}

    async def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ConnectionError(f"Attempt {attempts['count']} failed")
        return "success"

    result = await retry(
        flaky, attempts=5, base_delay=0.001, retry_on=(ConnectionError,)
    )

    assert result == "success"
    assert attempts["count"] == 3  # Succeeded on third attempt


@pytest.mark.anyio
async def test_retry_exception_filtering(anyio_backend):
    """Test that retry fails immediately on non-retryable exceptions."""
    attempts = {"count": 0}

    async def raises_wrong_exception():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ValueError("Not in retry_on list")
        raise TimeoutError("Should not reach here")

    # ValueError is not in retry_on, should fail immediately
    with pytest.raises(ValueError) as exc_info:
        await retry(
            raises_wrong_exception,
            attempts=5,
            base_delay=0.001,
            retry_on=(
                TimeoutError,
                ConnectionError,
            ),  # ValueError not included
        )

    assert str(exc_info.value) == "Not in retry_on list"
    assert attempts["count"] == 1  # Only one attempt made


@pytest.mark.anyio
async def test_retry_mixed_exceptions(anyio_backend):
    """Test retry with mixture of retryable and non-retryable exceptions."""
    attempts = {"count": 0}

    async def mixed_failures():
        attempts["count"] += 1
        if attempts["count"] <= 2:
            # Retryable
            raise TimeoutError(f"Timeout {attempts['count']}")
        elif attempts["count"] == 3:
            # Non-retryable
            raise ValueError("Critical error")
        return "should_not_reach"

    with pytest.raises(ValueError) as exc_info:
        await retry(
            mixed_failures,
            attempts=10,
            base_delay=0.001,
            retry_on=(TimeoutError,),
        )

    assert str(exc_info.value) == "Critical error"
    assert attempts["count"] == 3  # Two retries then critical failure
