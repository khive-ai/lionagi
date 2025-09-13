"""Critical regression tests for the 4 fixed bugs in AsyncExecutor.

These tests specifically prevent the return of:
1. CompletionStream cancellation bug
2. anyio.to_thread.run_sync kwargs bug
3. Async callable detection bug
4. AsyncExecutor.stream() method bug
"""

import asyncio
import functools
import time
from typing import Any

import anyio
import pytest

from lionagi.ln.concurrency.executor import AsyncExecutor
from lionagi.ln.concurrency.patterns import CompletionStream
from lionagi.ln.concurrency.utils import is_coro_func

# =============================================================================
# Bug 1: CompletionStream Early Cancellation Regression Test
# =============================================================================


@pytest.mark.anyio
@pytest.mark.regression
async def test_completion_stream_early_break_cancels_remaining_tasks(
    anyio_backend, cancel_guard
):
    """REGRESSION TEST: Ensure early break from CompletionStream cancels remaining tasks.

    Previously fixed bug: Tasks continued running after early break from async iteration.
    """
    async with cancel_guard():
        cancelled_tasks = 0
        finished_tasks = 0
        CancelledError = anyio.get_cancelled_exc_class()

        async def long_running_task(task_id):
            nonlocal cancelled_tasks, finished_tasks
            try:
                await anyio.sleep(0.5)  # Long enough to be cancelled
                finished_tasks += 1
                return f"completed_{task_id}"
            except CancelledError:
                cancelled_tasks += 1
                raise

        # Create 10 long-running tasks
        awaitables = [long_running_task(i) for i in range(10)]

        start_time = time.perf_counter()

        async with CompletionStream(awaitables, limit=3) as stream:
            async for idx, result in stream:
                if idx >= 1:  # Break after getting 2 results
                    break

        elapsed = time.perf_counter() - start_time

        # Wait briefly for cancellation to propagate
        await anyio.sleep(0.1)

        # CRITICAL ASSERTIONS:
        # 1. Should complete quickly (tasks were cancelled, not waited for)
        assert (
            elapsed < 1.0
        ), f"Took too long: {elapsed}s - tasks not cancelled"

        # 2. Should have some finished tasks (at least the ones that completed)
        assert finished_tasks >= 1, "No tasks finished - test setup issue"

        # 3. Should have cancelled tasks (the remaining ones)
        assert cancelled_tasks > 0, "No tasks were cancelled - BUG REGRESSION!"

        # 4. Total should make sense
        assert finished_tasks + cancelled_tasks <= 10


# =============================================================================
# Bug 2: anyio.to_thread.run_sync kwargs Regression Test
# =============================================================================


def sync_function_with_complex_kwargs(a, b=0, *, c=0, d="default"):
    """Sync function that requires complex kwargs to test the fix."""
    return f"{a}-{b}-{c}-{d}"


@pytest.mark.anyio
@pytest.mark.regression
async def test_executor_sync_function_kwargs_fix(anyio_backend, cancel_guard):
    """REGRESSION TEST: Ensure sync functions with kwargs work via functools.partial.

    Previously fixed bug: kwargs were incorrectly passed to anyio.to_thread.run_sync,
    causing crashes or incorrect parameter interpretation.
    """
    async with cancel_guard():
        async with AsyncExecutor(max_concurrent=2) as executor:
            # Test single function with kwargs
            result = await executor(
                lambda _: sync_function_with_complex_kwargs(
                    1, b=2, c=3, d="test"
                ),
                [0],
            )
            assert result == ["1-2-3-test"]

            # Test multiple functions with different kwargs
            funcs = [
                sync_function_with_complex_kwargs,
                sync_function_with_complex_kwargs,
            ]
            args_kwargs = [
                ((10,), {"b": 20, "c": 30, "d": "first"}),
                ((40,), {"b": 50, "c": 60, "d": "second"}),
            ]

            results = await executor(funcs, args_kwargs)
            expected = ["10-20-30-first", "40-50-60-second"]
            assert results == expected


@pytest.mark.anyio
@pytest.mark.regression
async def test_executor_sync_function_exception_propagation(
    anyio_backend, cancel_guard
):
    """REGRESSION TEST: Ensure sync function exceptions propagate correctly through threads."""
    async with cancel_guard():

        def sync_function_that_fails(x):
            if x == 5:
                raise ValueError(f"Sync function failed on {x}")
            return x * 2

        async with AsyncExecutor(max_concurrent=3) as executor:
            with pytest.raises(ValueError, match="Sync function failed on 5"):
                await executor(sync_function_that_fails, [1, 2, 5, 6])


# =============================================================================
# Bug 3: Async Callable Detection Regression Test
# =============================================================================


class AsyncCallableObject:
    """Test class with async __call__ method."""

    async def __call__(self, x):
        await anyio.sleep(0.001)
        return x * 4


class SyncCallableObject:
    """Test class with sync __call__ method."""

    def __call__(self, x):
        return x * 5


async def regular_async_function(x):
    """Regular async function for comparison."""
    await anyio.sleep(0.001)
    return x * 2


def regular_sync_function(x):
    """Regular sync function for comparison."""
    return x * 3


@pytest.mark.regression
def test_is_coro_func_async_callable_detection():
    """REGRESSION TEST: Ensure is_coro_func detects async __call__ methods.

    Previously fixed bug: asyncio.iscoroutinefunction missed objects with async __call__.
    """
    # Test regular functions
    assert is_coro_func(regular_async_function) is True
    assert is_coro_func(regular_sync_function) is False

    # Test callable objects (the critical fix)
    assert (
        is_coro_func(AsyncCallableObject()) is True
    ), "REGRESSION: Failed to detect async __call__"
    assert is_coro_func(SyncCallableObject()) is False

    # Test lambda functions
    assert is_coro_func(lambda x: x) is False

    # Test None and other edge cases
    assert is_coro_func(None) is False


@pytest.mark.anyio
@pytest.mark.regression
async def test_executor_handles_async_callable_objects(
    anyio_backend, cancel_guard
):
    """REGRESSION TEST: Ensure AsyncExecutor correctly executes async callable objects."""
    async with cancel_guard():
        async_callable = AsyncCallableObject()
        sync_callable = SyncCallableObject()

        async with AsyncExecutor(max_concurrent=2) as executor:
            # Test async callable object
            async_results = await executor(async_callable, [1, 2, 3])
            assert async_results == [4, 8, 12]  # x * 4

            # Test sync callable object (should go through thread pool)
            sync_results = await executor(sync_callable, [1, 2, 3])
            assert sync_results == [5, 10, 15]  # x * 5


# =============================================================================
# Bug 4: AsyncExecutor.stream() Method Bug Regression Test
# =============================================================================


@pytest.mark.anyio
@pytest.mark.regression
async def test_executor_stream_method_fix(anyio_backend, cancel_guard):
    """REGRESSION TEST: Ensure AsyncExecutor.stream() calls correct internal method.

    Previously fixed bug: stream() called non-existent self.execute() method instead
    of self._execute_single().
    """
    async with cancel_guard():

        async def test_task(x):
            await anyio.sleep(0.001)
            return x * x

        async with AsyncExecutor(max_concurrent=3) as executor:
            # This would have crashed before the fix
            batches = []
            async for batch in executor.stream(
                test_task, [1, 2, 3, 4, 5], batch_size=2
            ):
                batches.append(batch)

            # Verify correct batching and results
            expected_batches = [
                [1, 4],
                [9, 16],
                [25],
            ]  # [1²,2²], [3²,4²], [5²]
            assert batches == expected_batches


@pytest.mark.anyio
@pytest.mark.regression
async def test_executor_stream_with_sync_functions(
    anyio_backend, cancel_guard
):
    """REGRESSION TEST: Ensure stream() works with sync functions too."""
    async with cancel_guard():

        def sync_square(x):
            return x * x

        async with AsyncExecutor(max_concurrent=2) as executor:
            results = []
            async for batch in executor.stream(
                sync_square, range(6), batch_size=3
            ):
                results.extend(batch)

            assert results == [0, 1, 4, 9, 16, 25]  # squares of 0-5


# =============================================================================
# Combined Integration Regression Test
# =============================================================================


@pytest.mark.anyio
@pytest.mark.regression
async def test_all_fixes_integration(anyio_backend, cancel_guard):
    """REGRESSION TEST: Integration test covering all 4 fixed bugs together."""
    async with cancel_guard():
        # Mix of async callable, sync function with kwargs, and regular functions
        async_callable = AsyncCallableObject()

        def sync_with_kwargs(x, multiplier=10):
            return x * multiplier

        async def async_func(x):
            await anyio.sleep(0.001)
            return x + 100

        async with AsyncExecutor(
            max_concurrent=4, retry_attempts=1
        ) as executor:
            # Test 1: Multiple function types
            funcs = [async_callable, sync_with_kwargs, async_func]
            args_kwargs = [
                ((2,), None),  # async callable: 2 * 4 = 8
                ((3,), {"multiplier": 5}),  # sync with kwargs: 3 * 5 = 15
                ((4,), None),  # async func: 4 + 100 = 104
            ]

            results = await executor(funcs, args_kwargs)
            assert results == [8, 15, 104]

            # Test 2: Stream with mixed types
            stream_results = []
            async for batch in executor.stream(
                async_callable, [1, 2], batch_size=1
            ):
                stream_results.extend(batch)

            assert stream_results == [4, 8]  # [1*4, 2*4]

            # Test 3: Early termination test (smaller scale)
            async def slow_task(x):
                await anyio.sleep(0.2)
                return x

            # This tests that we can break early without resource leaks
            # (though this is more of a smoke test than the detailed CompletionStream test above)
            start = time.perf_counter()
            try:
                with anyio.move_on_after(0.05):  # Short timeout
                    await executor(slow_task, [1, 2, 3])
            except anyio.get_cancelled_exc_class():
                pass

            elapsed = time.perf_counter() - start
            assert elapsed < 0.1  # Should timeout quickly, not wait for tasks
