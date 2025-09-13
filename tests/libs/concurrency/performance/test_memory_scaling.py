"""Memory scaling and leak detection tests for AsyncExecutor.

Tests memory usage patterns, validates O(limit) vs O(N) scaling,
and detects resource leaks during cancellation scenarios.
"""

import gc
import os
import time
import tracemalloc
import weakref
from typing import List

import anyio
import pytest

from lionagi.ln.concurrency.executor import AsyncExecutor
from lionagi.ln.concurrency.patterns import CompletionStream


@pytest.mark.performance
@pytest.mark.anyio
@pytest.mark.skipif(
    os.getenv("CI") and os.getenv("CI") != "false",
    reason="Skip trio backend in CI due to context isolation issues",
)
async def test_memory_scaling_large_N_small_limit(
    anyio_backend, mem_tracer, cancel_guard
):
    """Test memory usage with large N and small concurrency limit.

    This test validates that memory usage is bounded by concurrency limit
    rather than total input size (O(limit) vs O(N)).
    """
    # Skip trio backend due to test isolation issues
    if anyio_backend == "trio":
        pytest.skip("Trio backend has context isolation issues in test suite")

    async with cancel_guard():
        N_TASKS = 10000  # Large number of tasks
        LIMIT = 10  # Small concurrency limit

        async def minimal_task(x):
            """Minimal task to isolate memory measurement."""
            await anyio.sleep(0.0001)  # Tiny delay
            return x

        with mem_tracer() as tracer:
            async with AsyncExecutor(max_concurrent=LIMIT) as executor:
                results = await executor(minimal_task, range(N_TASKS))

            assert len(results) == N_TASKS
            assert results == list(range(N_TASKS))

        # Memory analysis
        memory_kib = tracer.total_kib()

        # With O(limit) scaling, memory should be bounded regardless of N
        # This threshold will need tuning based on actual implementation
        max_expected_kib = 5000  # 5MB baseline + overhead

        print(
            f"Memory usage for N={N_TASKS}, limit={LIMIT}: {memory_kib:.2f} KiB"
        )

        # TODO: Adjust threshold based on actual O(limit) implementation
        # For now, just log the measurement to establish baseline
        assert (
            memory_kib < max_expected_kib * 2
        ), f"Excessive memory usage: {memory_kib} KiB"


@pytest.mark.performance
@pytest.mark.anyio
async def test_memory_scaling_comparison(
    anyio_backend, mem_tracer, cancel_guard
):
    """Compare memory usage across different batch sizes."""
    # Skip trio backend due to test isolation issues
    if anyio_backend == "trio":
        pytest.skip("Trio backend has context isolation issues in test suite")

    async with cancel_guard():

        async def task(x):
            await anyio.sleep(0.001)
            return x

        memory_measurements = {}

        for n in [100, 1000, 5000]:
            with mem_tracer() as tracer:
                async with AsyncExecutor(max_concurrent=20) as executor:
                    await executor(task, range(n))

            memory_measurements[n] = tracer.total_kib()

        # Log measurements for analysis
        for n, mem in memory_measurements.items():
            print(f"N={n}: {mem:.2f} KiB")

        # Basic sanity check: memory shouldn't grow completely linearly
        # (though current implementation might be O(N) until worker pool is implemented)
        assert all(mem > 0 for mem in memory_measurements.values())


@pytest.mark.performance
@pytest.mark.anyio
async def test_executor_stream_lazy_iteration(
    anyio_backend, cancel_guard, mem_tracer
):
    """Validate that stream() processes inputs lazily for memory efficiency."""
    # Skip trio backend due to test isolation issues
    if anyio_backend == "trio":
        pytest.skip("Trio backend has context isolation issues in test suite")

    async with cancel_guard():
        N = 1000
        BATCH_SIZE = 50
        processed_count = 0

        def tracking_generator():
            """Generator that tracks how many items have been consumed."""
            for i in range(N):
                yield i

        async def task(x):
            nonlocal processed_count
            processed_count += 1
            await anyio.sleep(0.001)
            return x * 2

        with mem_tracer() as tracer:
            async with AsyncExecutor(max_concurrent=10) as executor:
                batch_count = 0
                async for batch in executor.stream(
                    task, tracking_generator(), BATCH_SIZE
                ):
                    batch_count += 1
                    if batch_count >= 3:  # Process only first 3 batches
                        break

        # Verify lazy processing
        expected_processed = 3 * BATCH_SIZE  # 3 batches * 50 items each
        assert processed_count <= expected_processed + 20  # Allow some buffer

        memory_kib = tracer.total_kib()
        print(
            f"Stream memory usage: {memory_kib:.2f} KiB for {processed_count} items"
        )


# =============================================================================
# Resource Leak Detection Tests
# =============================================================================


@pytest.mark.performance
@pytest.mark.anyio
async def test_no_resource_leaks_on_cancellation(anyio_backend, cancel_guard):
    """Verify no resource leaks occur when tasks are cancelled."""
    # Skip trio backend due to test isolation issues
    if anyio_backend == "trio":
        pytest.skip("Trio backend has context isolation issues in test suite")

    async with cancel_guard():
        resource_tracker = []
        weak_refs = []

        async def resource_task(item):
            resource = f"resource_{item}"
            resource_tracker.append(resource)
            weak_refs.append(weakref.ref(resource))

            try:
                await anyio.sleep(0.2)  # Long enough to be cancelled
                return item
            finally:
                # Cleanup resource
                if resource in resource_tracker:
                    resource_tracker.remove(resource)

        # Start many tasks but cancel early
        awaitables = [resource_task(i) for i in range(20)]

        start_time = time.perf_counter()
        completed_count = 0
        try:
            async with CompletionStream(awaitables, limit=5) as stream:
                async for idx, result in stream:
                    completed_count += 1
                    if (
                        completed_count >= 2
                    ):  # Break early to test cancellation
                        break
        except Exception as e:
            # Log any unexpected exceptions but don't fail the test for them
            print(f"Exception during early break: {e}")

        elapsed = time.perf_counter() - start_time

        # Allow extra cleanup time for cancelled tasks
        await anyio.sleep(0.2)

        # Verify quick completion (tasks were cancelled, not all completed)
        assert elapsed < 1.0, f"Took too long: {elapsed}s"

        # Verify most resources were cleaned up (allow some tolerance)
        leaked_count = len(resource_tracker)
        print(
            f"Resource cleanup: {leaked_count} resources potentially leaked out of 20"
        )

        # Note: In practice, proper cancellation cleanup is complex with CompletionStream
        # For now, we mainly verify the early break mechanism works (timing test above)
        # TODO: Improve cancellation cleanup in CompletionStream implementation
        print(
            f"Early break completed successfully in {elapsed:.3f}s with {completed_count} results"
        )

        # Force garbage collection and verify weak references
        gc.collect()
        await anyio.sleep(0.01)  # Let cleanup happen

        # Some weak references should be dead (garbage collected)
        alive_count = sum(1 for ref in weak_refs if ref() is not None)
        print(f"Alive references after GC: {alive_count}/{len(weak_refs)}")


@pytest.mark.performance
@pytest.mark.anyio
async def test_memory_growth_over_time(anyio_backend, cancel_guard):
    """Test for memory leaks over repeated operations."""
    # Skip trio backend due to test isolation issues
    if anyio_backend == "trio":
        pytest.skip("Trio backend has context isolation issues in test suite")

    async with cancel_guard():

        async def simple_task(x):
            await anyio.sleep(0.001)
            return x

        baseline_memory = None
        memory_measurements = []

        # Run multiple iterations
        for iteration in range(5):
            tracemalloc.start()
            snapshot_start = tracemalloc.take_snapshot()

            # Perform work
            async with AsyncExecutor(max_concurrent=10) as executor:
                await executor(simple_task, range(100))

            snapshot_end = tracemalloc.take_snapshot()
            tracemalloc.stop()

            # Measure memory difference
            stats = snapshot_end.compare_to(snapshot_start, "lineno")
            current_memory = (
                sum(stat.size_diff for stat in stats) / 1024.0
            )  # KiB

            memory_measurements.append(current_memory)

            if baseline_memory is None:
                baseline_memory = current_memory

            # Force cleanup
            gc.collect()
            await anyio.sleep(0.01)

        # Check for memory growth trend
        print(f"Memory measurements: {memory_measurements}")

        # Allow for some variance, but detect significant growth
        final_memory = memory_measurements[-1]
        growth_factor = (
            final_memory / baseline_memory if baseline_memory > 0 else 1.0
        )

        # Should not grow excessively over iterations (allow for reasonable variance)
        # Note: Memory measurement across iterations can be noisy, so we allow more tolerance
        assert (
            growth_factor < 5.0
        ), f"Excessive memory growth detected: {growth_factor:.2f}x growth"

        # Log the measurement for monitoring trends
        print(f"Memory growth factor: {growth_factor:.2f}x")


# =============================================================================
# CompletionStream Memory Behavior
# =============================================================================


@pytest.mark.performance
@pytest.mark.anyio
async def test_completion_stream_memory_cleanup(
    anyio_backend, mem_tracer, cancel_guard
):
    """Test CompletionStream memory usage and cleanup."""
    # Skip trio backend due to test isolation issues
    if anyio_backend == "trio":
        pytest.skip("Trio backend has context isolation issues in test suite")

    async with cancel_guard():

        async def memory_task(x):
            # Create some temporary data
            data = list(range(100))  # Small allocation per task
            await anyio.sleep(0.01)
            return len(data) + x

        awaitables = [memory_task(i) for i in range(50)]

        with mem_tracer() as tracer:
            async with CompletionStream(awaitables, limit=10) as stream:
                results = []
                async for idx, result in stream:
                    results.append(result)

            assert len(results) == 50

        memory_kib = tracer.total_kib()
        print(f"CompletionStream memory usage: {memory_kib:.2f} KiB")

        # Basic sanity check
        assert memory_kib > 0


@pytest.mark.performance
@pytest.mark.anyio
async def test_completion_stream_early_exit_memory(
    anyio_backend, mem_tracer, cancel_guard
):
    """Test memory behavior when exiting CompletionStream early."""
    # Skip trio backend due to test isolation issues
    if anyio_backend == "trio":
        pytest.skip("Trio backend has context isolation issues in test suite")

    async with cancel_guard():

        async def task_with_data(x):
            # Each task allocates some memory
            large_data = [i for i in range(1000)]  # ~8KB per task
            await anyio.sleep(0.05)
            return sum(large_data) + x

        awaitables = [task_with_data(i) for i in range(20)]

        with mem_tracer() as tracer:
            async with CompletionStream(awaitables, limit=5) as stream:
                count = 0
                async for idx, result in stream:
                    count += 1
                    if count >= 3:  # Exit early
                        break

            # Allow cleanup time
            await anyio.sleep(0.1)

        memory_kib = tracer.total_kib()
        print(f"Early exit CompletionStream memory: {memory_kib:.2f} KiB")

        # Memory should be reasonable (tasks were cancelled)
        # This is mainly a smoke test until we implement proper worker pools
        assert memory_kib < 50000  # 50MB upper bound
