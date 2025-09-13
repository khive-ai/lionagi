"""Performance benchmarks for AsyncExecutor using pytest-benchmark.

Provides throughput and latency benchmarks for different workload types
to establish performance baselines and catch regressions.
"""

import asyncio
import time
from functools import partial

import anyio
import pytest

from lionagi.ln.concurrency.executor import AsyncExecutor


def run_async_benchmark(coro_factory, backend="asyncio"):
    """Helper to run async coroutines in pytest-benchmark."""

    def _runner():
        return anyio.run(coro_factory, backend=backend)

    return _runner


# =============================================================================
# I/O Bound Throughput Benchmarks
# =============================================================================


@pytest.mark.performance
@pytest.mark.benchmark(group="io-throughput")
def test_bench_io_throughput_small_batch(benchmark):
    """Benchmark I/O throughput with small batch (100 tasks)."""
    N = 100
    LIMIT = 20

    async def io_task(item):
        await anyio.sleep(0.001)  # 1ms simulated I/O
        return item * 2

    async def workload():
        async with AsyncExecutor(max_concurrent=LIMIT) as executor:
            return await executor(io_task, range(N))

    result = benchmark(run_async_benchmark(workload))
    assert len(result) == N


@pytest.mark.performance
@pytest.mark.benchmark(group="io-throughput")
def test_bench_io_throughput_large_batch(benchmark):
    """Benchmark I/O throughput with large batch (1000 tasks)."""
    N = 1000
    LIMIT = 50

    async def io_task(item):
        await anyio.sleep(0.001)  # 1ms simulated I/O
        return item * 2

    async def workload():
        async with AsyncExecutor(max_concurrent=LIMIT) as executor:
            return await executor(io_task, range(N))

    result = benchmark(run_async_benchmark(workload))
    assert len(result) == N


@pytest.mark.performance
@pytest.mark.benchmark(group="io-throughput")
def test_bench_io_throughput_high_concurrency(benchmark):
    """Benchmark I/O throughput with high concurrency limit."""
    N = 500
    LIMIT = 100

    async def io_task(item):
        await anyio.sleep(0.002)  # 2ms simulated I/O
        return item * 2

    async def workload():
        async with AsyncExecutor(max_concurrent=LIMIT) as executor:
            return await executor(io_task, range(N))

    result = benchmark(run_async_benchmark(workload))
    assert len(result) == N


# =============================================================================
# CPU Bound (Thread Pool) Benchmarks
# =============================================================================


def cpu_intensive_task(n, iterations=1000):
    """CPU-intensive synchronous task."""
    result = 0
    for i in range(iterations):
        result += (n * i) % 7
    return result


@pytest.mark.performance
@pytest.mark.benchmark(group="cpu-throughput")
def test_bench_cpu_throughput_threadpool(benchmark):
    """Benchmark CPU-bound tasks in thread pool."""
    N = 100
    LIMIT = 4  # Reasonable for CPU-bound

    async def workload():
        async with AsyncExecutor(max_concurrent=LIMIT) as executor:
            return await executor(
                lambda x: cpu_intensive_task(
                    x, 500
                ),  # Lighter CPU work for benchmarking
                range(N),
            )

    result = benchmark(run_async_benchmark(workload))
    assert len(result) == N


@pytest.mark.performance
@pytest.mark.benchmark(group="cpu-throughput")
def test_bench_mixed_sync_async_workload(benchmark):
    """Benchmark mixed sync/async workload."""
    N = 50

    def sync_task(x):
        return cpu_intensive_task(x, 200)

    async def async_task(x):
        await anyio.sleep(0.005)
        return x * 10

    async def workload():
        async with AsyncExecutor(max_concurrent=8) as executor:
            # Interleave sync and async tasks
            funcs = [sync_task if i % 2 == 0 else async_task for i in range(N)]
            args_kwargs = [((i,), None) for i in range(N)]
            return await executor(funcs, args_kwargs)

    result = benchmark(run_async_benchmark(workload))
    assert len(result) == N


# =============================================================================
# Streaming Benchmarks
# =============================================================================


@pytest.mark.performance
@pytest.mark.benchmark(group="streaming")
def test_bench_streaming_throughput(benchmark):
    """Benchmark streaming execution (bcall replacement)."""
    N = 500
    BATCH_SIZE = 25

    async def stream_task(x):
        await anyio.sleep(0.001)
        return x**2

    async def workload():
        async with AsyncExecutor(max_concurrent=20) as executor:
            all_results = []
            async for batch in executor.stream(
                stream_task, range(N), BATCH_SIZE
            ):
                all_results.extend(batch)
            return all_results

    result = benchmark(run_async_benchmark(workload))
    assert len(result) == N


# =============================================================================
# Error Handling & Retry Benchmarks
# =============================================================================


@pytest.mark.performance
@pytest.mark.benchmark(group="reliability")
def test_bench_retry_performance(benchmark):
    """Benchmark performance with retry logic."""
    N = 100
    failure_rate = 0.2  # 20% of tasks fail once

    async def flaky_task(x):
        # Use deterministic "randomness" based on x
        if (x * 7) % 10 < 2:  # ~20% failure rate
            if not hasattr(flaky_task, "failed"):
                flaky_task.failed = set()

            if x not in flaky_task.failed:
                flaky_task.failed.add(x)
                raise RuntimeError(f"Transient failure on {x}")

        await anyio.sleep(0.002)
        return x * 3

    async def workload():
        async with AsyncExecutor(
            max_concurrent=15,
            retry_attempts=2,
            retry_delay=0.001,  # Fast retry for benchmarking
        ) as executor:
            return await executor(flaky_task, range(N))

    result = benchmark(run_async_benchmark(workload))
    assert len(result) == N


# =============================================================================
# Latency Distribution Benchmarks
# =============================================================================


@pytest.mark.performance
@pytest.mark.benchmark(group="latency")
def test_bench_latency_distribution(benchmark):
    """Benchmark latency distribution for mixed task durations."""
    N = 200

    async def variable_latency_task(x):
        # Variable sleep times based on x
        delay = 0.001 + (x % 10) * 0.0005  # 1-5ms range
        await anyio.sleep(delay)
        return x

    async def workload():
        start_times = {}
        end_times = {}

        async def timed_task(x):
            start_times[x] = time.perf_counter()
            result = await variable_latency_task(x)
            end_times[x] = time.perf_counter()
            return result

        async with AsyncExecutor(max_concurrent=25) as executor:
            results = await executor(timed_task, range(N))

        # Calculate latency statistics
        latencies = [end_times[x] - start_times[x] for x in range(N)]
        latencies.sort()

        # Store stats for analysis
        workload.p50 = latencies[N // 2]
        workload.p95 = latencies[int(N * 0.95)]
        workload.p99 = latencies[int(N * 0.99)]

        return results

    result = benchmark(run_async_benchmark(workload))
    assert len(result) == N

    # Log latency percentiles
    print(
        f"Latency - P50: {workload.p50 * 1000:.2f}ms, "
        f"P95: {workload.p95 * 1000:.2f}ms, "
        f"P99: {workload.p99 * 1000:.2f}ms"
    )


# =============================================================================
# Resource Usage Benchmarks
# =============================================================================


@pytest.mark.performance
@pytest.mark.benchmark(group="resource-efficiency")
def test_bench_memory_efficiency(benchmark):
    """Benchmark memory-efficient execution patterns."""
    N = 2000
    LIMIT = 30

    async def memory_task(x):
        # Create some temporary data
        temp_data = [i for i in range(100)]  # Small allocation
        await anyio.sleep(0.001)
        return sum(temp_data) + x

    async def workload():
        async with AsyncExecutor(max_concurrent=LIMIT) as executor:
            return await executor(memory_task, range(N))

    result = benchmark(run_async_benchmark(workload))
    assert len(result) == N
    assert all(r >= 4950 for r in result)  # Basic result validation


# =============================================================================
# Backend Comparison Benchmarks
# =============================================================================


@pytest.mark.performance
@pytest.mark.benchmark(group="backend-comparison")
@pytest.mark.parametrize("backend", ["asyncio", "trio"])
def test_bench_backend_performance(benchmark, backend):
    """Compare performance across different async backends."""
    N = 300
    LIMIT = 20

    async def cross_backend_task(x):
        await anyio.sleep(0.002)
        return x * 2

    async def workload():
        async with AsyncExecutor(max_concurrent=LIMIT) as executor:
            return await executor(cross_backend_task, range(N))

    result = benchmark(run_async_benchmark(workload, backend=backend))
    assert len(result) == N


# =============================================================================
# Regression Performance Benchmarks
# =============================================================================


@pytest.mark.performance
@pytest.mark.benchmark(group="regression-performance")
def test_bench_cancellation_overhead(benchmark):
    """Benchmark performance impact of cancellation-safe patterns."""
    N = 400
    LIMIT = 15

    async def cancellable_task(x):
        try:
            await anyio.sleep(0.003)
            return x * 5
        except anyio.get_cancelled_exc_class():
            # Cleanup work
            await anyio.sleep(0.001)
            raise

    async def workload():
        async with AsyncExecutor(max_concurrent=LIMIT) as executor:
            return await executor(cancellable_task, range(N))

    result = benchmark(run_async_benchmark(workload))
    assert len(result) == N


@pytest.mark.performance
@pytest.mark.benchmark(group="regression-performance")
def test_bench_kwargs_overhead(benchmark):
    """Benchmark performance of functools.partial kwargs fix."""
    N = 200

    def task_with_kwargs(x, multiplier=2, offset=0):
        return (x * multiplier) + offset

    async def workload():
        async with AsyncExecutor(max_concurrent=10) as executor:
            # This tests the functools.partial pathway
            funcs = [task_with_kwargs] * N
            args_kwargs = [
                ((i,), {"multiplier": 3, "offset": i}) for i in range(N)
            ]
            return await executor(funcs, args_kwargs)

    result = benchmark(run_async_benchmark(workload))
    assert len(result) == N
    # Verify correct calculation: (i * 3) + i = i * 4
    assert result[0] == 0  # (0 * 3) + 0 = 0
    assert result[1] == 4  # (1 * 3) + 1 = 4
    assert result[10] == 40  # (10 * 3) + 10 = 40
