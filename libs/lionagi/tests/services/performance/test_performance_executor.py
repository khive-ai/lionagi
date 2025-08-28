# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmarks for RateLimitedExecutor under realistic load conditions.

Tests validate executor throughput, queue efficiency, memory usage patterns, and
overhead measurements with/without hooks+metrics+retry middleware.
"""

import asyncio
import gc
import time
import tracemalloc
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import anyio
import pytest

from lionagi.services.core import CallContext, Service
from lionagi.services.endpoint import RequestModel
from lionagi.services.executor import ExecutorConfig, RateLimitedExecutor, ServiceCall


class MockFastService:
    """Mock service that responds quickly for throughput testing."""

    name = "mock_fast"

    def __init__(self, delay_ms: float = 1.0):
        self.delay_ms = delay_ms
        self.call_count = 0

    async def call(self, req: RequestModel, *, ctx: CallContext) -> dict:
        """Fast mock call with configurable delay."""
        await anyio.sleep(self.delay_ms / 1000.0)  # Convert ms to seconds
        self.call_count += 1
        return {
            "result": f"response_{self.call_count}",
            "call_id": str(ctx.call_id),
            "delay_ms": self.delay_ms,
        }

    async def stream(self, req: RequestModel, *, ctx: CallContext) -> AsyncIterator[dict]:
        """Fast mock streaming with configurable chunks."""
        for i in range(10):  # 10 small chunks
            await anyio.sleep(self.delay_ms / 10000.0)  # Tiny delay per chunk
            yield {"chunk_id": i, "content": f"chunk_{i}", "call_id": str(ctx.call_id)}


class MockSlowService:
    """Mock service that responds slowly to test mixed loads."""

    name = "mock_slow"

    def __init__(self, delay_ms: float = 1000.0):
        self.delay_ms = delay_ms
        self.call_count = 0

    async def call(self, req: RequestModel, *, ctx: CallContext) -> dict:
        """Slow mock call for testing mixed workloads."""
        await anyio.sleep(self.delay_ms / 1000.0)
        self.call_count += 1
        return {
            "result": f"slow_response_{self.call_count}",
            "call_id": str(ctx.call_id),
            "delay_ms": self.delay_ms,
        }

    async def stream(self, req: RequestModel, *, ctx: CallContext) -> AsyncIterator[dict]:
        """Slow mock streaming."""
        for i in range(5):  # Fewer, slower chunks
            await anyio.sleep(self.delay_ms / 5000.0)
            yield {
                "chunk_id": i,
                "content": f"slow_chunk_{i}",
                "call_id": str(ctx.call_id),
            }


@pytest.fixture
def mock_request():
    """Create a mock request for testing."""
    return RequestModel(messages=[{"role": "user", "content": "test message"}], model="test-model")


class TestExecutorThroughput:
    """Test executor throughput under various load conditions."""

    @pytest.mark.anyio
    async def test_high_throughput_fast_calls(self, mock_request):
        """Benchmark executor throughput with many fast calls."""
        config = ExecutorConfig(
            queue_capacity=1000,
            limit_requests=100,
            capacity_refresh_time=1.0,
            concurrency_limit=20,
        )

        executor = RateLimitedExecutor(config)
        service = MockFastService(delay_ms=10)  # 10ms per call

        await executor.start()

        # Submit 200 calls
        start_time = time.perf_counter()
        calls = []

        for i in range(200):
            ctx = CallContext.with_timeout(
                branch_id=uuid4(),
                timeout_s=30.0,
                batch_id=f"batch_{i // 50}",  # Group into batches
            )
            call = await executor.submit_call(service, mock_request, ctx)
            calls.append(call)

        # Wait for all calls to complete
        results = []
        for call in calls:
            result = await call.wait_completion()
            results.append(result)

        total_time = time.perf_counter() - start_time

        await executor.stop()

        # Validate performance
        assert len(results) == 200
        assert service.call_count == 200

        # Should handle at least 50 calls/second (conservative)
        calls_per_second = len(results) / total_time
        assert calls_per_second >= 50, f"Only achieved {calls_per_second:.1f} calls/sec"

        # Validate queue statistics
        stats = executor.stats
        assert stats["calls_completed"] == 200
        assert stats["calls_failed"] == 0

        # Queue wait times should be reasonable
        if stats["queue_wait_times"]:
            avg_wait = sum(stats["queue_wait_times"]) / len(stats["queue_wait_times"])
            assert avg_wait < 0.5, f"Average queue wait too high: {avg_wait:.3f}s"

    @pytest.mark.anyio
    async def test_mixed_load_performance(self, mock_request):
        """Test executor performance with mixed fast/slow calls."""
        config = ExecutorConfig(
            queue_capacity=100,
            limit_requests=20,
            capacity_refresh_time=1.0,
            concurrency_limit=10,
        )

        executor = RateLimitedExecutor(config)
        fast_service = MockFastService(delay_ms=5)
        slow_service = MockSlowService(delay_ms=200)  # 200ms

        await executor.start()

        start_time = time.perf_counter()
        calls = []

        # Submit mix of fast and slow calls
        for i in range(40):
            ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

            if i % 3 == 0:  # Every 3rd call is slow
                call = await executor.submit_call(slow_service, mock_request, ctx)
            else:
                call = await executor.submit_call(fast_service, mock_request, ctx)
            calls.append(call)

        # Wait for completion
        results = []
        for call in calls:
            result = await call.wait_completion()
            results.append(result)

        total_time = time.perf_counter() - start_time

        await executor.stop()

        # Validate mixed load handling
        assert len(results) == 40
        assert fast_service.call_count + slow_service.call_count == 40

        # Should complete within reasonable time despite slow calls
        assert total_time < 15.0, f"Mixed load took too long: {total_time:.2f}s"

        stats = executor.stats
        assert stats["calls_completed"] == 40
        assert stats["calls_failed"] == 0


class TestQueueEfficiency:
    """Test queue efficiency and latency measurement."""

    @pytest.mark.anyio
    async def test_queue_latency_measurement(self, mock_request):
        """Measure latency between submit and execution start."""
        config = ExecutorConfig(
            queue_capacity=50,
            limit_requests=5,  # Low limit to create queueing
            capacity_refresh_time=0.5,
        )

        executor = RateLimitedExecutor(config)
        service = MockFastService(delay_ms=100)

        await executor.start()

        # Submit calls that will queue due to rate limits
        submission_times = []
        calls = []

        for i in range(20):
            submit_time = time.perf_counter()
            ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)
            call = await executor.submit_call(service, mock_request, ctx)

            submission_times.append(submit_time)
            calls.append(call)

        # Wait for all to complete
        for call in calls:
            await call.wait_completion()

        await executor.stop()

        # Analyze queue latencies
        stats = executor.stats
        assert "queue_wait_times" in stats
        assert len(stats["queue_wait_times"]) > 0

        # First calls should have minimal wait, later calls should queue
        wait_times = stats["queue_wait_times"]
        avg_wait = sum(wait_times) / len(wait_times)
        max_wait = max(wait_times)

        # Some calls should have been queued (non-zero wait)
        assert max_wait > 0.01, "Expected some calls to be queued"

        # But average wait shouldn't be excessive
        assert avg_wait < 2.0, f"Average queue wait too high: {avg_wait:.3f}s"

    @pytest.mark.anyio
    async def test_efficient_memory_streams(self, mock_request):
        """Verify efficient queuing with anyio memory object streams."""
        config = ExecutorConfig(queue_capacity=10)

        executor = RateLimitedExecutor(config)
        service = MockFastService(delay_ms=1)

        await executor.start()

        # Fill queue to capacity
        calls = []
        for i in range(10):
            ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)
            call = await executor.submit_call(service, mock_request, ctx)
            calls.append(call)

        # 11th call should fail immediately (no polling delay)
        start_time = time.perf_counter()
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)

        with pytest.raises(Exception, match="queue at capacity"):
            await executor.submit_call(service, mock_request, ctx)

        rejection_time = time.perf_counter() - start_time

        # Should reject immediately, not after polling delay
        assert rejection_time < 0.01, f"Queue rejection too slow: {rejection_time:.3f}s"

        # Clean up
        for call in calls:
            await call.wait_completion()
        await executor.stop()


class TestMemoryUsage:
    """Test memory usage patterns under sustained load."""

    @pytest.mark.anyio
    async def test_memory_usage_sustained_load(self, mock_request):
        """Monitor memory usage during sustained load."""
        config = ExecutorConfig(queue_capacity=200, limit_requests=50, capacity_refresh_time=1.0)

        executor = RateLimitedExecutor(config)
        service = MockFastService(delay_ms=20)

        # Start memory tracking
        tracemalloc.start()
        gc.collect()  # Clean start
        baseline_snapshot = tracemalloc.take_snapshot()

        await executor.start()

        # Sustained load - submit calls in waves
        total_calls = 300
        wave_size = 50

        for wave in range(total_calls // wave_size):
            wave_calls = []

            # Submit wave
            for i in range(wave_size):
                ctx = CallContext.with_timeout(
                    branch_id=uuid4(), timeout_s=30.0, wave_id=f"wave_{wave}"
                )
                call = await executor.submit_call(service, mock_request, ctx)
                wave_calls.append(call)

            # Wait for wave to complete
            for call in wave_calls:
                await call.wait_completion()

            # Brief pause between waves
            await anyio.sleep(0.1)

        # Take final memory snapshot
        final_snapshot = tracemalloc.take_snapshot()
        await executor.stop()

        # Analyze memory usage
        top_stats = final_snapshot.compare_to(baseline_snapshot, "lineno")
        total_memory_mb = sum(stat.size for stat in top_stats) / 1024 / 1024

        # Memory usage should be reasonable for 300 calls
        assert total_memory_mb < 50, f"Memory usage too high: {total_memory_mb:.1f}MB"

        # Validate completion
        assert service.call_count == total_calls
        stats = executor.stats
        assert stats["calls_completed"] == total_calls

        tracemalloc.stop()

    @pytest.mark.anyio
    async def test_completed_calls_cleanup(self, mock_request):
        """Test that completed calls are cleaned up to prevent memory leaks."""
        config = ExecutorConfig(queue_capacity=50)

        executor = RateLimitedExecutor(config)
        service = MockFastService(delay_ms=1)

        await executor.start()

        # Submit many calls to trigger cleanup
        for batch in range(25):  # 25 batches of 10 = 250 calls
            batch_calls = []

            for i in range(10):
                ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)
                call = await executor.submit_call(service, mock_request, ctx)
                batch_calls.append(call)

            # Wait for batch completion
            for call in batch_calls:
                await call.wait_completion()

        await executor.stop()

        # Check that completed calls were cleaned up
        stats = executor.stats
        completed_count = stats["completed_calls"]

        # Should have automatically cleaned up old calls
        assert completed_count <= 1000, f"Too many completed calls retained: {completed_count}"
        assert service.call_count == 250


class TestOverheadMeasurement:
    """Measure overhead with/without hooks, metrics, and retry middleware."""

    @pytest.mark.anyio
    async def test_baseline_executor_overhead(self, mock_request):
        """Measure baseline executor overhead without middleware."""
        config = ExecutorConfig(
            queue_capacity=100,
            limit_requests=200,  # High limit to avoid rate limiting
            concurrency_limit=50,
        )

        executor = RateLimitedExecutor(config)
        service = MockFastService(delay_ms=1)  # Minimal service time

        await executor.start()

        # Measure direct service call time
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)

        direct_start = time.perf_counter()
        direct_result = await service.call(mock_request, ctx=ctx)
        direct_time = time.perf_counter() - direct_start

        # Measure executor call time
        executor_start = time.perf_counter()
        call = await executor.submit_call(service, mock_request, ctx)
        executor_result = await call.wait_completion()
        executor_time = time.perf_counter() - executor_start

        await executor.stop()

        # Calculate overhead
        overhead_ms = (executor_time - direct_time) * 1000
        overhead_ratio = executor_time / direct_time if direct_time > 0 else float("inf")

        # Overhead should be minimal
        assert overhead_ms < 10, f"Executor overhead too high: {overhead_ms:.2f}ms"
        assert overhead_ratio < 5, f"Overhead ratio too high: {overhead_ratio:.1f}x"

        # Results should be equivalent
        assert direct_result["result"].startswith("response_")
        assert executor_result["result"].startswith("response_")

    @pytest.mark.anyio
    async def test_concurrent_execution_efficiency(self, mock_request):
        """Test that concurrent execution is efficient vs sequential."""
        config = ExecutorConfig(queue_capacity=100, limit_requests=100, concurrency_limit=10)

        executor = RateLimitedExecutor(config)
        service = MockFastService(delay_ms=100)  # 100ms per call

        await executor.start()

        call_count = 20

        # Measure concurrent execution
        concurrent_start = time.perf_counter()
        calls = []

        for i in range(call_count):
            ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)
            call = await executor.submit_call(service, mock_request, ctx)
            calls.append(call)

        # Wait for all concurrent calls
        results = []
        for call in calls:
            result = await call.wait_completion()
            results.append(result)

        concurrent_time = time.perf_counter() - concurrent_start

        await executor.stop()

        # Sequential time would be ~20 * 100ms = 2000ms
        # Concurrent should be much faster due to parallelism
        expected_sequential_time = call_count * 0.1  # 2 seconds

        assert concurrent_time < (
            expected_sequential_time * 0.6
        ), f"Concurrent execution not efficient: {concurrent_time:.2f}s vs expected <{expected_sequential_time * 0.6:.2f}s"

        assert len(results) == call_count
        assert service.call_count == call_count


class TestRateLimitingAccuracy:
    """Test rate limiting accuracy under concurrent load."""

    @pytest.mark.anyio
    async def test_request_rate_limiting_accuracy(self, mock_request):
        """Test that request rate limiting is accurate under concurrent load."""
        config = ExecutorConfig(
            queue_capacity=200,
            limit_requests=10,  # 10 requests per refresh period
            capacity_refresh_time=1.0,  # 1 second refresh
            concurrency_limit=20,
        )

        executor = RateLimitedExecutor(config)
        service = MockFastService(delay_ms=10)

        await executor.start()

        # Submit 30 calls rapidly (should take ~3 seconds due to rate limit)
        start_time = time.perf_counter()
        calls = []

        for i in range(30):
            ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=15.0)
            call = await executor.submit_call(service, mock_request, ctx)
            calls.append(call)

        # Wait for all to complete
        for call in calls:
            await call.wait_completion()

        total_time = time.perf_counter() - start_time

        await executor.stop()

        # Should take approximately 3 seconds (30 calls / 10 per second)
        expected_min_time = 2.5  # Allow some tolerance
        expected_max_time = 4.0

        assert (
            expected_min_time <= total_time <= expected_max_time
        ), f"Rate limiting inaccurate: {total_time:.2f}s (expected {expected_min_time}-{expected_max_time}s)"

        stats = executor.stats
        assert stats["calls_completed"] == 30

        # Verify rate limiting worked (some calls should have waited)
        if stats["queue_wait_times"]:
            max_wait = max(stats["queue_wait_times"])
            assert max_wait > 0.5, "Expected some calls to wait for rate limit refresh"


class TestResourceUtilization:
    """Test resource utilization patterns."""

    @pytest.mark.anyio
    async def test_cpu_and_memory_efficiency(self, mock_request):
        """Test CPU and memory efficiency under load."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        config = ExecutorConfig(
            queue_capacity=100,
            limit_requests=100,
            capacity_refresh_time=1.0,
            concurrency_limit=20,
        )

        executor = RateLimitedExecutor(config)
        service = MockFastService(delay_ms=50)

        # Baseline measurements
        baseline_cpu = process.cpu_percent()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        await executor.start()

        # Submit sustained load
        calls = []
        for i in range(50):
            ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)
            call = await executor.submit_call(service, mock_request, ctx)
            calls.append(call)

            # Small delay to spread submissions
            if i % 10 == 0:
                await anyio.sleep(0.1)

        # Monitor during execution
        cpu_samples = []
        memory_samples = []

        for _ in range(10):  # Sample 10 times during execution
            cpu_samples.append(process.cpu_percent())
            memory_samples.append(process.memory_info().rss / 1024 / 1024)
            await anyio.sleep(0.2)

        # Wait for completion
        for call in calls:
            await call.wait_completion()

        await executor.stop()

        # Analyze resource usage
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
        max_memory = max(memory_samples) if memory_samples else baseline_memory
        memory_increase = max_memory - baseline_memory

        # CPU usage should be reasonable
        assert avg_cpu < 80, f"CPU usage too high: {avg_cpu:.1f}%"

        # Memory increase should be bounded
        assert memory_increase < 100, f"Memory increase too high: {memory_increase:.1f}MB"

        stats = executor.stats
        assert stats["calls_completed"] == 50
