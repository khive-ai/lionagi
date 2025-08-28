# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmarks for streaming operations in lionagi v1 services.

Tests validate time-to-first-byte, sustained throughput for large streams,
memory consumption patterns, and circuit breaker + hooks streaming latency impact.
"""

import asyncio
import gc
import time
import tracemalloc
from collections.abc import AsyncIterator
from typing import Any, List
from unittest.mock import AsyncMock
from uuid import uuid4

import anyio
import pytest

from lionagi.services.core import CallContext, Service
from lionagi.services.endpoint import RequestModel
from lionagi.services.executor import ExecutorConfig, RateLimitedExecutor


class MockStreamingService:
    """Mock streaming service for performance testing."""

    name = "mock_streaming"

    def __init__(
        self,
        chunk_count: int = 100,
        chunk_size_bytes: int = 1024,
        chunk_interval_ms: float = 10,
        initial_delay_ms: float = 50,
    ):
        self.chunk_count = chunk_count
        self.chunk_size_bytes = chunk_size_bytes
        self.chunk_interval_ms = chunk_interval_ms
        self.initial_delay_ms = initial_delay_ms
        self.stream_count = 0

    async def call(self, req: RequestModel, *, ctx: CallContext) -> dict:
        """Non-streaming call."""
        return {"message": "Use stream() method for streaming"}

    async def stream(self, req: RequestModel, *, ctx: CallContext) -> AsyncIterator[dict]:
        """Stream chunks with configurable timing and size."""
        self.stream_count += 1
        stream_id = f"stream_{self.stream_count}"

        # Initial delay before first chunk (simulates API processing)
        if self.initial_delay_ms > 0:
            await anyio.sleep(self.initial_delay_ms / 1000.0)

        # Yield chunks
        for i in range(self.chunk_count):
            chunk_data = {
                "stream_id": stream_id,
                "chunk_index": i,
                "content": "x" * self.chunk_size_bytes,  # Fixed size content
                "timestamp": time.time(),
                "call_id": str(ctx.call_id),
                "is_last": i == self.chunk_count - 1,
            }

            yield chunk_data

            # Inter-chunk delay
            if i < self.chunk_count - 1 and self.chunk_interval_ms > 0:
                await anyio.sleep(self.chunk_interval_ms / 1000.0)


class MockLargeStreamService:
    """Mock service for testing very large streams."""

    name = "mock_large_stream"

    def __init__(self, total_mb: float = 10.0, chunk_size_kb: int = 64):
        self.total_bytes = int(total_mb * 1024 * 1024)
        self.chunk_size_bytes = chunk_size_kb * 1024
        self.chunk_count = self.total_bytes // self.chunk_size_bytes

    async def call(self, req: RequestModel, *, ctx: CallContext) -> dict:
        return {"message": "Use stream() for large streaming"}

    async def stream(self, req: RequestModel, *, ctx: CallContext) -> AsyncIterator[dict]:
        """Stream large amounts of data efficiently."""
        bytes_sent = 0
        chunk_index = 0

        while bytes_sent < self.total_bytes:
            remaining = self.total_bytes - bytes_sent
            chunk_size = min(self.chunk_size_bytes, remaining)

            chunk_data = {
                "chunk_index": chunk_index,
                "chunk_size": chunk_size,
                "bytes_sent": bytes_sent,
                "content": "A" * chunk_size,
                "progress": bytes_sent / self.total_bytes,
                "call_id": str(ctx.call_id),
            }

            yield chunk_data

            bytes_sent += chunk_size
            chunk_index += 1

            # Minimal delay to avoid overwhelming
            await anyio.sleep(0.001)


@pytest.fixture
def mock_request():
    """Standard mock request."""
    return RequestModel(
        messages=[{"role": "user", "content": "stream this"}],
        model="test-streaming-model",
    )


class TestTimeToFirstByte:
    """Test time-to-first-byte (TTFB) for streaming responses."""

    @pytest.mark.anyio
    async def test_ttfb_fast_stream(self, mock_request):
        """Measure TTFB for fast streaming service."""
        config = ExecutorConfig(queue_capacity=10, concurrency_limit=5)
        executor = RateLimitedExecutor(config)

        # Fast service with 10ms initial delay
        service = MockStreamingService(chunk_count=50, chunk_interval_ms=1, initial_delay_ms=10)

        await executor.start()

        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=30.0)

        # Measure TTFB
        start_time = time.perf_counter()
        stream = executor.submit_stream(service, mock_request, ctx)

        # Get first chunk
        first_chunk = None
        async for chunk in stream:
            first_chunk = chunk
            ttfb = time.perf_counter() - start_time
            break

        await executor.stop()

        # TTFB should be close to initial_delay_ms plus minimal overhead
        assert ttfb < 0.05, f"TTFB too high: {ttfb * 1000:.1f}ms"
        assert first_chunk is not None
        assert first_chunk["chunk_index"] == 0
        assert "call_id" in first_chunk

    @pytest.mark.anyio
    async def test_ttfb_with_queue_wait(self, mock_request):
        """Measure TTFB when streams must wait in queue."""
        config = ExecutorConfig(
            queue_capacity=20,
            concurrency_limit=2,  # Low limit forces queueing
            limit_requests=10,
        )
        executor = RateLimitedExecutor(config)

        service = MockStreamingService(
            chunk_count=10,
            chunk_interval_ms=50,  # Longer chunks to hold concurrency slots
            initial_delay_ms=20,
        )

        await executor.start()

        # Start multiple streams to saturate concurrency
        streams = []
        ttfbs = []

        for i in range(5):  # More than concurrency limit
            ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)
            start_time = time.perf_counter()

            stream = executor.submit_stream(service, mock_request, ctx)
            streams.append((stream, start_time))

        # Measure TTFB for each stream
        for stream, start_time in streams:
            async for chunk in stream:
                ttfb = time.perf_counter() - start_time
                ttfbs.append(ttfb)
                break  # Only need first chunk

        await executor.stop()

        # First streams should have fast TTFB, later ones may queue
        assert len(ttfbs) == 5
        assert min(ttfbs) < 0.1, "Even queued streams should have reasonable TTFB"

        # Some variance expected due to queueing
        ttfb_range = max(ttfbs) - min(ttfbs)
        assert ttfb_range < 2.0, f"TTFB variance too high: {ttfb_range:.3f}s"

    @pytest.mark.anyio
    async def test_ttfb_multiple_concurrent_streams(self, mock_request):
        """Test TTFB consistency across multiple concurrent streams."""
        config = ExecutorConfig(queue_capacity=50, concurrency_limit=10, limit_requests=50)
        executor = RateLimitedExecutor(config)

        service = MockStreamingService(chunk_count=20, chunk_interval_ms=5, initial_delay_ms=25)

        await executor.start()

        # Start many concurrent streams
        concurrent_streams = 15
        ttfbs = []

        async def measure_stream_ttfb():
            ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)
            start_time = time.perf_counter()

            stream = executor.submit_stream(service, mock_request, ctx)
            async for chunk in stream:
                ttfb = time.perf_counter() - start_time
                return ttfb

        # Launch concurrent measurements
        async with anyio.create_task_group() as tg:
            results = []
            for _ in range(concurrent_streams):
                results.append(await tg.start_soon(measure_stream_ttfb))

            # Wait for results
            ttfbs = [await result for result in results]

        await executor.stop()

        # Validate TTFB consistency
        assert len(ttfbs) == concurrent_streams
        avg_ttfb = sum(ttfbs) / len(ttfbs)
        max_ttfb = max(ttfbs)

        assert avg_ttfb < 0.1, f"Average TTFB too high: {avg_ttfb * 1000:.1f}ms"
        assert max_ttfb < 0.3, f"Max TTFB too high: {max_ttfb * 1000:.1f}ms"


class TestSustainedThroughput:
    """Test sustained throughput for long-running streams."""

    @pytest.mark.anyio
    async def test_small_chunks_high_frequency(self, mock_request):
        """Test throughput with many small chunks (10k chunks)."""
        config = ExecutorConfig(queue_capacity=5, concurrency_limit=5)
        executor = RateLimitedExecutor(config)

        # 10k small chunks
        service = MockStreamingService(
            chunk_count=10000,
            chunk_size_bytes=100,  # 100 bytes per chunk
            chunk_interval_ms=0.1,  # Very fast
            initial_delay_ms=5,
        )

        await executor.start()

        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=60.0)

        start_time = time.perf_counter()
        chunk_count = 0
        total_bytes = 0

        stream = executor.submit_stream(service, mock_request, ctx)
        async for chunk in stream:
            chunk_count += 1
            total_bytes += len(chunk.get("content", ""))

            # Sample progress every 1000 chunks
            if chunk_count % 1000 == 0:
                elapsed = time.perf_counter() - start_time
                rate = chunk_count / elapsed if elapsed > 0 else 0
                print(f"  {chunk_count} chunks, {rate:.0f} chunks/sec")

        total_time = time.perf_counter() - start_time

        await executor.stop()

        # Validate throughput
        chunks_per_second = chunk_count / total_time
        bytes_per_second = total_bytes / total_time

        assert chunk_count == 10000, f"Missing chunks: got {chunk_count}"
        assert chunks_per_second > 1000, f"Throughput too low: {chunks_per_second:.0f} chunks/sec"
        assert bytes_per_second > 100000, f"Bandwidth too low: {bytes_per_second / 1024:.0f} KB/sec"
        assert total_time < 30, f"Stream took too long: {total_time:.1f}s"

    @pytest.mark.anyio
    async def test_large_chunks_sustained_bandwidth(self, mock_request):
        """Test sustained bandwidth with large chunks."""
        config = ExecutorConfig(queue_capacity=5, concurrency_limit=3)
        executor = RateLimitedExecutor(config)

        # Large stream: 10MB total
        service = MockLargeStreamService(total_mb=10.0, chunk_size_kb=256)

        await executor.start()

        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=30.0)

        start_time = time.perf_counter()
        chunk_count = 0
        total_bytes = 0

        stream = executor.submit_stream(service, mock_request, ctx)
        async for chunk in stream:
            chunk_count += 1
            total_bytes += chunk["chunk_size"]

            # Progress reporting
            if chunk_count % 10 == 0:
                progress = chunk.get("progress", 0)
                print(f"  Progress: {progress * 100:.1f}% ({total_bytes // 1024} KB)")

        total_time = time.perf_counter() - start_time

        await executor.stop()

        # Validate sustained performance
        mb_per_second = (total_bytes / 1024 / 1024) / total_time
        expected_bytes = 10 * 1024 * 1024  # 10MB

        assert total_bytes >= expected_bytes * 0.99, f"Data loss: {total_bytes} < {expected_bytes}"
        assert mb_per_second > 5, f"Bandwidth too low: {mb_per_second:.1f} MB/sec"
        assert total_time < 10, f"Stream too slow: {total_time:.1f}s for 10MB"

    @pytest.mark.anyio
    async def test_concurrent_streams_throughput(self, mock_request):
        """Test throughput with multiple concurrent streams."""
        config = ExecutorConfig(queue_capacity=20, concurrency_limit=8, limit_requests=50)
        executor = RateLimitedExecutor(config)

        # Medium streams concurrently
        service = MockStreamingService(
            chunk_count=1000,
            chunk_size_bytes=512,
            chunk_interval_ms=1,
            initial_delay_ms=10,
        )

        await executor.start()

        concurrent_streams = 5
        results = []

        async def process_stream():
            ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=30.0)

            start_time = time.perf_counter()
            chunk_count = 0
            total_bytes = 0

            stream = executor.submit_stream(service, mock_request, ctx)
            async for chunk in stream:
                chunk_count += 1
                total_bytes += len(chunk.get("content", ""))

            duration = time.perf_counter() - start_time
            return {
                "chunks": chunk_count,
                "bytes": total_bytes,
                "duration": duration,
                "chunks_per_sec": chunk_count / duration,
                "bytes_per_sec": total_bytes / duration,
            }

        # Run concurrent streams
        async with anyio.create_task_group() as tg:
            tasks = []
            for _ in range(concurrent_streams):
                task = await tg.start_soon(process_stream)
                tasks.append(task)

            results = [await task for task in tasks]

        await executor.stop()

        # Validate concurrent performance
        assert len(results) == concurrent_streams

        total_chunks = sum(r["chunks"] for r in results)
        total_bytes = sum(r["bytes"] for r in results)
        avg_duration = sum(r["duration"] for r in results) / len(results)
        combined_throughput = total_chunks / avg_duration

        assert total_chunks == concurrent_streams * 1000, "Missing chunks in concurrent streams"
        assert (
            combined_throughput > 2000
        ), f"Combined throughput too low: {combined_throughput:.0f} chunks/sec"

        # Individual streams should maintain reasonable performance
        min_individual_rate = min(r["chunks_per_sec"] for r in results)
        assert (
            min_individual_rate > 300
        ), f"Slowest stream too slow: {min_individual_rate:.0f} chunks/sec"


class TestMemoryConsumption:
    """Test memory consumption during streaming to ensure no buffering regression."""

    @pytest.mark.anyio
    async def test_streaming_memory_efficiency(self, mock_request):
        """Verify streaming doesn't buffer entire response in memory."""
        config = ExecutorConfig(queue_capacity=5, concurrency_limit=2)
        executor = RateLimitedExecutor(config)

        # Large stream that would use significant memory if buffered
        service = MockLargeStreamService(total_mb=50.0, chunk_size_kb=128)

        # Start memory tracking
        tracemalloc.start()
        gc.collect()
        baseline = tracemalloc.take_snapshot()

        await executor.start()

        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=60.0)

        # Process stream while monitoring memory
        peak_memory_mb = 0
        chunk_count = 0
        processed_mb = 0

        stream = executor.submit_stream(service, mock_request, ctx)
        async for chunk in stream:
            chunk_count += 1
            processed_mb += chunk["chunk_size"] / 1024 / 1024

            # Check memory every 50 chunks
            if chunk_count % 50 == 0:
                current = tracemalloc.take_snapshot()
                top_stats = current.compare_to(baseline, "lineno")
                memory_mb = sum(stat.size for stat in top_stats) / 1024 / 1024
                peak_memory_mb = max(peak_memory_mb, memory_mb)

                print(f"  Processed: {processed_mb:.1f}MB, Memory: {memory_mb:.1f}MB")

        await executor.stop()
        tracemalloc.stop()

        # Memory usage should be bounded, not proportional to stream size
        assert processed_mb > 45, f"Stream incomplete: {processed_mb:.1f}MB < 45MB"
        assert peak_memory_mb < 10, f"Memory usage too high: {peak_memory_mb:.1f}MB for 50MB stream"

        # Memory should not scale with stream size (key streaming property)
        memory_ratio = peak_memory_mb / processed_mb
        assert memory_ratio < 0.2, f"Memory scales with stream size: {memory_ratio:.2f} ratio"

    @pytest.mark.anyio
    async def test_concurrent_streams_memory_isolation(self, mock_request):
        """Test memory isolation between concurrent streams."""
        config = ExecutorConfig(queue_capacity=15, concurrency_limit=5)
        executor = RateLimitedExecutor(config)

        # Multiple medium-size streams
        service = MockStreamingService(
            chunk_count=2000,
            chunk_size_bytes=1024,
            chunk_interval_ms=0.5,  # 1KB chunks
        )

        tracemalloc.start()
        gc.collect()
        baseline = tracemalloc.take_snapshot()

        await executor.start()

        concurrent_streams = 4
        memory_samples = []

        async def monitor_memory_stream():
            ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=30.0)

            chunk_count = 0
            stream = executor.submit_stream(service, mock_request, ctx)

            async for chunk in stream:
                chunk_count += 1

                # Sample memory periodically
                if chunk_count % 200 == 0:
                    current = tracemalloc.take_snapshot()
                    top_stats = current.compare_to(baseline, "lineno")
                    memory_mb = sum(stat.size for stat in top_stats) / 1024 / 1024
                    memory_samples.append(memory_mb)

            return chunk_count

        # Run concurrent streams with memory monitoring
        async with anyio.create_task_group() as tg:
            tasks = []
            for _ in range(concurrent_streams):
                task = await tg.start_soon(monitor_memory_stream)
                tasks.append(task)

            chunk_counts = [await task for task in tasks]

        await executor.stop()
        tracemalloc.stop()

        # Validate memory behavior
        total_chunks = sum(chunk_counts)
        assert total_chunks == concurrent_streams * 2000, "Missing chunks in concurrent streams"

        if memory_samples:
            max_memory = max(memory_samples)
            avg_memory = sum(memory_samples) / len(memory_samples)

            # Memory should not scale linearly with number of concurrent streams
            assert (
                max_memory < 20
            ), f"Peak memory too high: {max_memory:.1f}MB for {concurrent_streams} streams"
            assert avg_memory < 15, f"Average memory too high: {avg_memory:.1f}MB"


class TestCircuitBreakerStreamingLatency:
    """Test impact of circuit breaker and hooks on streaming latency."""

    @pytest.mark.anyio
    async def test_streaming_without_middleware_baseline(self, mock_request):
        """Establish baseline streaming performance without middleware."""
        config = ExecutorConfig(queue_capacity=10, concurrency_limit=5)
        executor = RateLimitedExecutor(config)

        service = MockStreamingService(chunk_count=500, chunk_size_bytes=256, chunk_interval_ms=2)

        await executor.start()

        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=20.0)

        # Measure baseline performance
        start_time = time.perf_counter()
        chunk_count = 0

        stream = executor.submit_stream(service, mock_request, ctx)
        async for chunk in stream:
            chunk_count += 1

        baseline_time = time.perf_counter() - start_time
        baseline_rate = chunk_count / baseline_time

        await executor.stop()

        assert chunk_count == 500
        assert baseline_rate > 100, f"Baseline too slow: {baseline_rate:.0f} chunks/sec"

        return baseline_time, baseline_rate

    @pytest.mark.anyio
    async def test_streaming_with_circuit_breaker_overhead(self, mock_request):
        """Test streaming performance impact of circuit breaker middleware."""
        # This would require circuit breaker middleware implementation
        # For now, we'll simulate the test structure

        config = ExecutorConfig(queue_capacity=10, concurrency_limit=5)
        executor = RateLimitedExecutor(config)

        service = MockStreamingService(chunk_count=500, chunk_size_bytes=256, chunk_interval_ms=2)

        await executor.start()

        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=20.0)

        # TODO: Add circuit breaker middleware when available
        # For now, measure performance with potential middleware overhead simulation
        start_time = time.perf_counter()
        chunk_count = 0

        stream = executor.submit_stream(service, mock_request, ctx)
        async for chunk in stream:
            chunk_count += 1
            # Simulate minimal middleware overhead
            await anyio.sleep(0.0001)

        middleware_time = time.perf_counter() - start_time
        middleware_rate = chunk_count / middleware_time

        await executor.stop()

        # Middleware overhead should be minimal
        assert chunk_count == 500
        assert (
            middleware_rate > 90
        ), f"Middleware overhead too high: {middleware_rate:.0f} chunks/sec"


class TestStreamingResilience:
    """Test streaming performance under failure conditions."""

    @pytest.mark.anyio
    async def test_partial_stream_failure_recovery(self, mock_request):
        """Test performance when streams fail partway through."""

        class FailingStreamService:
            name = "failing_stream"

            def __init__(self, fail_after_chunks: int = 100):
                self.fail_after_chunks = fail_after_chunks

            async def call(self, req: RequestModel, *, ctx: CallContext) -> dict:
                return {"message": "Use stream()"}

            async def stream(self, req: RequestModel, *, ctx: CallContext) -> AsyncIterator[dict]:
                for i in range(self.fail_after_chunks + 50):  # Will fail before completion
                    if i == self.fail_after_chunks:
                        raise Exception("Simulated stream failure")

                    yield {
                        "chunk_index": i,
                        "content": f"chunk_{i}",
                        "call_id": str(ctx.call_id),
                    }
                    await anyio.sleep(0.001)

        config = ExecutorConfig(queue_capacity=10, concurrency_limit=5)
        executor = RateLimitedExecutor(config)

        service = FailingStreamService(fail_after_chunks=100)

        await executor.start()

        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        # Stream should fail after 100 chunks
        chunk_count = 0
        failed = False

        try:
            stream = executor.submit_stream(service, mock_request, ctx)
            async for chunk in stream:
                chunk_count += 1
        except Exception:
            failed = True

        await executor.stop()

        # Should have received some chunks before failure
        assert chunk_count == 100, f"Expected 100 chunks before failure, got {chunk_count}"
        assert failed, "Stream should have failed"

    @pytest.mark.anyio
    async def test_stream_cancellation_cleanup(self, mock_request):
        """Test performance impact of stream cancellation and cleanup."""
        config = ExecutorConfig(queue_capacity=10, concurrency_limit=5)
        executor = RateLimitedExecutor(config)

        service = MockStreamingService(
            chunk_count=10000,  # Very long stream
            chunk_size_bytes=256,
            chunk_interval_ms=1,
        )

        await executor.start()

        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)

        # Start stream and cancel after receiving some chunks
        chunk_count = 0
        start_time = time.perf_counter()

        try:
            stream = executor.submit_stream(service, mock_request, ctx)

            async for chunk in stream:
                chunk_count += 1

                # Cancel after 100 chunks
                if chunk_count == 100:
                    break

        except Exception:
            pass

        cancel_time = time.perf_counter() - start_time

        await executor.stop()

        # Cancellation should be fast
        assert chunk_count <= 100, f"Received too many chunks: {chunk_count}"
        assert cancel_time < 2.0, f"Cancellation took too long: {cancel_time:.2f}s"
