# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Integration performance benchmarks for full lionagi v1 services pipeline.

Tests validate full pipeline latency with all middleware enabled, concurrent request
handling scalability, and resource utilization under various realistic load patterns.
"""

import asyncio
import gc
import json
import time
import tracemalloc
from collections.abc import AsyncIterator
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import anyio
import pytest

from lionagi.services.core import CallContext, Service
from lionagi.services.endpoint import ChatRequestModel, RequestModel
from lionagi.services.executor import ExecutorConfig, RateLimitedExecutor


class MockFullPipelineService:
    """Mock service that simulates realistic API behavior for integration testing."""

    name = "mock_full_pipeline"
    requires = {"net.out:api.mock.com"}

    def __init__(
        self,
        latency_ms: float = 100,
        failure_rate: float = 0.0,
        rate_limit_delay_ms: float = 0,
        response_size_bytes: int = 2048,
    ):
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self.rate_limit_delay_ms = rate_limit_delay_ms
        self.response_size_bytes = response_size_bytes
        self.call_count = 0
        self.failure_count = 0

    async def call(self, req: RequestModel, *, ctx: CallContext) -> dict:
        """Simulate realistic API call with configurable behavior."""
        self.call_count += 1

        # Simulate rate limiting
        if self.rate_limit_delay_ms > 0:
            await anyio.sleep(self.rate_limit_delay_ms / 1000.0)

        # Simulate processing latency
        await anyio.sleep(self.latency_ms / 1000.0)

        # Simulate failures
        if self.failure_rate > 0 and (self.call_count * self.failure_rate) > self.failure_count:
            self.failure_count += 1
            raise Exception(f"Simulated API failure #{self.failure_count}")

        # Generate realistic response
        response_content = "A" * self.response_size_bytes
        return {
            "id": f"response_{self.call_count}",
            "model": "mock-model-v1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(str(req)),
                "completion_tokens": self.response_size_bytes // 4,  # Rough estimate
                "total_tokens": len(str(req)) + (self.response_size_bytes // 4),
            },
            "call_id": str(ctx.call_id),
            "processing_time_ms": self.latency_ms,
        }

    async def stream(self, req: RequestModel, *, ctx: CallContext) -> AsyncIterator[dict]:
        """Simulate streaming response."""
        self.call_count += 1

        # Initial delay
        await anyio.sleep(self.latency_ms / 1000.0)

        # Stream response in chunks
        chunk_count = 20
        chunk_size = self.response_size_bytes // chunk_count

        for i in range(chunk_count):
            if (
                self.failure_rate > 0
                and i > 10
                and (self.call_count * self.failure_rate) > self.failure_count
            ):
                self.failure_count += 1
                raise Exception(f"Simulated streaming failure #{self.failure_count}")

            yield {
                "id": f"stream_{self.call_count}",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "B" * chunk_size,
                            "role": "assistant" if i == 0 else None,
                        },
                        "finish_reason": "stop" if i == chunk_count - 1 else None,
                    }
                ],
                "chunk_index": i,
                "call_id": str(ctx.call_id),
            }

            await anyio.sleep(0.01)  # Small inter-chunk delay


class MockMiddleware:
    """Mock middleware for testing pipeline overhead."""

    def __init__(self, name: str, overhead_ms: float = 1.0):
        self.name = name
        self.overhead_ms = overhead_ms
        self.call_count = 0

    async def __call__(self, request, context, next_call):
        """Simulate middleware processing overhead."""
        self.call_count += 1

        # Pre-processing overhead
        await anyio.sleep(self.overhead_ms / 2000.0)  # Half before

        try:
            result = await next_call(request, context)
        except Exception:
            # Post-processing even on failure
            await anyio.sleep(self.overhead_ms / 2000.0)  # Half after
            raise

        # Post-processing overhead
        await anyio.sleep(self.overhead_ms / 2000.0)  # Half after

        return result


@pytest.fixture
def integration_request():
    """Realistic request for integration testing."""
    return ChatRequestModel(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Write a detailed explanation of quantum computing principles and applications in modern technology.",
            },
        ],
        model="gpt-4",
        temperature=0.7,
        max_tokens=2048,
    )


class TestFullPipelineLatency:
    """Test end-to-end pipeline latency with all middleware enabled."""

    @pytest.mark.anyio
    async def test_baseline_pipeline_latency(self, integration_request):
        """Establish baseline pipeline latency without middleware."""
        config = ExecutorConfig(
            queue_capacity=50,
            limit_requests=100,
            capacity_refresh_time=1.0,
            concurrency_limit=10,
        )

        executor = RateLimitedExecutor(config)
        service = MockFullPipelineService(latency_ms=50, response_size_bytes=4096)

        await executor.start()

        # Single call baseline
        ctx = CallContext.with_timeout(
            branch_id=uuid4(), timeout_s=10.0, capabilities={"net.out:api.mock.com"}
        )

        start_time = time.perf_counter()
        call = await executor.submit_call(service, integration_request, ctx)
        result = await call.wait_completion()
        end_time = time.perf_counter()

        await executor.stop()

        pipeline_latency = end_time - start_time

        # Validate baseline performance
        assert result is not None
        assert result["id"].startswith("response_")
        assert pipeline_latency < 0.3, f"Baseline latency too high: {pipeline_latency * 1000:.1f}ms"

        # Should be close to service latency + minimal overhead
        expected_latency = 0.05 + 0.02  # 50ms service + 20ms overhead allowance
        assert (
            pipeline_latency < expected_latency
        ), f"Pipeline overhead too high: {(pipeline_latency - 0.05) * 1000:.1f}ms"

    @pytest.mark.anyio
    async def test_pipeline_with_middleware_stack(self, integration_request):
        """Test pipeline latency with typical middleware stack."""
        config = ExecutorConfig(queue_capacity=50, limit_requests=100, concurrency_limit=10)

        executor = RateLimitedExecutor(config)
        service = MockFullPipelineService(latency_ms=100, response_size_bytes=2048)

        # Simulate middleware stack overhead
        policy_mw = MockMiddleware("policy", overhead_ms=2)
        metrics_mw = MockMiddleware("metrics", overhead_ms=1)
        hooks_mw = MockMiddleware("hooks", overhead_ms=3)
        retry_mw = MockMiddleware("retry", overhead_ms=1)

        await executor.start()

        ctx = CallContext.with_timeout(
            branch_id=uuid4(), timeout_s=10.0, capabilities={"net.out:api.mock.com"}
        )

        # Simulate middleware chain execution
        async def execute_with_middleware():
            start_time = time.perf_counter()

            # Simulate middleware chain (simplified)
            for mw in [policy_mw, metrics_mw, hooks_mw, retry_mw]:
                await anyio.sleep(mw.overhead_ms / 2000.0)  # Pre-processing

            call = await executor.submit_call(service, integration_request, ctx)
            result = await call.wait_completion()

            for mw in reversed([policy_mw, metrics_mw, hooks_mw, retry_mw]):
                await anyio.sleep(mw.overhead_ms / 2000.0)  # Post-processing
                mw.call_count += 1

            end_time = time.perf_counter()
            return result, end_time - start_time

        result, middleware_latency = await execute_with_middleware()

        await executor.stop()

        # Validate middleware performance impact
        assert result is not None
        assert (
            middleware_latency < 0.5
        ), f"Middleware latency too high: {middleware_latency * 1000:.1f}ms"

        # Middleware overhead should be reasonable
        expected_overhead = 0.007  # 7ms total middleware overhead
        service_time = 0.1  # 100ms service time
        assert middleware_latency < (
            service_time + expected_overhead + 0.05
        ), f"Total middleware overhead excessive: {(middleware_latency - service_time) * 1000:.1f}ms"

    @pytest.mark.anyio
    async def test_pipeline_streaming_latency(self, integration_request):
        """Test streaming pipeline latency with middleware."""
        config = ExecutorConfig(queue_capacity=20, concurrency_limit=5)
        executor = RateLimitedExecutor(config)

        service = MockFullPipelineService(latency_ms=80, response_size_bytes=8192)

        await executor.start()

        ctx = CallContext.with_timeout(
            branch_id=uuid4(), timeout_s=15.0, capabilities={"net.out:api.mock.com"}
        )

        # Measure streaming performance
        start_time = time.perf_counter()
        ttfb = None
        chunk_count = 0
        total_bytes = 0

        stream = executor.submit_stream(service, integration_request, ctx)
        async for chunk in stream:
            if ttfb is None:
                ttfb = time.perf_counter() - start_time

            chunk_count += 1
            chunk_content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            total_bytes += len(chunk_content)

        total_time = time.perf_counter() - start_time

        await executor.stop()

        # Validate streaming performance
        assert chunk_count == 20, f"Expected 20 chunks, got {chunk_count}"
        assert ttfb < 0.15, f"TTFB too high: {ttfb * 1000:.1f}ms"
        assert total_time < 1.0, f"Total streaming time too high: {total_time:.2f}s"
        assert total_bytes > 7000, f"Insufficient data streamed: {total_bytes} bytes"


class TestConcurrentRequestHandling:
    """Test concurrent request handling scalability."""

    @pytest.mark.anyio
    async def test_concurrent_calls_scalability(self, integration_request):
        """Test scalability with increasing concurrent request loads."""
        config = ExecutorConfig(
            queue_capacity=200,
            limit_requests=50,
            capacity_refresh_time=1.0,
            concurrency_limit=20,
        )

        executor = RateLimitedExecutor(config)
        service = MockFullPipelineService(latency_ms=100, response_size_bytes=1024)

        await executor.start()

        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20, 40]
        results = {}

        for concurrency in concurrency_levels:
            print(f"  Testing concurrency level: {concurrency}")

            # Submit concurrent requests
            start_time = time.perf_counter()
            calls = []

            for i in range(concurrency):
                ctx = CallContext.with_timeout(
                    branch_id=uuid4(),
                    timeout_s=20.0,
                    capabilities={"net.out:api.mock.com"},
                    test_batch=f"batch_{concurrency}",
                )
                call = await executor.submit_call(service, integration_request, ctx)
                calls.append(call)

            # Wait for all to complete
            completed_results = []
            for call in calls:
                result = await call.wait_completion()
                completed_results.append(result)

            end_time = time.perf_counter()

            # Calculate metrics
            total_time = end_time - start_time
            avg_latency = total_time  # All started together
            throughput = len(completed_results) / total_time

            results[concurrency] = {
                "total_time": total_time,
                "avg_latency": avg_latency,
                "throughput": throughput,
                "completed": len(completed_results),
            }

            # Brief pause between tests
            await anyio.sleep(0.5)

        await executor.stop()

        # Validate scalability
        assert len(results) == len(concurrency_levels)

        # Throughput should scale with concurrency (up to limits)
        throughput_1 = results[1]["throughput"]
        throughput_10 = results[10]["throughput"]
        throughput_20 = results[20]["throughput"]

        assert (
            throughput_10 > throughput_1 * 3
        ), f"Poor scaling: {throughput_10:.1f} vs {throughput_1:.1f}"
        assert (
            throughput_20 > throughput_1 * 5
        ), f"Poor scaling at 20: {throughput_20:.1f} vs {throughput_1:.1f}"

        # Higher concurrency should not degrade drastically
        if 40 in results:
            throughput_40 = results[40]["throughput"]
            assert (
                throughput_40 > throughput_20 * 0.7
            ), "Performance degradation at high concurrency"

    @pytest.mark.anyio
    async def test_mixed_workload_handling(self, integration_request):
        """Test handling mixed call and streaming workloads concurrently."""
        config = ExecutorConfig(queue_capacity=100, limit_requests=40, concurrency_limit=15)

        executor = RateLimitedExecutor(config)
        call_service = MockFullPipelineService(latency_ms=50, response_size_bytes=1024)
        stream_service = MockFullPipelineService(latency_ms=100, response_size_bytes=4096)

        await executor.start()

        # Launch mixed workload
        call_results = []
        stream_results = []

        start_time = time.perf_counter()

        async def run_calls():
            for i in range(15):
                ctx = CallContext.with_timeout(
                    branch_id=uuid4(),
                    timeout_s=10.0,
                    capabilities={"net.out:api.mock.com"},
                )
                call = await executor.submit_call(call_service, integration_request, ctx)
                result = await call.wait_completion()
                call_results.append(result)

                await anyio.sleep(0.1)  # Spread out submissions

        async def run_streams():
            for i in range(8):
                ctx = CallContext.with_timeout(
                    branch_id=uuid4(),
                    timeout_s=15.0,
                    capabilities={"net.out:api.mock.com"},
                )

                chunks = []
                stream = executor.submit_stream(stream_service, integration_request, ctx)
                async for chunk in stream:
                    chunks.append(chunk)

                stream_results.append(chunks)
                await anyio.sleep(0.2)  # Spread out submissions

        # Run both workloads concurrently
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_calls)
            tg.start_soon(run_streams)

        total_time = time.perf_counter() - start_time

        await executor.stop()

        # Validate mixed workload handling
        assert len(call_results) == 15, f"Expected 15 call results, got {len(call_results)}"
        assert len(stream_results) == 8, f"Expected 8 stream results, got {len(stream_results)}"

        # Should complete in reasonable time despite mixed load
        assert total_time < 15, f"Mixed workload took too long: {total_time:.2f}s"

        # Validate stream completeness
        total_stream_chunks = sum(len(chunks) for chunks in stream_results)
        expected_stream_chunks = 8 * 20  # 8 streams × 20 chunks each
        assert (
            total_stream_chunks == expected_stream_chunks
        ), f"Missing stream chunks: {total_stream_chunks}/{expected_stream_chunks}"

    @pytest.mark.anyio
    async def test_burst_traffic_handling(self, integration_request):
        """Test handling burst traffic patterns."""
        config = ExecutorConfig(
            queue_capacity=150,
            limit_requests=30,
            capacity_refresh_time=1.0,
            concurrency_limit=25,
        )

        executor = RateLimitedExecutor(config)
        service = MockFullPipelineService(latency_ms=80, response_size_bytes=2048)

        await executor.start()

        # Simulate burst traffic: quick bursts followed by quiet periods
        total_results = []

        for burst in range(3):  # 3 bursts
            print(f"  Running burst {burst + 1}/3")

            burst_start = time.perf_counter()
            burst_calls = []

            # Submit burst of requests rapidly
            for i in range(25):  # 25 requests per burst
                ctx = CallContext.with_timeout(
                    branch_id=uuid4(),
                    timeout_s=15.0,
                    capabilities={"net.out:api.mock.com"},
                    burst_id=f"burst_{burst}",
                )
                call = await executor.submit_call(service, integration_request, ctx)
                burst_calls.append(call)

            # Wait for burst to complete
            burst_results = []
            for call in burst_calls:
                result = await call.wait_completion()
                burst_results.append(result)

            burst_time = time.perf_counter() - burst_start
            total_results.extend(burst_results)

            print(
                f"    Burst {burst + 1} completed: {len(burst_results)} calls in {burst_time:.2f}s"
            )

            # Quiet period between bursts
            if burst < 2:  # Not after last burst
                await anyio.sleep(2.0)

        await executor.stop()

        # Validate burst handling
        assert len(total_results) == 75, f"Expected 75 results, got {len(total_results)}"

        # All requests should complete successfully
        successful_results = [r for r in total_results if r.get("id")]
        assert len(successful_results) == 75, "Some burst requests failed"


class TestResourceUtilization:
    """Test resource utilization under various load patterns."""

    @pytest.mark.anyio
    async def test_memory_utilization_under_load(self, integration_request):
        """Monitor memory utilization during sustained load."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        config = ExecutorConfig(queue_capacity=100, limit_requests=40, concurrency_limit=15)

        executor = RateLimitedExecutor(config)
        service = MockFullPipelineService(latency_ms=100, response_size_bytes=4096)

        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        tracemalloc.start()
        baseline_trace = tracemalloc.take_snapshot()

        await executor.start()

        # Sustained load test
        completed_results = []
        memory_samples = []

        for wave in range(5):  # 5 waves of requests
            wave_calls = []

            # Submit wave
            for i in range(20):
                ctx = CallContext.with_timeout(
                    branch_id=uuid4(),
                    timeout_s=15.0,
                    capabilities={"net.out:api.mock.com"},
                    wave=wave,
                )
                call = await executor.submit_call(service, integration_request, ctx)
                wave_calls.append(call)

            # Process wave
            wave_results = []
            for call in wave_calls:
                result = await call.wait_completion()
                wave_results.append(result)

            completed_results.extend(wave_results)

            # Sample memory
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory - baseline_memory)

            print(
                f"  Wave {wave + 1}: {len(wave_results)} completed, memory delta: {memory_samples[-1]:.1f}MB"
            )

            await anyio.sleep(0.5)  # Brief pause

        await executor.stop()

        # Final memory analysis
        final_trace = tracemalloc.take_snapshot()
        top_stats = final_trace.compare_to(baseline_trace, "lineno")
        trace_memory_mb = sum(stat.size for stat in top_stats) / 1024 / 1024

        tracemalloc.stop()

        # Validate memory utilization
        assert len(completed_results) == 100, f"Expected 100 results, got {len(completed_results)}"

        max_memory_delta = max(memory_samples)
        avg_memory_delta = sum(memory_samples) / len(memory_samples)

        # Memory usage should be bounded
        assert max_memory_delta < 50, f"Peak memory usage too high: {max_memory_delta:.1f}MB"
        assert avg_memory_delta < 30, f"Average memory usage too high: {avg_memory_delta:.1f}MB"
        assert (
            trace_memory_mb < 20
        ), f"Tracemalloc detected high memory usage: {trace_memory_mb:.1f}MB"

    @pytest.mark.anyio
    async def test_cpu_utilization_efficiency(self, integration_request):
        """Test CPU utilization efficiency under concurrent load."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        config = ExecutorConfig(queue_capacity=80, limit_requests=50, concurrency_limit=20)

        executor = RateLimitedExecutor(config)
        service = MockFullPipelineService(latency_ms=200, response_size_bytes=1024)  # I/O bound

        await executor.start()

        # Monitor CPU during concurrent execution
        cpu_samples = []
        start_time = time.perf_counter()

        async def cpu_monitor():
            """Monitor CPU usage during test."""
            for _ in range(20):  # Sample for 10 seconds
                cpu_percent = process.cpu_percent()
                cpu_samples.append(cpu_percent)
                await anyio.sleep(0.5)

        async def workload():
            """Execute concurrent workload."""
            calls = []

            # Submit concurrent requests
            for i in range(30):
                ctx = CallContext.with_timeout(
                    branch_id=uuid4(),
                    timeout_s=20.0,
                    capabilities={"net.out:api.mock.com"},
                )
                call = await executor.submit_call(service, integration_request, ctx)
                calls.append(call)

                if i % 5 == 0:  # Small pause every 5 requests
                    await anyio.sleep(0.1)

            # Wait for completion
            results = []
            for call in calls:
                result = await call.wait_completion()
                results.append(result)

            return results

        # Run concurrent monitoring and workload
        async with anyio.create_task_group() as tg:
            monitor_task = tg.start_soon(cpu_monitor)
            workload_task = tg.start_soon(workload)

            results = await workload_task

        total_time = time.perf_counter() - start_time

        await executor.stop()

        # Analyze CPU utilization
        if cpu_samples:
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
            max_cpu = max(cpu_samples)
        else:
            avg_cpu = max_cpu = 0

        # Validate efficient resource usage
        assert len(results) == 30, f"Expected 30 results, got {len(results)}"
        assert total_time < 25, f"Workload took too long: {total_time:.2f}s"

        # CPU usage should be reasonable for I/O bound workload
        assert avg_cpu < 50, f"Average CPU too high: {avg_cpu:.1f}%"
        assert max_cpu < 80, f"Peak CPU too high: {max_cpu:.1f}%"

    @pytest.mark.anyio
    async def test_resource_cleanup_after_load(self, integration_request):
        """Test that resources are properly cleaned up after high load."""
        import gc

        config = ExecutorConfig(queue_capacity=100, limit_requests=60, concurrency_limit=25)

        executor = RateLimitedExecutor(config)
        service = MockFullPipelineService(latency_ms=50, response_size_bytes=2048)

        # Baseline measurement
        gc.collect()
        baseline_objects = len(gc.get_objects())

        await executor.start()

        # High load phase
        all_results = []
        for batch in range(10):  # 10 batches of 15 = 150 total requests
            batch_calls = []

            for i in range(15):
                ctx = CallContext.with_timeout(
                    branch_id=uuid4(),
                    timeout_s=10.0,
                    capabilities={"net.out:api.mock.com"},
                )
                call = await executor.submit_call(service, integration_request, ctx)
                batch_calls.append(call)

            # Complete batch
            batch_results = []
            for call in batch_calls:
                result = await call.wait_completion()
                batch_results.append(result)

            all_results.extend(batch_results)

            # Brief pause between batches
            await anyio.sleep(0.1)

        # Stop executor and force cleanup
        await executor.stop()

        # Force garbage collection
        for _ in range(3):
            gc.collect()

        # Measure cleanup effectiveness
        final_objects = len(gc.get_objects())
        object_growth = final_objects - baseline_objects

        # Validate cleanup
        assert len(all_results) == 150, f"Expected 150 results, got {len(all_results)}"

        # Object growth should be bounded (some growth expected from test infrastructure)
        assert object_growth < 1000, f"Too many objects not cleaned up: {object_growth}"

        # Executor stats should show completion
        stats = executor.stats
        assert stats["active_calls"] == 0, f"Active calls not cleaned up: {stats['active_calls']}"
        assert (
            stats["calls_completed"] == 150
        ), f"Incorrect completion count: {stats['calls_completed']}"
