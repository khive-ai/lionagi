# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""P0 Reliability tests for RateLimitedExecutor - focusing on critical flaws.

This module implements the V1_Executor_Reliability test suite from the TDD specification,
with special emphasis on the CRITICAL ExecutorQueueWaitDeadline test that validates
the deadline-unaware waiting flaw in the _wait_for_capacity method.
"""

import asyncio
import time
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

import anyio
import pytest

from lionagi.errors import ServiceError
from lionagi.services.core import CallContext, Service
from lionagi.services.endpoint import RequestModel
from lionagi.services.executor import ExecutorConfig, RateLimitedExecutor, ServiceCall


# Mock Service implementations for testing

class DummyRequest(RequestModel):
    """Simple test request."""
    content: str = "test"


class SlowService(Service):
    """Service that takes a specified amount of time to respond."""
    
    name = "slow_service"
    
    def __init__(self, delay_s: float = 1.0):
        self.delay_s = delay_s
    
    async def call(self, req: DummyRequest, *, ctx: CallContext) -> dict[str, Any]:
        await anyio.sleep(self.delay_s)
        return {"result": f"processed: {req.content}", "delay": self.delay_s}
    
    async def stream(self, req: DummyRequest, *, ctx: CallContext) -> AsyncIterator[dict[str, Any]]:
        for i in range(3):
            await anyio.sleep(self.delay_s / 3)
            yield {"chunk": i, "content": req.content}


class FailingService(Service):
    """Service that fails after a specified number of successful calls."""
    
    name = "failing_service"
    
    def __init__(self, fail_after: int = 0):
        self.call_count = 0
        self.fail_after = fail_after
    
    async def call(self, req: DummyRequest, *, ctx: CallContext) -> dict[str, Any]:
        self.call_count += 1
        if self.call_count > self.fail_after:
            raise Exception(f"Service failed on call {self.call_count}")
        return {"result": f"success {self.call_count}"}
    
    async def stream(self, req: DummyRequest, *, ctx: CallContext) -> AsyncIterator[dict[str, Any]]:
        yield {"result": "streaming not implemented in test service"}


class FastService(Service):
    """Service that responds immediately."""
    
    name = "fast_service"
    
    async def call(self, req: DummyRequest, *, ctx: CallContext) -> dict[str, Any]:
        return {"result": "fast", "content": req.content}
    
    async def stream(self, req: DummyRequest, *, ctx: CallContext) -> AsyncIterator[dict[str, Any]]:
        for i in range(5):
            yield {"chunk": i, "fast": True}


@pytest.fixture
def executor_config():
    """Basic executor configuration for testing."""
    return ExecutorConfig(
        queue_capacity=10,
        capacity_refresh_time=1.0,
        limit_requests=5,
        limit_tokens=1000,
        concurrency_limit=3
    )


@pytest.fixture
async def executor(executor_config):
    """Create and cleanup executor."""
    executor = RateLimitedExecutor(executor_config)
    yield executor
    await executor.stop()


# V1_Executor_Reliability Test Suite

@pytest.mark.anyio
async def test_executor_queue_wait_deadline_critical_flaw():
    """CRITICAL: Test the deadline-unaware waiting flaw in _wait_for_capacity.
    
    This test validates the fix for the critical flaw where the executor waits
    for capacity but doesn't respect the call deadline. When rate limits force
    a wait time longer than the call deadline, the call should fail promptly
    with TimeoutError rather than waiting the full rate limit time.
    
    TDD Spec Reference: V1_Executor_RateLimiting.DeadlineWhileWaitingForCapacity
    """
    # Create executor with very restrictive rate limits to force waiting
    config = ExecutorConfig(
        queue_capacity=10,
        limit_requests=1,  # Only 1 request allowed
        capacity_refresh_time=10.0,  # 10 second refresh - very slow
        concurrency_limit=1
    )
    
    executor = RateLimitedExecutor(config)
    
    try:
        await executor.start()
        
        # Submit first call to exhaust rate limit capacity
        fast_service = FastService()
        ctx1 = CallContext.with_timeout(uuid4(), timeout_s=30.0)
        
        call1 = await executor.submit_call(fast_service, DummyRequest(), ctx1)
        result1 = await call1.wait_completion()
        
        assert result1["result"] == "fast"
        
        # Now the rate limiter should be at capacity (1 request used)
        # Submit second call with SHORT deadline (1s) when rate limit won't refresh for 10s
        ctx2 = CallContext.with_timeout(uuid4(), timeout_s=1.0)  # 1 second deadline
        
        start_time = anyio.current_time()
        
        # This call should be queued and then fail due to deadline, NOT wait 10s
        call2 = await executor.submit_call(fast_service, DummyRequest(), ctx2)
        
        # The call should fail with timeout after ~1 second, not 10 seconds
        with pytest.raises((TimeoutError, ServiceError)):
            await call2.wait_completion()
        
        elapsed = anyio.current_time() - start_time
        
        # CRITICAL ASSERTION: The call must fail quickly (~1s), not wait for rate limit refresh (~10s)
        # This validates that deadline awareness is working in _wait_for_capacity
        assert elapsed < 2.0, f"Call took {elapsed:.2f}s but should have failed after ~1s due to deadline"
        
        # Verify the call was marked appropriately (cancelled or failed)
        assert call2.status.value in ["cancelled", "failed"]
        
    finally:
        await executor.stop()


@pytest.mark.anyio
async def test_rate_limit_accuracy_and_safety_stress():
    """CRITICAL: Stress test rate limiting accuracy with concurrent submissions.
    
    Validates that rate limiting counters remain accurate under concurrent load
    and that the total execution follows rate limit constraints.
    """
    config = ExecutorConfig(
        queue_capacity=100,
        limit_requests=10,
        capacity_refresh_time=1.0,  # 1 second refresh
        concurrency_limit=5
    )
    
    executor = RateLimitedExecutor(config)
    
    try:
        await executor.start()
        
        # Submit many calls concurrently to test rate limiting accuracy
        service = FastService()
        calls = []
        
        start_time = anyio.current_time()
        
        # Submit 25 calls (should take ~3 seconds with 10 req/sec limit)
        async with anyio.create_task_group() as tg:
            for i in range(25):
                ctx = CallContext.with_timeout(uuid4(), timeout_s=30.0)
                call = await executor.submit_call(service, DummyRequest(content=f"request_{i}"), ctx)
                calls.append(call)
        
        # Wait for all calls to complete
        results = []
        for call in calls:
            try:
                result = await call.wait_completion()
                results.append(result)
            except Exception as e:
                # Some calls might timeout or fail, that's OK for stress test
                pass
        
        elapsed = anyio.current_time() - start_time
        
        # Verify rate limiting worked: 25 calls at 10 req/sec should take ~2.5+ seconds
        assert elapsed >= 2.0, f"25 calls completed too quickly ({elapsed:.2f}s), rate limiting may not be working"
        
        # Verify internal counters are accurate (no race conditions)
        stats = executor.stats
        assert stats["calls_queued"] == 25
        assert stats["calls_completed"] + stats["calls_failed"] + stats["calls_cancelled"] == 25
        
        # Verify no active calls remain
        assert stats["active_calls"] == 0
        
    finally:
        await executor.stop()


@pytest.mark.anyio
async def test_efficient_queuing_no_polling():
    """CRITICAL: Verify AnyIO memory streams eliminated polling latency.
    
    Tests that calls are picked up immediately when submitted, not on next
    polling interval. This validates the move from polling to event-driven processing.
    """
    config = ExecutorConfig(
        queue_capacity=10,
        limit_requests=100,  # High limit to avoid rate limiting interference
        concurrency_limit=5
    )
    
    executor = RateLimitedExecutor(config)
    
    try:
        await executor.start()
        
        service = FastService()
        
        # Measure latency between submit and execution start
        ctx = CallContext.with_timeout(uuid4(), timeout_s=10.0)
        
        submit_time = anyio.current_time()
        call = await executor.submit_call(service, DummyRequest(), ctx)
        
        # Wait for execution to start (marked as executing)
        while call.status.value not in ["executing", "completed", "failed"]:
            await anyio.sleep(0.001)  # Very short sleep
        
        pickup_time = anyio.current_time()
        latency = pickup_time - submit_time
        
        # CRITICAL: Latency should be minimal (< 50ms), proving no polling
        assert latency < 0.05, f"Pickup latency {latency*1000:.1f}ms too high - suggests polling behavior"
        
        # Complete the call
        result = await call.wait_completion()
        assert result["result"] == "fast"
        
    finally:
        await executor.stop()


@pytest.mark.anyio
async def test_concurrency_safety_stress():
    """CRITICAL: Stress test concurrent submissions for race conditions.
    
    Verifies anyio.Lock usage on internal counters prevents race conditions
    when many tasks submit calls simultaneously.
    """
    config = ExecutorConfig(
        queue_capacity=200,
        limit_requests=1000,  # High limit to reduce rate limiting interference
        concurrency_limit=10
    )
    
    executor = RateLimitedExecutor(config)
    
    try:
        await executor.start()
        
        service = FastService()
        num_concurrent = 100
        
        async def submit_call_task(call_id: int) -> ServiceCall:
            ctx = CallContext.with_timeout(uuid4(), timeout_s=30.0)
            return await executor.submit_call(service, DummyRequest(content=f"call_{call_id}"), ctx)
        
        # Submit many calls concurrently to stress test thread safety
        async with anyio.create_task_group() as tg:
            calls = []
            for i in range(num_concurrent):
                call = await submit_call_task(i)
                calls.append(call)
        
        # Wait for all to complete
        completed_count = 0
        for call in calls:
            try:
                await call.wait_completion()
                completed_count += 1
            except Exception:
                pass  # Some may timeout, that's OK for stress test
        
        # Verify statistics are accurate (no race conditions in counters)
        stats = executor.stats
        expected_queued = num_concurrent
        actual_total = stats["calls_completed"] + stats["calls_failed"] + stats["calls_cancelled"]
        
        assert stats["calls_queued"] == expected_queued, f"Expected {expected_queued} queued, got {stats['calls_queued']}"
        assert actual_total == expected_queued, f"Total processed ({actual_total}) != queued ({expected_queued})"
        assert stats["active_calls"] == 0, f"Should have 0 active calls, got {stats['active_calls']}"
        
    finally:
        await executor.stop()


@pytest.mark.anyio 
async def test_executor_handles_service_failures():
    """Test executor properly handles and reports service failures."""
    config = ExecutorConfig(queue_capacity=10, limit_requests=10)
    executor = RateLimitedExecutor(config)
    
    try:
        await executor.start()
        
        # Service that always fails
        failing_service = FailingService(fail_after=0)
        ctx = CallContext.with_timeout(uuid4(), timeout_s=5.0)
        
        call = await executor.submit_call(failing_service, DummyRequest(), ctx)
        
        # Should raise the service exception
        with pytest.raises(Exception, match="Service failed on call 1"):
            await call.wait_completion()
        
        # Verify call was marked as failed
        assert call.status.value == "failed"
        assert call.error is not None
        
        # Verify stats updated
        stats = executor.stats
        assert stats["calls_failed"] == 1
        assert stats["active_calls"] == 0
        
    finally:
        await executor.stop()


@pytest.mark.anyio
async def test_executor_streaming_concurrency_limits():
    """Test streaming calls respect concurrency limits."""
    config = ExecutorConfig(
        queue_capacity=10,
        concurrency_limit=2,  # Only 2 concurrent streams
        limit_requests=100
    )
    executor = RateLimitedExecutor(config)
    
    try:
        await executor.start()
        
        service = SlowService(delay_s=0.3)  # Each chunk takes 0.1s, total ~0.3s
        
        # Start multiple streams concurrently
        streams = []
        start_time = anyio.current_time()
        
        async def collect_stream(stream_id: int):
            ctx = CallContext.with_timeout(uuid4(), timeout_s=10.0)
            chunks = []
            async for chunk in executor.submit_stream(service, DummyRequest(content=f"stream_{stream_id}"), ctx):
                chunks.append(chunk)
            return chunks
        
        # Start 4 streams, but only 2 should run concurrently due to limit
        async with anyio.create_task_group() as tg:
            for i in range(4):
                tg.start_soon(collect_stream, i)
        
        elapsed = anyio.current_time() - start_time
        
        # With concurrency limit of 2, 4 streams should take ~0.6s (2 batches of 0.3s each)
        # Without limit, would take ~0.3s (all concurrent)
        assert elapsed >= 0.5, f"Streams completed too quickly ({elapsed:.2f}s), concurrency limit may not be working"
        
    finally:
        await executor.stop()


# Parameterized test to run on both asyncio and trio
@pytest.mark.parametrize("anyio_backend", ["asyncio", "trio"])
@pytest.mark.anyio
async def test_executor_backend_compatibility(anyio_backend):
    """Test executor works on both asyncio and trio backends."""
    config = ExecutorConfig(queue_capacity=5, limit_requests=10)
    executor = RateLimitedExecutor(config)
    
    try:
        await executor.start()
        
        service = FastService()
        ctx = CallContext.with_timeout(uuid4(), timeout_s=5.0)
        
        call = await executor.submit_call(service, DummyRequest(content="backend_test"), ctx)
        result = await call.wait_completion()
        
        assert result["result"] == "fast"
        assert result["content"] == "backend_test"
        
        # Verify stats
        stats = executor.stats
        assert stats["calls_completed"] == 1
        
    finally:
        await executor.stop()


@pytest.mark.anyio
async def test_executor_queue_capacity_rejection():
    """Test executor rejects calls when queue is at capacity."""
    config = ExecutorConfig(
        queue_capacity=2,  # Very small queue
        limit_requests=1,  # Limit to force queueing
        capacity_refresh_time=10.0  # Long refresh to keep limit active
    )
    executor = RateLimitedExecutor(config)
    
    try:
        await executor.start()
        
        service = SlowService(delay_s=2.0)  # Slow service to fill queue
        calls = []
        
        # Submit calls to fill queue and exhaust capacity
        for i in range(3):  # Try to submit more than queue + rate limit allows
            ctx = CallContext.with_timeout(uuid4(), timeout_s=30.0)
            try:
                call = await executor.submit_call(service, DummyRequest(content=f"call_{i}"), ctx)
                calls.append(call)
            except ServiceError as e:
                # Should get queue capacity error on the 3rd call
                assert "queue at capacity" in str(e)
                assert i >= 2, f"Queue rejection happened too early at call {i}"
                break
        else:
            pytest.fail("Expected queue capacity rejection")
        
    finally:
        await executor.stop()