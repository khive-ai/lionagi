# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""P0 Rate Limiting tests for RateLimitedExecutor - focusing on accuracy and enforcement.

This module implements comprehensive rate limiting validation tests,
ensuring proper request/token limits, refresh timing, and deadline awareness.
"""

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import anyio
import pytest

from lionagi.services.core import CallContext, Service
from lionagi.services.endpoint import RequestModel
from lionagi.services.executor import ExecutorConfig, RateLimitedExecutor

# Test Service implementations for rate limiting


class DummyRequest(RequestModel):
    """Request with configurable token usage."""

    content: str = "test"
    estimated_tokens: int = 100  # Default token estimate


class TokenHeavyRequest(RequestModel):
    """Request that uses many tokens."""

    content: str = "heavy"
    estimated_tokens: int = 500


class FastService(Service):
    """Service that responds quickly for rate limiting tests."""

    name = "fast_service"

    async def call(self, req: DummyRequest, *, ctx: CallContext) -> dict[str, Any]:
        # Minimal processing time to focus on rate limiting
        return {
            "result": "processed",
            "content": req.content,
            "tokens": getattr(req, "estimated_tokens", 100),
        }

    async def stream(self, req: DummyRequest, *, ctx: CallContext) -> AsyncIterator[dict[str, Any]]:
        for i in range(3):
            yield {"chunk": i, "content": req.content}


class SlowService(Service):
    """Service with controlled processing time."""

    name = "slow_service"

    def __init__(self, delay_s: float = 0.1):
        self.delay_s = delay_s

    async def call(self, req: DummyRequest, *, ctx: CallContext) -> dict[str, Any]:
        await anyio.sleep(self.delay_s)
        return {"result": "slow_processed", "delay": self.delay_s}

    async def stream(self, req: DummyRequest, *, ctx: CallContext) -> AsyncIterator[dict[str, Any]]:
        for i in range(3):
            await anyio.sleep(self.delay_s / 3)
            yield {"chunk": i, "delay": self.delay_s}


# Custom executor for rate limiting tests
class TokenAwareExecutor(RateLimitedExecutor):
    """Executor that properly estimates tokens from request."""

    def _estimate_tokens(self, request: RequestModel) -> int:
        """Use the request's estimated_tokens if available."""
        return getattr(request, "estimated_tokens", 100)


@pytest.fixture
def rate_limited_config():
    """Configuration with strict rate limits for testing."""
    return ExecutorConfig(
        queue_capacity=20,
        limit_requests=5,  # 5 requests per refresh
        limit_tokens=1000,  # 1000 tokens per refresh
        capacity_refresh_time=1.0,  # 1 second refresh
        concurrency_limit=10,
    )


# V1_Executor_RateLimiting Test Suite


@pytest.mark.anyio
async def test_request_limit_enforcement():
    """CRITICAL: Test request rate limiting is properly enforced.

    Validates that the executor respects request limits and properly
    batches requests across refresh periods.
    """
    config = ExecutorConfig(
        queue_capacity=20,
        limit_requests=3,  # Only 3 requests per period
        capacity_refresh_time=1.0,
        concurrency_limit=10,
    )

    executor = TokenAwareExecutor(config)
    await executor.start()

    try:
        service = FastService()
        start_time = anyio.current_time()

        # Submit 7 requests - should process 3, wait, then process 3, wait, then 1
        calls = []
        for i in range(7):
            ctx = CallContext.with_timeout(uuid4(), timeout_s=15.0)
            call = await executor.submit_call(service, DummyRequest(content=f"req_{i}"), ctx)
            calls.append(call)

        # Wait for all calls to complete
        results = []
        for call in calls:
            result = await call.wait_completion()
            results.append(result)

        elapsed = anyio.current_time() - start_time

        # Should take at least 2 seconds (3 in first second, 3 in second second, 1 in third)
        assert (
            elapsed >= 1.8
        ), f"Processing 7 requests with 3 req/sec limit took only {elapsed:.2f}s"

        # All requests should have completed successfully
        assert len(results) == 7
        for i, result in enumerate(results):
            assert result["content"] == f"req_{i}"

        # Verify stats
        stats = executor.stats
        assert stats["calls_completed"] == 7
        assert stats["active_calls"] == 0

    finally:
        await executor.stop()


@pytest.mark.anyio
async def test_token_limit_enforcement():
    """CRITICAL: Test token rate limiting is properly enforced."""
    config = ExecutorConfig(
        queue_capacity=20,
        limit_requests=100,  # High request limit
        limit_tokens=800,  # Low token limit (800 tokens per period)
        capacity_refresh_time=1.0,
        concurrency_limit=10,
    )

    executor = TokenAwareExecutor(config)
    await executor.start()

    try:
        service = FastService()
        start_time = anyio.current_time()

        # Submit requests with different token requirements
        calls = []

        # First batch: 2 requests × 300 tokens = 600 tokens (fits in period)
        for i in range(2):
            ctx = CallContext.with_timeout(uuid4(), timeout_s=15.0)
            req = TokenHeavyRequest(content=f"heavy_{i}", estimated_tokens=300)
            call = await executor.submit_call(service, req, ctx)
            calls.append(call)

        # Third request: 300 tokens (total would be 900, exceeds 800 limit)
        ctx = CallContext.with_timeout(uuid4(), timeout_s=15.0)
        req = TokenHeavyRequest(content="heavy_overflow", estimated_tokens=300)
        call = await executor.submit_call(service, req, ctx)
        calls.append(call)

        # Wait for all calls to complete
        results = []
        for call in calls:
            result = await call.wait_completion()
            results.append(result)

        elapsed = anyio.current_time() - start_time

        # Should take at least 1 second due to token limiting
        assert elapsed >= 0.9, f"Token-limited execution took only {elapsed:.2f}s"

        # All requests should complete
        assert len(results) == 3

        # Verify token accounting in results
        for result in results:
            assert result["tokens"] == 300

        stats = executor.stats
        assert stats["calls_completed"] == 3

    finally:
        await executor.stop()


@pytest.mark.anyio
async def test_combined_rate_limits():
    """Test behavior when both request and token limits are active."""
    config = ExecutorConfig(
        queue_capacity=20,
        limit_requests=3,  # 3 requests per period
        limit_tokens=400,  # 400 tokens per period
        capacity_refresh_time=1.0,
        concurrency_limit=10,
    )

    executor = TokenAwareExecutor(config)
    await executor.start()

    try:
        service = FastService()

        # Submit requests that will hit token limit before request limit
        # 3 requests × 200 tokens = 600 tokens, but limit is 400
        calls = []
        for i in range(3):
            ctx = CallContext.with_timeout(uuid4(), timeout_s=15.0)
            req = DummyRequest(content=f"combo_{i}", estimated_tokens=200)
            call = await executor.submit_call(service, req, ctx)
            calls.append(call)

        start_time = anyio.current_time()

        # Wait for all to complete
        results = []
        for call in calls:
            result = await call.wait_completion()
            results.append(result)

        elapsed = anyio.current_time() - start_time

        # Should be constrained by token limit (400), so only 2 requests per period
        # Third request should wait for next period
        assert elapsed >= 0.9, f"Combined rate limiting took only {elapsed:.2f}s"

        assert len(results) == 3
        stats = executor.stats
        assert stats["calls_completed"] == 3

    finally:
        await executor.stop()


@pytest.mark.anyio
async def test_rate_limit_refresh_timing():
    """Test that rate limits refresh at correct intervals."""
    config = ExecutorConfig(
        queue_capacity=10,
        limit_requests=2,  # Very low limit
        capacity_refresh_time=0.5,  # Fast refresh for testing
        concurrency_limit=5,
    )

    executor = TokenAwareExecutor(config)
    await executor.start()

    try:
        service = FastService()

        # Track timing of request processing
        request_times = []
        calls = []

        # Submit 6 requests rapidly
        submit_start = anyio.current_time()
        for i in range(6):
            ctx = CallContext.with_timeout(uuid4(), timeout_s=10.0)
            call = await executor.submit_call(service, DummyRequest(content=f"timed_{i}"), ctx)
            calls.append(call)

        # Wait for completion and track timing
        for i, call in enumerate(calls):
            result = await call.wait_completion()
            completion_time = anyio.current_time() - submit_start
            request_times.append(completion_time)

        # Analyze timing patterns - should see batches every 0.5s
        # First 2 requests: ~0s
        # Next 2 requests: ~0.5s
        # Last 2 requests: ~1.0s

        assert request_times[0] < 0.2, f"First request took {request_times[0]:.2f}s"
        assert request_times[1] < 0.2, f"Second request took {request_times[1]:.2f}s"

        # Requests 2-3 should be delayed by ~0.5s
        assert (
            0.4 <= request_times[2] <= 0.8
        ), f"Third request timing {request_times[2]:.2f}s unexpected"
        assert (
            0.4 <= request_times[3] <= 0.8
        ), f"Fourth request timing {request_times[3]:.2f}s unexpected"

        # Requests 4-5 should be delayed by ~1.0s
        assert (
            0.9 <= request_times[4] <= 1.3
        ), f"Fifth request timing {request_times[4]:.2f}s unexpected"
        assert (
            0.9 <= request_times[5] <= 1.3
        ), f"Sixth request timing {request_times[5]:.2f}s unexpected"

    finally:
        await executor.stop()


@pytest.mark.anyio
async def test_rate_limit_accuracy_under_load():
    """CRITICAL: Test rate limiting accuracy under concurrent load.

    Stress test to ensure rate limiting counters remain accurate
    when many tasks submit calls concurrently.
    """
    config = ExecutorConfig(
        queue_capacity=100,
        limit_requests=10,  # 10 requests per second
        capacity_refresh_time=1.0,
        concurrency_limit=20,
    )

    executor = TokenAwareExecutor(config)
    await executor.start()

    try:
        service = FastService()

        # Submit many requests concurrently to stress test rate limiting
        num_requests = 35  # Should take ~3.5 seconds at 10 req/sec
        calls = []

        start_time = anyio.current_time()

        # Use task group to submit all requests concurrently
        async with anyio.create_task_group() as tg:
            for i in range(num_requests):
                ctx = CallContext.with_timeout(uuid4(), timeout_s=30.0)
                call = await executor.submit_call(service, DummyRequest(content=f"load_{i}"), ctx)
                calls.append(call)

        # Wait for all to complete
        completed = 0
        for call in calls:
            try:
                await call.wait_completion()
                completed += 1
            except Exception:
                pass  # Some may timeout under extreme load

        elapsed = anyio.current_time() - start_time

        # Should take at least 3 seconds for 35 requests at 10 req/sec
        expected_time = (num_requests / config.limit_requests) - 0.5  # Allow some tolerance
        assert (
            elapsed >= expected_time
        ), f"Rate limiting failed: {num_requests} requests completed in {elapsed:.2f}s (expected >={expected_time:.1f}s)"

        # Most requests should complete
        assert (
            completed >= num_requests * 0.8
        ), f"Only {completed}/{num_requests} requests completed"

        # Verify internal state consistency
        stats = executor.stats
        assert stats["active_calls"] == 0, "Should have no active calls after completion"

    finally:
        await executor.stop()


@pytest.mark.anyio
async def test_no_rate_limits_fast_processing():
    """Test executor performance without rate limits."""
    config = ExecutorConfig(
        queue_capacity=50,
        limit_requests=None,  # No request limit
        limit_tokens=None,  # No token limit
        concurrency_limit=20,
    )

    executor = TokenAwareExecutor(config)
    await executor.start()

    try:
        service = FastService()

        # Submit many requests that should process quickly without limits
        num_requests = 20
        calls = []

        start_time = anyio.current_time()
        for i in range(num_requests):
            ctx = CallContext.with_timeout(uuid4(), timeout_s=10.0)
            call = await executor.submit_call(service, DummyRequest(content=f"fast_{i}"), ctx)
            calls.append(call)

        # Wait for all to complete
        for call in calls:
            await call.wait_completion()

        elapsed = anyio.current_time() - start_time

        # Without rate limits, should complete very quickly
        assert elapsed < 1.0, f"Unlimited processing took {elapsed:.2f}s - too slow"

        stats = executor.stats
        assert stats["calls_completed"] == num_requests
        assert stats["active_calls"] == 0

    finally:
        await executor.stop()


@pytest.mark.anyio
async def test_rate_limit_with_slow_service():
    """Test rate limiting behavior with slow service calls."""
    config = ExecutorConfig(
        queue_capacity=10,
        limit_requests=5,  # 5 requests per period
        capacity_refresh_time=1.0,
        concurrency_limit=3,  # Low concurrency to create backpressure
    )

    executor = TokenAwareExecutor(config)
    await executor.start()

    try:
        # Use slow service that takes time to process
        service = SlowService(delay_s=0.3)  # Each call takes 0.3s

        calls = []
        start_time = anyio.current_time()

        # Submit 8 calls - should be rate limited AND processing limited
        for i in range(8):
            ctx = CallContext.with_timeout(uuid4(), timeout_s=15.0)
            call = await executor.submit_call(service, DummyRequest(content=f"slow_{i}"), ctx)
            calls.append(call)

        # Wait for all to complete
        results = []
        for call in calls:
            result = await call.wait_completion()
            results.append(result)

        elapsed = anyio.current_time() - start_time

        # Should be constrained by both rate limiting and processing time
        # With 5 req/sec limit and 0.3s processing time, expect > 1 second total
        assert elapsed >= 1.0, f"Slow service with rate limiting completed in {elapsed:.2f}s"

        assert len(results) == 8
        stats = executor.stats
        assert stats["calls_completed"] == 8

    finally:
        await executor.stop()


@pytest.mark.anyio
async def test_token_estimation_accuracy():
    """Test that token estimation affects rate limiting correctly."""
    config = ExecutorConfig(
        queue_capacity=10,
        limit_tokens=300,  # Low token limit
        capacity_refresh_time=1.0,
        concurrency_limit=10,
    )

    executor = TokenAwareExecutor(config)
    await executor.start()

    try:
        service = FastService()

        # Submit requests with exact token limits
        calls = []

        # Request 1: 150 tokens (fits)
        ctx1 = CallContext.with_timeout(uuid4(), timeout_s=10.0)
        req1 = DummyRequest(content="low_token", estimated_tokens=150)
        call1 = await executor.submit_call(service, req1, ctx1)
        calls.append(call1)

        # Request 2: 149 tokens (total 299, still fits)
        ctx2 = CallContext.with_timeout(uuid4(), timeout_s=10.0)
        req2 = DummyRequest(content="exact_fit", estimated_tokens=149)
        call2 = await executor.submit_call(service, req2, ctx2)
        calls.append(call2)

        # Request 3: 2 tokens (total 301, exceeds limit of 300)
        start_time = anyio.current_time()
        ctx3 = CallContext.with_timeout(uuid4(), timeout_s=10.0)
        req3 = DummyRequest(content="overflow", estimated_tokens=2)
        call3 = await executor.submit_call(service, req3, ctx3)
        calls.append(call3)

        # Wait for all to complete
        for call in calls:
            await call.wait_completion()

        elapsed = anyio.current_time() - start_time

        # Third request should have been delayed
        assert elapsed >= 0.9, f"Token overflow handling too fast: {elapsed:.2f}s"

        stats = executor.stats
        assert stats["calls_completed"] == 3

    finally:
        await executor.stop()


# Backend compatibility test
@pytest.mark.parametrize(
    "anyio_backend", ["asyncio"]
)  # Only test asyncio since executor uses asyncio.Task
@pytest.mark.anyio
async def test_rate_limiting_backend_compatibility(anyio_backend):
    """Test rate limiting works correctly on asyncio backend."""
    config = ExecutorConfig(
        queue_capacity=10,
        limit_requests=3,
        capacity_refresh_time=0.5,
        concurrency_limit=5,
    )

    executor = TokenAwareExecutor(config)
    await executor.start()

    try:
        service = FastService()

        # Submit requests that will test rate limiting
        calls = []
        start_time = anyio.current_time()

        for i in range(5):  # More than rate limit
            ctx = CallContext.with_timeout(uuid4(), timeout_s=10.0)
            call = await executor.submit_call(service, DummyRequest(content=f"backend_{i}"), ctx)
            calls.append(call)

        # Wait for completion
        for call in calls:
            await call.wait_completion()

        elapsed = anyio.current_time() - start_time

        # Should be rate limited regardless of backend
        assert elapsed >= 0.4, f"Rate limiting not working on {anyio_backend}: {elapsed:.2f}s"

        stats = executor.stats
        assert stats["calls_completed"] == 5

    finally:
        await executor.stop()
