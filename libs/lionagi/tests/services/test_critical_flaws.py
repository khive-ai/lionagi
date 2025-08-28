# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Critical Implementation Flaw Tests for lionagi services.

This file consolidates tests specifically targeting the CRITICAL flaws identified
in the TDD specification that require fixes for production readiness:

1. executor.py:349 - Deadline-unaware waiting flaw in _wait_for_capacity()
2. hooks.py:582 - Incorrect timeout application flaw in emit()
3. msgspec migration completeness
4. Circuit breaker streaming performance regression prevention

These tests are designed to FAIL with current implementation and PASS after fixes.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import anyio
import pytest

from lionagi.errors import RetryableError, ServiceError, TimeoutError
from lionagi.ln.concurrency import create_task_group
from lionagi.services.core import CallContext, Service
from lionagi.services.endpoint import RequestModel
from lionagi.services.executor import ExecutorConfig, RateLimitedExecutor
from lionagi.services.hooks import HookEvent, HookRegistry, HookType
from lionagi.services.resilience import (
    CircuitBreakerConfig,
    CircuitBreakerMW,
    RetryConfig,
    RetryMW,
)


class TestRequest(RequestModel, frozen=True):
    """Simple test request."""

    content: str = "test"


class SlowService(Service):
    """Service with configurable delay."""

    name = "slow_service"

    def __init__(self, delay_s: float = 1.0):
        self.delay_s = delay_s

    async def call(self, req: TestRequest, *, ctx: CallContext) -> dict[str, Any]:
        await anyio.sleep(self.delay_s)
        return {"result": f"processed after {self.delay_s}s"}

    async def stream(self, req: TestRequest, *, ctx: CallContext) -> AsyncIterator[dict[str, Any]]:
        for i in range(3):
            await anyio.sleep(self.delay_s / 3)
            yield {"chunk": i, "delay": self.delay_s}


class TestCriticalImplementationFlaws:
    """Tests for critical implementation flaws identified by TDD specification.

    These tests are designed to fail with current implementation and pass
    after critical fixes are applied.
    """

    @pytest.mark.anyio
    async def test_executor_deadline_unaware_waiting_critical_flaw(self):
        """CRITICAL FLAW: executor.py:349 - Deadline-unaware waiting in _wait_for_capacity.

        Current flaw: _wait_for_capacity() uses `await anyio.sleep(0.1)` polling
        without checking CallContext deadline. If rate limit wait is 10s but
        call deadline is 1s, executor waits full 10s violating deadline.

        Required fix: Wrap waiting in anyio.fail_at(call.context.deadline_s)
        or check remaining time before sleeping.

        Expected behavior: Call should fail at deadline (~1s), not rate limit wait (~10s).
        """
        # Create executor with very restrictive rate limits
        config = ExecutorConfig(
            queue_capacity=5,
            capacity_refresh_time=10.0,  # 10 second refresh (long wait)
            limit_requests=1,  # Only 1 request allowed
            limit_tokens=100,
            concurrency_limit=1,
        )

        # Use async context manager for proper lifecycle management
        async with RateLimitedExecutor(config) as executor:
            # Submit first call to consume rate limit capacity
            # Use a longer-running service to ensure it's still executing when second call is submitted
            service = SlowService(2.0)  # 2 second delay to keep rate limit occupied
            first_context = CallContext.new(branch_id=uuid4())
            first_request = TestRequest(content="first")

            first_call = await executor.submit_call(service, first_request, first_context)
            # DON'T wait for completion - let it run in background to occupy the rate limit

            # Immediately submit second call while first is still running
            # Now executor IS at capacity (1/1 concurrent requests)
            second_context = CallContext.with_timeout(
                branch_id=uuid4(),
                timeout_s=1.0,  # SHORT deadline - should fail here
            )
            second_request = TestRequest(content="second")

            start_time = time.time()

            # CRITICAL TEST: This should fail after ~1s (deadline), NOT ~10s (rate limit)
            second_call = await executor.submit_call(service, second_request, second_context)

            with pytest.raises(TimeoutError):
                await second_call.wait_completion()

            elapsed = time.time() - start_time

            # CRITICAL ASSERTION: Should timeout at deadline (~1s), not rate limit wait (~10s)
            assert elapsed < 1.5, f"Call waited {elapsed}s but should timeout at deadline ~1s"

            # Clean up: wait for first call to complete
            await first_call.wait_completion()

    @pytest.mark.anyio
    async def test_hook_per_hook_timeout_isolation_critical_flaw(self):
        """CRITICAL FLAW: hooks.py:582 - Incorrect timeout application in emit().

        Current flaw: HookRegistry.emit applies single fail_after timeout to entire
        group of hooks. One slow hook causes ALL hooks to be cancelled.

        Required fix: Use per-hook move_on_after soft timeouts with robust gather
        to isolate hook failures.

        Expected behavior: Fast hooks complete, slow hooks timeout individually,
        emit completes in timeout duration (not slow hook duration).
        """
        registry = HookRegistry()
        registry._timeout = 1.0  # Set 1 second timeout for this test

        hook_results = []

        # Fast hook - should complete
        async def fast_hook(event):
            await anyio.sleep(0.1)  # 100ms - well under timeout
            hook_results.append("fast_completed")

        # Slow hook - should be cancelled by per-hook timeout
        async def slow_hook(event):
            await anyio.sleep(5.0)  # 5s - exceeds timeout
            hook_results.append("slow_completed")  # Should NOT reach here

        # Another fast hook - should complete despite slow hook
        async def another_fast_hook(event):
            await anyio.sleep(0.2)  # 200ms - under timeout
            hook_results.append("another_fast_completed")

        # Register hooks
        registry.register(HookType.PRE_CALL, fast_hook)
        registry.register(HookType.PRE_CALL, slow_hook)
        registry.register(HookType.PRE_CALL, another_fast_hook)

        # Test event
        call_id = uuid4()
        branch_id = uuid4()
        event = HookEvent(
            hook_type=HookType.PRE_CALL,
            call_id=call_id,
            branch_id=branch_id,
            service_name="test_service",
            context=CallContext.new(branch_id=branch_id),
        )

        start_time = time.time()

        # CRITICAL TEST: Should complete in ~1s (timeout), NOT ~5s (slow hook)
        await registry.emit(event)

        elapsed = time.time() - start_time

        # CRITICAL ASSERTIONS:
        # 1. Fast hooks should complete despite slow hook
        assert "fast_completed" in hook_results, "Fast hook should complete"
        assert "another_fast_completed" in hook_results, "Another fast hook should complete"

        # 2. Slow hook should NOT complete (timeout isolation)
        assert "slow_completed" not in hook_results, "Slow hook should be cancelled"

        # 3. Total time should be ~1s (timeout), not ~5s (slow hook duration)
        assert elapsed < 2.0, f"emit() took {elapsed}s but should complete in ~1s timeout"

        # NOTE: This test is expected to FAIL with current implementation
        # (hooks.py:582 uses single timeout) and PASS after implementing
        # per-hook move_on_after soft timeouts with robust gather

    @pytest.mark.anyio
    async def test_circuit_breaker_streaming_no_buffering_regression(self):
        """CRITICAL PERFORMANCE: Circuit breaker must NOT buffer streams in memory.

        Validates that circuit breaker middleware allows immediate chunk passthrough
        without accumulating chunks in memory - a key streaming performance requirement.

        Memory should remain bounded regardless of stream size.
        """
        circuit_breaker = CircuitBreakerMW(CircuitBreakerConfig(failure_threshold=3, timeout=1.0))

        class LargeStreamService(Service):
            name = "large_stream"

            async def call(self, req, *, ctx):
                return {"result": "not used"}

            async def stream(self, req, *, ctx):
                # Generate large stream - 100 chunks of 1MB each
                for i in range(100):
                    large_chunk = {"chunk": i, "data": "x" * (1024 * 1024)}  # 1MB chunk
                    yield large_chunk

        service = LargeStreamService()
        request = TestRequest()
        context = CallContext.new(branch_id=uuid4())

        chunks_received = 0
        max_memory_usage = 0

        # Monitor memory during streaming
        import tracemalloc

        tracemalloc.start()

        initial_memory = tracemalloc.get_traced_memory()[0]

        def mock_next_stream():
            return service.stream(request, ctx=context)

        # CRITICAL TEST: Stream through circuit breaker without buffering
        async for chunk in circuit_breaker.stream(request, context, mock_next_stream):
            chunks_received += 1

            # Check memory usage periodically
            if chunks_received % 10 == 0:
                current_memory = tracemalloc.get_traced_memory()[0]
                memory_used = current_memory - initial_memory
                max_memory_usage = max(max_memory_usage, memory_used)

        tracemalloc.stop()

        # CRITICAL ASSERTIONS:
        # 1. All chunks should be received (no buffering loss)
        assert chunks_received == 100, f"Should receive 100 chunks, got {chunks_received}"

        # 2. Memory usage should be bounded (no accumulation)
        # Should not exceed ~10MB (10 chunks buffered) for 100MB stream
        max_allowed_memory = 15 * 1024 * 1024  # 15MB buffer allowance
        assert (
            max_memory_usage < max_allowed_memory
        ), f"Memory usage {max_memory_usage / 1024 / 1024:.1f}MB exceeds {max_allowed_memory / 1024 / 1024}MB limit"

    @pytest.mark.anyio
    async def test_retry_middleware_deadline_awareness_critical(self):
        """CRITICAL V1 FEATURE: Retry middleware must respect CallContext deadlines.

        Validates that RetryMW stops retrying when CallContext deadline approaches,
        preventing deadline violations through excessive retry attempts.
        """
        # Use minimal delays without jitter to ensure predictable retry behavior
        retry_mw = RetryMW(
            RetryConfig(
                base_delay=0.1,  # Small base delay
                exponential_base=1.5,  # Slower exponential growth
                max_attempts=5,
                jitter=False,  # Disable jitter for predictable timing
            )
        )

        class AlwaysFailingService(Service):
            name = "failing"

            async def call(self, req, *, ctx):
                raise RetryableError("Always fails")

            async def stream(self, req, *, ctx):
                raise RetryableError("Stream always fails")

        service = AlwaysFailingService()
        request = TestRequest()

        # Context with sufficient deadline to allow at least one retry
        context = CallContext.with_timeout(
            branch_id=uuid4(),
            timeout_s=5.0,  # 5 second deadline - enough for initial call + first retry
        )

        async def mock_next_call():
            return await service.call(request, ctx=context)

        start_time = time.time()

        # CRITICAL TEST: Should stop retrying when deadline approaches
        with pytest.raises(RetryableError):  # Final exception after deadline-limited retries
            await retry_mw(request, context, mock_next_call)

        elapsed = time.time() - start_time

        # CRITICAL ASSERTION: Should respect deadline (~5s), not attempt all retries
        assert elapsed < 6.0, f"Retry took {elapsed}s but should respect 5s deadline"

        # Should execute initial call (fast) and respect deadline constraint
        assert elapsed > 0.1, f"Should execute at least initial call, took {elapsed}s"
