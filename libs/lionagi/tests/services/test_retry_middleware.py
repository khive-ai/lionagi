# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive P0 tests for RetryMW resilience middleware.

Critical V1 Features Tested:
- DeadlineAwareness: RetryMW must respect CallContext deadlines and stop retrying when insufficient time remains
- Exponential backoff and jitter validation
- Retryable vs NonRetryableError behavior validation
- Streaming retry behavior and limitations

These are NOT trivial tests - they validate real functionality and edge cases that could
break in production. Each test validates actual behavior, not obvious outcomes.
"""

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from uuid import uuid4

import anyio
import pytest

# MockClock only available with trio backend
try:
    from trio.testing import MockClock
except ImportError:
    # Simple mock for asyncio backend tests
    class MockClock:
        def __init__(self, rate=0.0):
            self.rate = rate
            self._time = 0.0

        def advance(self, seconds):
            self._time += seconds

        def jump(self, seconds):
            self._time += seconds


from lionagi.errors import NonRetryableError, RetryableError, ServiceError, TimeoutError
from lionagi.services.core import CallContext
from lionagi.services.endpoint import RequestModel
from lionagi.services.resilience import RetryConfig, RetryMW


class MockRequest(RequestModel):
    """Mock request for testing."""

    model: str = "test-model"


class RetryTestService:
    """Mock service for testing retry behavior with configurable failure patterns."""

    def __init__(self, fail_pattern: list[Exception | None]):
        """Initialize with failure pattern.

        Args:
            fail_pattern: List where None means success, Exception means raise that exception
        """
        self.fail_pattern = fail_pattern
        self.call_count = 0
        self.stream_call_count = 0

    async def call_operation(self) -> dict[str, Any]:
        """Mock call operation that follows the failure pattern."""
        current_call = self.call_count
        self.call_count += 1

        if current_call >= len(self.fail_pattern):
            return {"success": True, "call": current_call}

        failure = self.fail_pattern[current_call]
        if failure is None:
            return {"success": True, "call": current_call}
        else:
            raise failure

    async def stream_operation(self) -> AsyncIterator[dict[str, Any]]:
        """Mock streaming operation that follows the failure pattern."""
        current_call = self.stream_call_count
        self.stream_call_count += 1

        if current_call >= len(self.fail_pattern):
            yield {"chunk": 1, "call": current_call}
            return

        failure = self.fail_pattern[current_call]
        if failure is None:
            yield {"chunk": 1, "call": current_call}
        else:
            raise failure


@pytest.fixture
def mock_request():
    """Create mock request for testing."""
    return MockRequest(model="test-model")


@pytest.fixture
def retry_config():
    """Create default retry config for testing."""
    return RetryConfig(
        max_attempts=3,
        base_delay=0.1,
        max_delay=10.0,
        exponential_base=2.0,
        jitter=False,  # Disable jitter for deterministic tests
    )


@pytest.fixture
def retry_mw(retry_config):
    """Create RetryMW instance for testing."""
    return RetryMW(retry_config)


class TestRetryMWDeadlineAwareness:
    """Test suite for CRITICAL V1 Feature: DeadlineAwareness.

    These tests validate that RetryMW respects CallContext deadlines and stops
    retrying when insufficient time remains. This is critical for preventing
    retry storms and ensuring predictable latency.
    """

    @pytest.mark.anyio
    async def test_deadline_prevents_retry_when_insufficient_time_remains(
        self, retry_mw, mock_request, mock_clock: MockClock
    ):
        """CRITICAL: RetryMW must stop retrying when CallContext deadline approaches.

        This test validates the core deadline awareness feature. RetryMW should
        calculate remaining time and skip retries if there isn't enough time for
        both the delay and the actual retry operation.
        """
        with mock_clock:
            # Create context with short deadline (2 seconds from now)
            ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=2.0)

            # Create service that fails twice with RetryableError, then succeeds
            # With base_delay=0.1s and exponential_base=2.0:
            # - First retry delay: 0.1s
            # - Second retry delay: 0.2s
            # - Total delay needed: 0.3s + buffer (1.0s) = 1.3s should fit in 2s deadline
            # - But if we advance time to make deadline approach, should skip retry
            service = RetryTestService(
                [
                    RetryableError("Network error"),
                    RetryableError("Another network error"),
                    None,  # Success on third attempt
                ]
            )

            # Advance time to 1.5 seconds, leaving only 0.5s until deadline
            mock_clock.advance(1.5)

            # At this point, remaining time (0.5s) is insufficient for retry
            # (delay + buffer = 0.1s + 1.0s = 1.1s > 0.5s remaining)

            # Attempt the operation - should fail on first try without retrying
            with pytest.raises(RetryableError, match="Network error"):
                await retry_mw(mock_request, ctx, service.call_operation)

            # Verify only one call was made (no retries due to deadline)
            assert service.call_count == 1

            # Verify deadline was respected
            assert ctx.remaining_time is not None
            assert ctx.remaining_time <= 0.5

    @pytest.mark.anyio
    async def test_retry_succeeds_when_deadline_allows_sufficient_time(
        self, retry_mw, mock_request, mock_clock: MockClock
    ):
        """Validate RetryMW succeeds with retries when deadline allows sufficient time."""
        with mock_clock:
            # Create context with generous deadline (10 seconds)
            ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

            # Create service that fails once, then succeeds
            service = RetryTestService(
                [RetryableError("Temporary failure"), None]  # Success on second attempt
            )

            # Should succeed after one retry
            result = await retry_mw(mock_request, ctx, service.call_operation)

            # Verify retry was attempted and succeeded
            assert result["success"] is True
            assert result["call"] == 1  # Second call succeeded
            assert service.call_count == 2  # One failure + one success

    @pytest.mark.anyio
    async def test_deadline_awareness_with_exponential_backoff(
        self, mock_request, mock_clock: MockClock
    ):
        """Validate deadline awareness works correctly with exponential backoff delays."""
        # Create config with longer delays to test deadline calculations
        config = RetryConfig(
            max_attempts=4,
            base_delay=1.0,  # 1 second base delay
            max_delay=10.0,
            exponential_base=2.0,
            jitter=False,
        )
        retry_mw = RetryMW(config)

        with mock_clock:
            # Create context with 5 second deadline
            ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)

            # Create service that always fails with RetryableError
            service = RetryTestService(
                [
                    RetryableError("Failure 1"),
                    RetryableError("Failure 2"),
                    RetryableError("Failure 3"),
                    RetryableError("Failure 4"),
                ]
            )

            # Expected delays: 1.0s, 2.0s, 4.0s (capped delays)
            # Total delay needed for all retries: 7.0s + 3.0s buffer = 10.0s
            # But deadline is only 5.0s, so should stop before exhausting all attempts

            start_time = mock_clock.current_time()

            with pytest.raises(RetryableError):
                await retry_mw(mock_request, ctx, service.call_operation)

            # Should have stopped early due to deadline, not because of max_attempts
            # Verify we didn't make all 4 possible attempts
            assert service.call_count < 4

            # Verify deadline was respected (approximately)
            elapsed = mock_clock.current_time() - start_time
            assert elapsed <= 5.5  # Allow small buffer for deadline checks

    @pytest.mark.anyio
    async def test_no_deadline_allows_full_retry_attempts(self, retry_mw, mock_request):
        """Validate RetryMW uses all retry attempts when no deadline is set."""
        # Create context without deadline
        ctx = CallContext.new(branch_id=uuid4(), deadline_s=None)

        # Create service that always fails
        service = RetryTestService(
            [
                RetryableError("Failure 1"),
                RetryableError("Failure 2"),
                RetryableError("Failure 3"),
            ]
        )

        # Should exhaust all retry attempts
        with pytest.raises(RetryableError, match="Failure 3"):
            await retry_mw(mock_request, ctx, service.call_operation)

        # Verify all attempts were made
        assert service.call_count == 3


class TestRetryMWBackoffAndJitter:
    """Test suite for retry logic with exponential backoff and jitter validation."""

    @pytest.mark.anyio
    async def test_exponential_backoff_calculation(self, mock_request):
        """Validate exponential backoff delays are calculated correctly."""
        config = RetryConfig(
            max_attempts=4,
            base_delay=0.1,
            max_delay=2.0,
            exponential_base=2.0,
            jitter=False,  # Disable for deterministic testing
        )
        retry_mw = RetryMW(config)

        # Test delay computation directly
        delays = [retry_mw._compute_delay(attempt) for attempt in range(4)]

        # Expected: 0.1, 0.2, 0.4, 0.8 (but capped at max_delay=2.0)
        expected_delays = [0.1, 0.2, 0.4, 0.8]

        assert delays == expected_delays

    @pytest.mark.anyio
    async def test_max_delay_cap_respected(self, mock_request):
        """Validate backoff delays are capped at max_delay."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=1.0,
            max_delay=3.0,  # Cap at 3 seconds
            exponential_base=2.0,
            jitter=False,
        )
        retry_mw = RetryMW(config)

        delays = [retry_mw._compute_delay(attempt) for attempt in range(5)]

        # Expected: 1.0, 2.0, 3.0 (capped), 3.0 (capped), 3.0 (capped)
        expected_delays = [1.0, 2.0, 3.0, 3.0, 3.0]

        assert delays == expected_delays

    @pytest.mark.anyio
    async def test_jitter_reduces_delay_variance(self, mock_request):
        """Validate jitter introduces randomness but keeps delays within bounds."""
        config = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=True,  # Enable jitter
        )
        retry_mw = RetryMW(config)

        # Generate multiple delays for each attempt to test jitter
        for attempt in range(3):
            delays = [retry_mw._compute_delay(attempt) for _ in range(10)]

            base_delay = min(
                config.base_delay * (config.exponential_base**attempt), config.max_delay
            )

            # All jittered delays should be between 0 and base_delay
            for delay in delays:
                assert 0 <= delay <= base_delay

            # Should have some variance (not all the same)
            assert len(set(delays)) > 1


class TestRetryMWErrorTypeHandling:
    """Test suite for retryable vs NonRetryableError behavior validation."""

    @pytest.mark.anyio
    async def test_non_retryable_error_fails_immediately(self, retry_mw, mock_request):
        """Validate NonRetryableError causes immediate failure without retries."""
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        service = RetryTestService(
            [
                NonRetryableError("Client error - invalid request"),
                None,  # This should never be reached
            ]
        )

        # Should fail immediately without retrying
        with pytest.raises(NonRetryableError, match="Client error - invalid request"):
            await retry_mw(mock_request, ctx, service.call_operation)

        # Verify no retries were attempted
        assert service.call_count == 1

    @pytest.mark.anyio
    async def test_retryable_error_triggers_retries(self, retry_mw, mock_request):
        """Validate RetryableError triggers retry attempts."""
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        service = RetryTestService(
            [
                RetryableError("Network timeout"),
                RetryableError("Rate limit exceeded"),
                None,  # Success on third attempt
            ]
        )

        # Should succeed after retries
        result = await retry_mw(mock_request, ctx, service.call_operation)

        assert result["success"] is True
        assert result["call"] == 2  # Third call (index 2) succeeded
        assert service.call_count == 3

    @pytest.mark.anyio
    async def test_timeout_error_triggers_retries(self, retry_mw, mock_request):
        """Validate TimeoutError is treated as retryable."""
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        service = RetryTestService(
            [TimeoutError("Request timeout"), None]  # Success on second attempt
        )

        result = await retry_mw(mock_request, ctx, service.call_operation)

        assert result["success"] is True
        assert service.call_count == 2

    @pytest.mark.anyio
    async def test_mixed_error_types_handled_correctly(self, retry_mw, mock_request):
        """Validate mixed error types are handled according to their retry behavior."""
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        # First call succeeds to establish baseline
        service = RetryTestService([None])
        result = await retry_mw(mock_request, ctx, service.call_operation)
        assert result["success"] is True

        # Reset service and test NonRetryableError stops immediately
        service = RetryTestService(
            [
                RetryableError("Transient failure"),  # This should be retried
                NonRetryableError("Permanent failure"),  # This should stop retries
            ]
        )

        with pytest.raises(NonRetryableError, match="Permanent failure"):
            await retry_mw(mock_request, ctx, service.call_operation)

        # Should have made both calls (retry then fail)
        assert service.call_count == 2

    @pytest.mark.anyio
    async def test_exhausted_retries_raises_last_error(self, retry_mw, mock_request):
        """Validate last error is raised when all retry attempts are exhausted."""
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        service = RetryTestService(
            [
                RetryableError("Error 1"),
                RetryableError("Error 2"),
                RetryableError("Final error"),  # This should be the raised error
            ]
        )

        # Should fail with the last error after exhausting retries
        with pytest.raises(RetryableError, match="Final error"):
            await retry_mw(mock_request, ctx, service.call_operation)

        assert service.call_count == 3


class TestRetryMWStreamingBehavior:
    """Test suite for streaming retry behavior and limitations."""

    @pytest.mark.anyio
    async def test_stream_retry_before_yielding_chunks(self, retry_mw, mock_request):
        """Validate stream can be retried if it fails before yielding any chunks."""
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        service = RetryTestService(
            [
                RetryableError("Stream initialization failed"),
                None,  # Success on second attempt
            ]
        )

        chunks = []
        async for chunk in retry_mw.stream(mock_request, ctx, service.stream_operation):
            chunks.append(chunk)

        # Should have succeeded after retry
        assert len(chunks) == 1
        assert chunks[0]["chunk"] == 1
        assert chunks[0]["call"] == 1  # Second call
        assert service.stream_call_count == 2

    @pytest.mark.anyio
    async def test_stream_cannot_retry_after_yielding_chunks(self, mock_request):
        """Validate stream cannot be retried once it has started yielding chunks."""
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)
        retry_mw = RetryMW(RetryConfig(max_attempts=3, base_delay=0.1, jitter=False))

        class PartialStreamService:
            def __init__(self):
                self.call_count = 0

            async def stream_operation(self) -> AsyncIterator[dict[str, Any]]:
                self.call_count += 1
                yield {"chunk": 1, "call": self.call_count}
                # Fail after yielding one chunk
                raise RetryableError("Stream failed mid-way")

        service = PartialStreamService()

        # Should fail without retrying since chunks were already yielded
        chunks = []
        with pytest.raises(RetryableError, match="Stream failed mid-way"):
            async for chunk in retry_mw.stream(mock_request, ctx, service.stream_operation):
                chunks.append(chunk)

        # Should have received the first chunk before failure
        assert len(chunks) == 1
        assert chunks[0]["chunk"] == 1

        # No retry should have been attempted
        assert service.call_count == 1

    @pytest.mark.anyio
    async def test_stream_non_retryable_error_fails_immediately(self, retry_mw, mock_request):
        """Validate NonRetryableError in streams fails immediately without retries."""
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        service = RetryTestService(
            [
                NonRetryableError("Invalid stream request"),
                None,  # Should never be reached
            ]
        )

        with pytest.raises(NonRetryableError, match="Invalid stream request"):
            async for _ in retry_mw.stream(mock_request, ctx, service.stream_operation):
                pass  # Should not yield anything

        assert service.stream_call_count == 1


class TestRetryMWEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.anyio
    async def test_zero_max_attempts_fails_immediately(self, mock_request):
        """Validate retry middleware with zero max attempts fails on first error."""
        config = RetryConfig(max_attempts=0, base_delay=0.1, jitter=False)
        retry_mw = RetryMW(config)
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        service = RetryTestService([RetryableError("Should fail immediately")])

        with pytest.raises(ServiceError, match="All 0 retry attempts failed"):
            await retry_mw(mock_request, ctx, service.call_operation)

        assert service.call_count == 0  # No attempts should be made

    @pytest.mark.anyio
    async def test_single_max_attempt_no_retries(self, mock_request):
        """Validate single max attempt means no retries on failure."""
        config = RetryConfig(max_attempts=1, base_delay=0.1, jitter=False)
        retry_mw = RetryMW(config)
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        service = RetryTestService(
            [RetryableError("First failure"), None]  # Should never be reached
        )

        with pytest.raises(RetryableError, match="First failure"):
            await retry_mw(mock_request, ctx, service.call_operation)

        assert service.call_count == 1  # Only one attempt

    @pytest.mark.anyio
    async def test_expired_deadline_at_start_fails_immediately(
        self, mock_request, mock_clock: MockClock
    ):
        """Validate already expired deadline fails immediately."""
        config = RetryConfig(max_attempts=3, base_delay=0.1, jitter=False)
        retry_mw = RetryMW(config)

        with mock_clock:
            # Create context that's already expired
            ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=1.0)
            mock_clock.advance(2.0)  # Advance past deadline

            assert ctx.is_expired  # Verify context is expired

            service = RetryTestService([RetryableError("Should not retry")])

            # Should fail immediately without any attempts
            with pytest.raises(RetryableError, match="Should not retry"):
                await retry_mw(mock_request, ctx, service.call_operation)

            assert service.call_count == 1  # One attempt, no retries
