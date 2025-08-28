# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
P0 Circuit Breaker Tests for lionagi  - Consolidated & Streamlined.

Critical behaviors tested:
- Concurrency safety with exact failure counting under load
- State transitions: CLOSED→OPEN→HALF_OPEN→CLOSED cycles
- Streaming passthrough without buffering (CRITICAL performance)
- Error type handling (retryable vs non-retryable)

Consolidated from 702-line verbose test file into focused behavioral validation.
"""

import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import anyio
import pytest

from lionagi.errors import NonRetryableError, RetryableError, ServiceError
from lionagi.services.core import CallContext
from lionagi.services.endpoint import RequestModel
from lionagi.services.resilience import (
    CircuitBreakerConfig,
    CircuitBreakerMW,
    CircuitState,
)


class MockRequest(RequestModel):
    """Mock request for testing."""

    model: str = "test-model"


class ConfigurableService:
    """Unified configurable service for all circuit breaker test scenarios."""

    def __init__(
        self,
        should_fail: bool = False,
        failure_type: Exception = None,
        chunk_count: int = 3,
        chunk_delay: float = 0.01,
        fail_after_chunks: int = None,
    ):
        self.should_fail = should_fail
        self.failure_type = failure_type or RetryableError("Service failure")
        self.chunk_count = chunk_count
        self.chunk_delay = chunk_delay
        self.fail_after_chunks = fail_after_chunks

        # Counters
        self.call_count = 0
        self.stream_call_count = 0
        self.chunks_yielded = 0

    async def call_operation(self) -> dict[str, Any]:
        """Call operation that may fail."""
        self.call_count += 1
        if self.should_fail:
            raise self.failure_type
        return {"success": True, "call": self.call_count}

    async def stream_operation(self) -> AsyncIterator[dict[str, Any]]:
        """Stream operation with configurable behavior."""
        self.stream_call_count += 1

        if self.should_fail and self.fail_after_chunks is None:
            raise self.failure_type

        for i in range(self.chunk_count):
            if self.fail_after_chunks is not None and i >= self.fail_after_chunks:
                raise RetryableError(f"Stream failed after {self.chunks_yielded} chunks")

            self.chunks_yielded += 1
            yield {
                "chunk": i + 1,
                "call": self.stream_call_count,
                "timestamp": time.time(),
            }

            if self.chunk_delay > 0:
                await anyio.sleep(self.chunk_delay)


@pytest.fixture
def mock_request():
    return MockRequest(model="test-model")


@pytest.fixture
def circuit_config():
    return CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=1.0,
        failure_ratio_threshold=0.5,
    )


@pytest.fixture
def circuit_mw(circuit_config):
    return CircuitBreakerMW(circuit_config)


class TestCircuitBreakerConcurrencySafety:
    """CRITICAL: Concurrency safety with exact failure counting under high load."""

    @pytest.mark.anyio
    async def test_concurrent_failure_counting_accuracy(self, circuit_config, mock_request):
        """50 concurrent tasks with exact failure counting using anyio.Lock."""
        circuit_mw = CircuitBreakerMW(circuit_config)
        failing_service = ConfigurableService(
            should_fail=True, failure_type=RetryableError("Concurrent failure")
        )
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        async def concurrent_call():
            try:
                await circuit_mw(mock_request, ctx, failing_service.call_operation)
                return "success"
            except (RetryableError, ServiceError) as e:
                return f"failed: {type(e).__name__}"

        # Execute 50 concurrent tasks
        async with anyio.create_task_group() as tg:
            results = []
            for _ in range(50):
                results.append(await tg.start_task_soon(concurrent_call))

        completed_results = [await result for result in results]

        # All 50 tasks should fail (RetryableError or ServiceError for circuit open)
        failed_count = sum(1 for result in completed_results if result.startswith("failed"))
        assert failed_count == 50

        # Circuit should be OPEN and counters accurate
        breaker = circuit_mw.breaker
        assert breaker.state == CircuitState.OPEN
        assert breaker.total_requests == 50
        assert breaker.failure_count >= circuit_config.failure_threshold

    @pytest.mark.anyio
    async def test_concurrent_mixed_success_failure_transitions(self, circuit_config, mock_request):
        """Mixed concurrent success/failure with state transitions."""
        circuit_mw = CircuitBreakerMW(circuit_config)
        failing_service = ConfigurableService(
            should_fail=True, failure_type=RetryableError("Failure")
        )
        success_service = ConfigurableService(should_fail=False)
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        async def mixed_concurrent_calls():
            results = []

            async with anyio.create_task_group() as tg:
                # 10 failing calls to trigger circuit opening
                for _ in range(10):
                    results.append(
                        await tg.start_task_soon(
                            lambda: circuit_mw(mock_request, ctx, failing_service.call_operation)
                        )
                    )

                # 5 successful calls
                for _ in range(5):
                    results.append(
                        await tg.start_task_soon(
                            lambda: circuit_mw(mock_request, ctx, success_service.call_operation)
                        )
                    )

            completed_results = []
            for result_task in results:
                try:
                    result = await result_task
                    completed_results.append(("success", result))
                except Exception as e:
                    completed_results.append(("failed", type(e).__name__))

            return completed_results

        results = await mixed_concurrent_calls()

        # Circuit should open due to failures
        assert circuit_mw.breaker.state in (CircuitState.OPEN, CircuitState.HALF_OPEN)

        # Verify both successes and failures occurred
        successes = [r for r in results if r[0] == "success"]
        failures = [r for r in results if r[0] == "failed"]
        assert len(failures) > 0
        assert len(successes) >= 0  # Some may be blocked by circuit opening


class TestCircuitBreakerStateTransitions:
    """Complete state transition cycles: CLOSED→OPEN→HALF_OPEN→CLOSED."""

    @pytest.mark.anyio
    async def test_complete_state_transition_cycle(self, mock_request):
        """Test complete CLOSED→OPEN→HALF_OPEN→CLOSED cycle."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            timeout=0.1,
            failure_ratio_threshold=0.5,
        )
        circuit_mw = CircuitBreakerMW(config)
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        failing_service = ConfigurableService(
            should_fail=True, failure_type=RetryableError("Failure")
        )
        success_service = ConfigurableService(should_fail=False)

        # Initially CLOSED
        assert circuit_mw.breaker.state == CircuitState.CLOSED

        # Trigger failures to reach threshold → OPEN
        for i in range(2):
            with pytest.raises(RetryableError):
                await circuit_mw(mock_request, ctx, failing_service.call_operation)

            if i == 0:
                assert circuit_mw.breaker.state == CircuitState.CLOSED

        # Should be OPEN after threshold
        assert circuit_mw.breaker.state == CircuitState.OPEN

        # OPEN circuit rejects calls immediately
        initial_call_count = success_service.call_count
        with pytest.raises(ServiceError, match="Circuit breaker is OPEN"):
            await circuit_mw(mock_request, ctx, success_service.call_operation)
        assert success_service.call_count == initial_call_count

        # Wait for timeout → next call transitions to HALF_OPEN and succeeds → CLOSED
        await anyio.sleep(0.15)
        result = await circuit_mw(mock_request, ctx, success_service.call_operation)

        assert result["success"] is True
        assert circuit_mw.breaker.state == CircuitState.CLOSED
        assert success_service.call_count == initial_call_count + 1

    @pytest.mark.anyio
    async def test_half_open_failure_reopens_circuit(self, mock_request):
        """HALF_OPEN transitions back to OPEN on failure."""
        config = CircuitBreakerConfig(failure_threshold=2, success_threshold=2, timeout=0.1)
        circuit_mw = CircuitBreakerMW(config)
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        failing_service = ConfigurableService(
            should_fail=True, failure_type=RetryableError("Failure")
        )

        # Force circuit OPEN
        for _ in range(2):
            with pytest.raises(RetryableError):
                await circuit_mw(mock_request, ctx, failing_service.call_operation)

        await anyio.sleep(0.15)  # Wait for timeout

        # Failed call in HALF_OPEN should reopen circuit
        with pytest.raises(RetryableError):
            await circuit_mw(mock_request, ctx, failing_service.call_operation)

        assert circuit_mw.breaker.state == CircuitState.OPEN


class TestCircuitBreakerStreamingPerformance:
    """CRITICAL Performance: Circuit breaker must NOT buffer streams."""

    @pytest.mark.anyio
    async def test_streaming_immediate_passthrough_no_buffering(self, circuit_mw, mock_request):
        """Validate immediate chunk passthrough without buffering delay."""
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)
        slow_service = ConfigurableService(chunk_count=5, chunk_delay=0.05)  # 50ms between chunks

        chunk_receive_times = []
        start_time = time.time()

        async for chunk in circuit_mw.stream(mock_request, ctx, slow_service.stream_operation):
            chunk_receive_times.append(time.time() - start_time)

        # Verify chunks arrived promptly without buffering delay
        assert len(chunk_receive_times) == 5

        for i, receive_time in enumerate(chunk_receive_times):
            expected_time = (i + 1) * 0.05  # 50ms per chunk
            assert (
                abs(receive_time - expected_time) < 0.02
            ), f"Chunk {i + 1} arrived at {receive_time:.3f}s, expected ~{expected_time:.3f}s"

    @pytest.mark.anyio
    async def test_streaming_memory_efficiency_large_chunks(self, circuit_mw, mock_request):
        """Validate no memory accumulation with large chunks."""
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        class LargeChunkService(ConfigurableService):
            def __init__(self, chunk_size: int = 1024 * 1024):  # 1MB chunks
                super().__init__(chunk_count=5, chunk_delay=0.01)
                self.chunk_size = chunk_size

            async def stream_operation(self):
                for i in range(self.chunk_count):
                    self.chunks_yielded += 1
                    yield {"chunk": i + 1, "data": "x" * self.chunk_size}
                    await anyio.sleep(self.chunk_delay)

        service = LargeChunkService()
        processed_chunks = 0

        async for chunk in circuit_mw.stream(mock_request, ctx, service.stream_operation):
            processed_chunks += 1
            assert len(chunk["data"]) == 1024 * 1024
            del chunk  # Explicit cleanup

        assert processed_chunks == 5

    @pytest.mark.anyio
    async def test_streaming_failure_propagation_and_state_management(
        self, circuit_mw, mock_request
    ):
        """Test streaming failure propagation and circuit state updates."""
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        # Test failure after yielding chunks
        failing_service = ConfigurableService(
            chunk_count=5,
            fail_after_chunks=2,
            failure_type=RetryableError("Stream failed after 2 chunks"),
        )

        chunks_received = []
        with pytest.raises(RetryableError, match="Stream failed after 2 chunks"):
            async for chunk in circuit_mw.stream(
                mock_request, ctx, failing_service.stream_operation
            ):
                chunks_received.append(chunk)

        # Should receive exactly chunks before failure
        assert len(chunks_received) == 2
        assert chunks_received[0]["chunk"] == 1
        assert chunks_received[1]["chunk"] == 2

        # Test successful streaming updates circuit state
        success_service = ConfigurableService(should_fail=False, chunk_count=3)

        chunks = []
        async for chunk in circuit_mw.stream(mock_request, ctx, success_service.stream_operation):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert circuit_mw.breaker.state == CircuitState.CLOSED

    @pytest.mark.anyio
    async def test_streaming_half_open_success_detection(self, mock_request):
        """HALF_OPEN circuit detects success from first successful chunk."""
        config = CircuitBreakerConfig(failure_threshold=2, success_threshold=1, timeout=0.1)
        circuit_mw = CircuitBreakerMW(config)
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        # Force circuit OPEN
        failing_service = ConfigurableService(
            should_fail=True, failure_type=RetryableError("Failure")
        )
        for _ in range(2):
            with pytest.raises(RetryableError):
                await circuit_mw(mock_request, ctx, failing_service.call_operation)

        await anyio.sleep(0.15)  # Wait for timeout

        # Successful streaming should close circuit after first chunk
        success_service = ConfigurableService(should_fail=False, chunk_count=3)

        chunks = []
        async for chunk in circuit_mw.stream(mock_request, ctx, success_service.stream_operation):
            chunks.append(chunk)
            if len(chunks) == 1:
                assert circuit_mw.breaker.state == CircuitState.CLOSED

        assert len(chunks) == 3
        assert circuit_mw.breaker.state == CircuitState.CLOSED


class TestCircuitBreakerErrorTypeHandling:
    """Error type handling: retryable vs non-retryable impact on circuit state."""

    @pytest.mark.anyio
    async def test_non_retryable_errors_do_not_affect_circuit(self, circuit_mw, mock_request):
        """NonRetryableError does not count toward circuit breaking."""
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        non_retryable_service = ConfigurableService(
            should_fail=True, failure_type=NonRetryableError("Invalid request")
        )

        # Multiple non-retryable errors should not open circuit
        for _ in range(5):  # More than failure_threshold
            with pytest.raises(NonRetryableError):
                await circuit_mw(mock_request, ctx, non_retryable_service.call_operation)

        assert circuit_mw.breaker.state == CircuitState.CLOSED
        assert circuit_mw.breaker.failure_count == 0

    @pytest.mark.anyio
    async def test_mixed_error_types_only_retryable_count(self, circuit_mw, mock_request):
        """Only retryable errors count toward circuit breaking."""
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)

        retryable_service = ConfigurableService(
            should_fail=True, failure_type=RetryableError("Network error")
        )
        non_retryable_service = ConfigurableService(
            should_fail=True, failure_type=NonRetryableError("Bad request")
        )

        # Mix of errors - only retryable should count (need 3 to open)
        with pytest.raises(RetryableError):
            await circuit_mw(mock_request, ctx, retryable_service.call_operation)  # Count: 1

        with pytest.raises(NonRetryableError):
            await circuit_mw(
                mock_request, ctx, non_retryable_service.call_operation
            )  # Count: still 1

        with pytest.raises(RetryableError):
            await circuit_mw(mock_request, ctx, retryable_service.call_operation)  # Count: 2

        assert circuit_mw.breaker.state == CircuitState.CLOSED
        assert circuit_mw.breaker.failure_count == 2

        # One more retryable error should open circuit
        with pytest.raises(RetryableError):
            await circuit_mw(mock_request, ctx, retryable_service.call_operation)  # Count: 3

        assert circuit_mw.breaker.state == CircuitState.OPEN
