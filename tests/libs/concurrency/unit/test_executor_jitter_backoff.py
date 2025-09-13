"""Tests for AsyncExecutor jitter and backoff retry features."""

import asyncio
import time
from typing import Any

import anyio
import pytest

from lionagi.ln.concurrency.executor import AsyncExecutor


class CustomRetryableError(Exception):
    """Custom exception for testing selective retry."""

    pass


class NonRetryableError(Exception):
    """Exception that should not trigger retries."""

    pass


@pytest.mark.anyio
async def test_retry_jitter_adds_randomness():
    """Test that jitter adds randomness to retry delays."""
    timestamps = []

    async def capture_timestamp_func(x):
        timestamps.append(time.perf_counter())
        if len(timestamps) <= 2:  # Fail first two attempts
            raise ValueError("Transient failure")
        return x * 2

    async with AsyncExecutor(
        retry_attempts=2,
        retry_delay=0.05,  # 50ms base
        retry_jitter=0.5,  # 50% jitter
        retry_on=(ValueError,),
    ) as executor:
        result = await executor(capture_timestamp_func, [1])
        assert result == [2]

    # Should have 3 timestamps
    assert len(timestamps) == 3

    # Calculate actual delays between attempts
    delay1 = timestamps[1] - timestamps[0]
    delay2 = timestamps[2] - timestamps[1]

    # Check that delays are reasonable (with jitter they won't be exact)
    assert 0.025 <= delay1 <= 0.125  # 50ms ± 50% jitter range
    assert 0.05 <= delay2 <= 0.25  # 100ms ± 50% jitter range


@pytest.mark.anyio
async def test_retry_on_selective_exception_handling():
    """Test that retry_on parameter selectively handles exceptions."""
    attempts = []

    async def selective_failure_func(x):
        attempts.append(x)
        if x == 1:
            raise CustomRetryableError("Should retry this")
        elif x == 2:
            raise NonRetryableError("Should not retry this")
        return x * 3

    async with AsyncExecutor(
        retry_attempts=2,
        retry_delay=0.01,
        retry_on=(CustomRetryableError,),  # Only retry CustomRetryableError
    ) as executor:
        # Test retryable error
        with pytest.raises(CustomRetryableError):
            await executor(selective_failure_func, [1])

        # Should have tried 3 times for x=1 (original + 2 retries)
        retry_attempts = [a for a in attempts if a == 1]
        assert len(retry_attempts) == 3

        # Reset for next test
        attempts.clear()

        # Test non-retryable error
        with pytest.raises(NonRetryableError):
            await executor(selective_failure_func, [2])

        # Should have tried only once for x=2 (no retries)
        no_retry_attempts = [a for a in attempts if a == 2]
        assert len(no_retry_attempts) == 1


@pytest.mark.anyio
async def test_exponential_backoff_timing():
    """Test that exponential backoff increases delay exponentially."""
    attempt_times = []

    async def timing_func(x):
        attempt_times.append(time.perf_counter())
        if len(attempt_times) <= 3:  # Fail first 3 attempts
            raise ValueError("Timing test failure")
        return x

    async with AsyncExecutor(
        retry_attempts=3,
        retry_delay=0.02,  # 20ms base
        retry_max_delay=0.16,  # 160ms max
        retry_jitter=0.0,  # No jitter for precise timing
        retry_on=(ValueError,),
    ) as executor:
        result = await executor(timing_func, [42])
        assert result == [42]

    # Should have 4 attempt times
    assert len(attempt_times) == 4

    # Calculate delays between attempts
    delays = [attempt_times[i + 1] - attempt_times[i] for i in range(3)]

    # Expected delays: ~20ms, ~40ms, ~80ms (exponential backoff)
    # Allow some tolerance for timing variations
    assert 0.015 <= delays[0] <= 0.035  # ~20ms
    assert 0.030 <= delays[1] <= 0.060  # ~40ms
    assert 0.060 <= delays[2] <= 0.120  # ~80ms


@pytest.mark.anyio
async def test_max_delay_caps_backoff():
    """Test that max_delay parameter caps exponential backoff."""
    attempt_times = []

    async def capped_delay_func(x):
        attempt_times.append(time.perf_counter())
        if len(attempt_times) <= 4:  # Fail first 4 attempts
            raise ValueError("Max delay test")
        return x

    async with AsyncExecutor(
        retry_attempts=4,
        retry_delay=0.01,  # 10ms base
        retry_max_delay=0.03,  # Cap at 30ms
        retry_jitter=0.0,  # No jitter
        retry_on=(ValueError,),
    ) as executor:
        result = await executor(capped_delay_func, [1])
        assert result == [1]

    # Calculate delays
    delays = [attempt_times[i + 1] - attempt_times[i] for i in range(4)]

    # Expected: ~10ms, ~20ms, ~30ms (capped), ~30ms (capped)
    assert 0.008 <= delays[0] <= 0.015  # ~10ms
    assert 0.015 <= delays[1] <= 0.030  # ~20ms
    assert 0.025 <= delays[2] <= 0.040  # ~30ms (capped)
    assert 0.025 <= delays[3] <= 0.040  # ~30ms (capped)


@pytest.mark.anyio
async def test_multiple_function_retry_with_jitter():
    """Test retry with jitter for multiple function execution."""
    call_count = {}

    def make_flaky_func(func_id: str):
        async def flaky_func(x, y=0):
            key = f"{func_id}_{x}_{y}"
            call_count[key] = call_count.get(key, 0) + 1
            if call_count[key] <= 2:  # Fail twice
                raise CustomRetryableError(f"Flaky {func_id}")
            return x + y + int(func_id)

        return flaky_func

    funcs = [make_flaky_func("1"), make_flaky_func("2")]
    args_kwargs = [((10,), {"y": 1}), ((20,), {"y": 2})]

    async with AsyncExecutor(
        retry_attempts=3,
        retry_delay=0.01,
        retry_jitter=0.2,
        retry_on=(CustomRetryableError,),
    ) as executor:
        results = await executor(funcs, args_kwargs)
        # Expected: [10+1+1, 20+2+2] = [12, 24]
        assert results == [12, 24]

    # Each function should have been called 3 times
    assert call_count["1_10_1"] == 3
    assert call_count["2_20_2"] == 3


@pytest.mark.anyio
async def test_no_retry_when_attempts_zero():
    """Test that setting retry_attempts=0 disables retry completely."""
    attempts = []

    async def no_retry_func(x):
        attempts.append(x)
        raise ValueError("Always fails")

    async with AsyncExecutor(
        retry_attempts=0,  # No retries
        retry_jitter=0.5,  # Should be ignored
        retry_on=(ValueError,),
    ) as executor:
        with pytest.raises(ValueError):
            await executor(no_retry_func, [1])

    # Should only attempt once
    assert len(attempts) == 1
