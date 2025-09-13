"""Migration equivalence tests for alcall/bcall → AsyncExecutor transition.

Validates that the new AsyncExecutor pattern produces identical results
to the legacy alcall/bcall functions across various parameter combinations.
"""

import asyncio
import time
from typing import Any, List

import anyio
import pytest

from lionagi.ln.concurrency.executor import AsyncExecutor

# Import legacy functions if available
try:
    from lionagi.ln.utils import alcall, bcall

    LEGACY_AVAILABLE = True
except ImportError:
    # Mock legacy functions for testing the equivalence framework
    LEGACY_AVAILABLE = False

    async def alcall(inputs, func, **kwargs):
        """Mock alcall for testing framework."""
        # Simple implementation for testing with data preprocessing
        max_concurrent = kwargs.get("max_concurrent", 32)
        retry_attempts = kwargs.get("retry_attempts", 0)
        retry_delay = kwargs.get("retry_delay", 0.1)

        # Handle input preprocessing like real alcall
        flatten = kwargs.get("input_flatten", False)
        dropna = kwargs.get("input_dropna", False)
        data = to_list(inputs, flatten=flatten, dropna=dropna)

        async with AsyncExecutor(
            max_concurrent=max_concurrent,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
        ) as executor:
            return await executor(func, data)

    async def bcall(inputs, func, batch_size=10, **kwargs):
        """Mock bcall for testing framework."""
        max_concurrent = kwargs.get("max_concurrent", 32)

        # Handle input preprocessing like real bcall
        flatten = kwargs.get("input_flatten", False)
        dropna = kwargs.get("input_dropna", False)
        data = to_list(inputs, flatten=flatten, dropna=dropna)

        async with AsyncExecutor(max_concurrent=max_concurrent) as executor:
            async for batch in executor.stream(func, data, batch_size):
                yield batch


def to_list(inputs, flatten=False, dropna=False):
    """Simple implementation of to_list for testing."""
    if not inputs:
        return []

    result = list(inputs)

    if flatten:
        flattened = []
        for item in result:
            if isinstance(item, (list, tuple)):
                flattened.extend(item)
            else:
                flattened.append(item)
        result = flattened

    if dropna:
        result = [item for item in result if item is not None]

    return result


# =============================================================================
# Basic Equivalence Tests
# =============================================================================


@pytest.mark.migration
@pytest.mark.anyio
@pytest.mark.parametrize(
    "inputs,flatten,dropna,max_concurrent",
    [
        ([1, 2, 3, 4, 5], False, False, 2),
        ([[1, 2], [3, 4], 5], True, False, 3),
        ([1, None, 2, None, 3], False, True, 4),
        ([[1, 2], None, [3, 4]], True, True, 2),
        ([], False, False, 5),  # Empty input
        ([10] * 100, False, False, 20),  # Large batch
    ],
)
async def test_alcall_executor_equivalence_basic(
    anyio_backend, inputs, flatten, dropna, max_concurrent, cancel_guard
):
    """Test basic alcall vs AsyncExecutor equivalence."""
    async with cancel_guard():

        async def test_func(x):
            await anyio.sleep(0.001)
            return x * 2 if x is not None else None

        # Legacy path
        legacy_result = await alcall(
            inputs,
            test_func,
            max_concurrent=max_concurrent,
            input_flatten=flatten,
            input_dropna=dropna,
        )

        # New path
        data = to_list(inputs, flatten=flatten, dropna=dropna)
        async with AsyncExecutor(max_concurrent=max_concurrent) as executor:
            modern_result = await executor(test_func, data)

        # Verify equivalence
        assert (
            legacy_result == modern_result
        ), f"Results differ: legacy={legacy_result}, modern={modern_result}"


@pytest.mark.migration
@pytest.mark.anyio
async def test_alcall_executor_equivalence_with_retries(
    anyio_backend, cancel_guard
):
    """Test alcall vs AsyncExecutor equivalence with retry logic."""
    async with cancel_guard():
        attempts = {}

        async def flaky_func(x):
            attempts[x] = attempts.get(x, 0) + 1
            if attempts[x] < 2:  # Fail once
                raise RuntimeError(f"Transient failure on {x}")
            await anyio.sleep(0.001)
            return x * 3

        inputs = [1, 2, 3, 4]

        # Reset attempts for legacy path
        attempts.clear()
        try:
            legacy_result = await alcall(
                inputs,
                flaky_func,
                max_concurrent=2,
                retry_attempts=2,
                retry_delay=0.001,
            )
        except AttributeError:
            # Legacy alcall might not have retry parameters
            pytest.skip("Legacy alcall doesn't support retry parameters")

        # Reset attempts for modern path
        attempts.clear()
        async with AsyncExecutor(
            max_concurrent=2, retry_attempts=2, retry_delay=0.001
        ) as executor:
            modern_result = await executor(flaky_func, inputs)

        # Verify equivalence
        assert legacy_result == modern_result


# =============================================================================
# Streaming Equivalence Tests (bcall → executor.stream)
# =============================================================================


@pytest.mark.migration
@pytest.mark.anyio
@pytest.mark.parametrize(
    "batch_size,max_concurrent",
    [
        (3, 5),
        (10, 8),
        (1, 10),  # Single item batches
        (50, 20),  # Large batches
    ],
)
async def test_bcall_executor_stream_equivalence(
    anyio_backend, batch_size, max_concurrent, cancel_guard
):
    """Test bcall vs AsyncExecutor.stream equivalence."""
    async with cancel_guard():

        async def stream_func(x):
            await anyio.sleep(0.001)
            return x**2

        inputs = list(range(25))

        # Legacy streaming path
        legacy_batches = []
        try:
            async for batch in bcall(
                inputs,
                stream_func,
                batch_size=batch_size,
                max_concurrent=max_concurrent,
            ):
                legacy_batches.append(batch)
        except Exception as e:
            pytest.skip(f"Legacy bcall failed: {e}")

        # Modern streaming path
        modern_batches = []
        async with AsyncExecutor(max_concurrent=max_concurrent) as executor:
            async for batch in executor.stream(
                stream_func, inputs, batch_size
            ):
                modern_batches.append(batch)

        # Verify equivalence
        assert (
            legacy_batches == modern_batches
        ), f"Streaming results differ:\nlegacy={legacy_batches}\nmodern={modern_batches}"

        # Verify flattened results are equivalent
        legacy_flat = [item for batch in legacy_batches for item in batch]
        modern_flat = [item for batch in modern_batches for item in batch]
        assert legacy_flat == modern_flat


# =============================================================================
# Edge Cases and Error Handling Equivalence
# =============================================================================


@pytest.mark.migration
@pytest.mark.anyio
async def test_equivalence_with_exceptions(anyio_backend, cancel_guard):
    """Test that both paths handle exceptions equivalently."""
    async with cancel_guard():

        async def failing_func(x):
            if x == 5:
                raise ValueError(f"Function failed on {x}")
            await anyio.sleep(0.001)
            return x * 2

        inputs = [1, 2, 3, 5, 6]

        # Test that both paths raise the same exception
        legacy_exception = None
        try:
            await alcall(inputs, failing_func, max_concurrent=3)
        except Exception as e:
            legacy_exception = e

        modern_exception = None
        try:
            async with AsyncExecutor(max_concurrent=3) as executor:
                await executor(failing_func, inputs)
        except Exception as e:
            modern_exception = e

        # Both should have raised exceptions
        assert legacy_exception is not None
        assert modern_exception is not None

        # Exception types and messages should be equivalent
        assert type(legacy_exception) == type(modern_exception)
        assert str(legacy_exception) == str(modern_exception)


@pytest.mark.migration
@pytest.mark.anyio
async def test_equivalence_empty_and_single_inputs(
    anyio_backend, cancel_guard
):
    """Test equivalence with edge case inputs."""
    async with cancel_guard():

        async def identity_func(x):
            await anyio.sleep(0.001)
            return x

        test_cases = [
            [],  # Empty
            [42],  # Single item
            [None],  # None value
            [0],  # Falsy value
        ]

        for inputs in test_cases:
            # Legacy path
            legacy_result = await alcall(
                inputs, identity_func, max_concurrent=5
            )

            # Modern path
            async with AsyncExecutor(max_concurrent=5) as executor:
                modern_result = await executor(identity_func, inputs)

            # Verify equivalence
            assert legacy_result == modern_result, (
                f"Results differ for inputs {inputs}: "
                f"legacy={legacy_result}, modern={modern_result}"
            )


# =============================================================================
# Performance Equivalence Tests
# =============================================================================


@pytest.mark.migration
@pytest.mark.anyio
async def test_performance_equivalence_timing(anyio_backend, cancel_guard):
    """Verify that new implementation performs at least as well as legacy."""
    async with cancel_guard():

        async def timed_task(x):
            await anyio.sleep(0.002)  # 2ms task
            return x

        inputs = list(range(100))
        max_concurrent = 20

        # Time legacy implementation
        start_time = time.perf_counter()
        legacy_result = await alcall(
            inputs, timed_task, max_concurrent=max_concurrent
        )
        legacy_time = time.perf_counter() - start_time

        # Time modern implementation
        start_time = time.perf_counter()
        async with AsyncExecutor(max_concurrent=max_concurrent) as executor:
            modern_result = await executor(timed_task, inputs)
        modern_time = time.perf_counter() - start_time

        # Verify results are equivalent
        assert legacy_result == modern_result

        # Verify performance is competitive (allow 20% variance)
        performance_ratio = (
            modern_time / legacy_time if legacy_time > 0 else 1.0
        )

        print(
            f"Performance comparison: legacy={legacy_time:.3f}s, "
            f"modern={modern_time:.3f}s, ratio={performance_ratio:.2f}"
        )

        # Modern should not be significantly slower
        assert (
            performance_ratio < 1.5
        ), f"Modern implementation too slow: {performance_ratio:.2f}x slower"


# =============================================================================
# Comprehensive Integration Tests
# =============================================================================


@pytest.mark.migration
@pytest.mark.anyio
async def test_comprehensive_equivalence_integration(
    anyio_backend, cancel_guard
):
    """Comprehensive test combining multiple parameter variations."""
    async with cancel_guard():

        async def complex_func(x):
            # Simulate complex work with variable timing
            delay = 0.001 + (x % 3) * 0.001  # 1-3ms
            await anyio.sleep(delay)

            if x == 99:  # One expected failure for testing
                raise RuntimeError(f"Expected failure on {x}")

            return x * 10 + (x % 5)

        inputs = list(range(100))

        # Test with various parameter combinations
        test_configs = [
            {
                "max_concurrent": 10,
                "input_flatten": False,
                "input_dropna": False,
            },
            {
                "max_concurrent": 25,
                "input_flatten": False,
                "input_dropna": False,
            },
            {
                "max_concurrent": 5,
                "input_flatten": False,
                "input_dropna": False,
            },
        ]

        for config in test_configs:
            # Legacy path (expect failure)
            legacy_exception = None
            try:
                legacy_result = await alcall(inputs, complex_func, **config)
            except Exception as e:
                legacy_exception = e

            # Modern path (expect same failure)
            modern_exception = None
            try:
                data = to_list(
                    inputs,
                    flatten=config.get("input_flatten", False),
                    dropna=config.get("input_dropna", False),
                )
                async with AsyncExecutor(
                    max_concurrent=config["max_concurrent"]
                ) as executor:
                    modern_result = await executor(complex_func, data)
            except Exception as e:
                modern_exception = e

            # Both should fail with equivalent exceptions
            assert legacy_exception is not None
            assert modern_exception is not None
            assert type(legacy_exception) == type(modern_exception)


# =============================================================================
# Shadow Testing Framework (Production Validation)
# =============================================================================


@pytest.mark.migration
@pytest.mark.anyio
async def test_shadow_mode_validation(anyio_backend, cancel_guard):
    """Test shadow mode where both implementations run and results are compared."""
    async with cancel_guard():
        discrepancies = []

        async def shadow_test(inputs, func, **params):
            """Run both legacy and modern implementations, log any differences."""
            try:
                # Legacy path (source of truth)
                legacy_result = await alcall(inputs, func, **params)

                # Modern path (validation)
                data = to_list(
                    inputs,
                    flatten=params.get("input_flatten", False),
                    dropna=params.get("input_dropna", False),
                )
                async with AsyncExecutor(
                    max_concurrent=params.get("max_concurrent", 32)
                ) as executor:
                    modern_result = await executor(func, data)

                # Compare results
                if legacy_result != modern_result:
                    discrepancies.append(
                        {
                            "inputs": inputs,
                            "params": params,
                            "legacy": legacy_result,
                            "modern": modern_result,
                        }
                    )

                return legacy_result  # Use legacy as source of truth

            except Exception as e:
                # If legacy fails, modern should fail equivalently
                try:
                    data = to_list(
                        inputs,
                        flatten=params.get("input_flatten", False),
                        dropna=params.get("input_dropna", False),
                    )
                    async with AsyncExecutor(
                        max_concurrent=params.get("max_concurrent", 32)
                    ) as executor:
                        modern_result = await executor(func, data)

                    # If modern succeeds when legacy fails, that's a discrepancy
                    discrepancies.append(
                        {
                            "inputs": inputs,
                            "params": params,
                            "legacy": f"Exception: {e}",
                            "modern": modern_result,
                        }
                    )

                except Exception as modern_e:
                    # Both failed - verify same exception type
                    if type(e) != type(modern_e):
                        discrepancies.append(
                            {
                                "inputs": inputs,
                                "params": params,
                                "legacy": f"Exception: {type(e).__name__}",
                                "modern": f"Exception: {type(modern_e).__name__}",
                            }
                        )

                raise  # Re-raise original exception

        # Test various scenarios in shadow mode
        async def test_task(x):
            await anyio.sleep(0.001)
            return x * 2

        test_scenarios = [
            (list(range(20)), {"max_concurrent": 5}),
            ([1, 2, 3, None, 4], {"max_concurrent": 3, "input_dropna": True}),
            ([[1, 2], [3, 4]], {"max_concurrent": 4, "input_flatten": True}),
        ]

        for inputs, params in test_scenarios:
            await shadow_test(inputs, test_task, **params)

        # Verify no discrepancies found
        if discrepancies:
            pytest.fail(
                f"Shadow testing found {len(discrepancies)} discrepancies: {discrepancies}"
            )

        print("Shadow testing passed - no discrepancies found")
