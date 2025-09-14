# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""Performance tests for V1 Observable Protocol.

Validates that isinstance() checks and ID normalization utilities meet
performance requirements for production use.
"""

import time
from uuid import UUID

import pytest

from lionagi.protocols.generic.element import Element
from lionagi.protocols.ids import canonical_id, to_uuid
from lionagi.protocols.types import Observable


class TestProtocolPerformance:
    """Test performance characteristics of V1 Observable Protocol."""

    def test_protocol_isinstance_performance(self):
        """Ensure isinstance() checks against Observable Protocol are fast."""
        ITERATIONS = 100_000
        elements = [Element() for _ in range(ITERATIONS)]

        # Benchmark the isinstance checks
        start_time = time.perf_counter()
        for e in elements:
            isinstance(e, Observable)
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        time_per_check_ns = (elapsed_time / ITERATIONS) * 1e9

        # Threshold: 1500ms for 100k checks (generous for CI environments)
        ACCEPTABLE_TIME = 1.5

        print(
            f"\nProtocol isinstance Performance: {ITERATIONS} checks took {elapsed_time:.4f}s. ({time_per_check_ns:.2f} ns/check)"
        )

        assert (
            elapsed_time < ACCEPTABLE_TIME
        ), f"isinstance checks exceeded threshold: {elapsed_time:.4f}s > {ACCEPTABLE_TIME}s"

    def test_canonical_id_performance(self):
        """Test performance of canonical_id utility function."""
        ITERATIONS = 50_000
        elements = [Element() for _ in range(ITERATIONS)]

        # Benchmark canonical_id calls
        start_time = time.perf_counter()
        for e in elements:
            canonical_id(e)
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        time_per_call_us = (elapsed_time / ITERATIONS) * 1e6

        # Threshold: 50ms for 50k calls (1us per call average)
        ACCEPTABLE_TIME = 0.05

        print(
            f"\ncanonical_id Performance: {ITERATIONS} calls took {elapsed_time:.4f}s. ({time_per_call_us:.2f} μs/call)"
        )

        assert (
            elapsed_time < ACCEPTABLE_TIME
        ), f"canonical_id calls exceeded threshold: {elapsed_time:.4f}s > {ACCEPTABLE_TIME}s"

    def test_to_uuid_performance(self):
        """Test performance of to_uuid utility function."""
        ITERATIONS = 50_000
        elements = [Element() for _ in range(ITERATIONS)]

        # Test different input types for comprehensive benchmark
        id_types = [e.id for e in elements]  # IDType objects
        uuids = [e.id._id for e in elements]  # UUID objects
        strings = [str(e.id) for e in elements]  # String representations

        test_cases = [
            ("Element", elements),
            ("IDType", id_types),
            ("UUID", uuids),
            ("String", strings),
        ]

        for input_type, inputs in test_cases:
            start_time = time.perf_counter()
            for input_val in inputs:
                to_uuid(input_val)
            end_time = time.perf_counter()

            elapsed_time = end_time - start_time
            time_per_call_us = (elapsed_time / len(inputs)) * 1e6

            # Threshold varies by input type complexity (generous for CI)
            if input_type == "UUID":
                max_time = 0.05  # Should be fast for UUID passthrough
            elif input_type in ["Element", "IDType"]:
                max_time = 0.1  # Should be fast for optimized direct access
            else:  # String
                max_time = 0.15  # More expensive due to parsing

            print(
                f"\nto_uuid({input_type}) Performance: {len(inputs)} calls took {elapsed_time:.4f}s. ({time_per_call_us:.2f} μs/call)"
            )

            assert (
                elapsed_time < max_time
            ), f"to_uuid({input_type}) exceeded threshold: {elapsed_time:.4f}s > {max_time}s"

    def test_optimized_vs_string_conversion_performance(self):
        """Compare optimized direct access vs string conversion performance."""
        ITERATIONS = 20_000
        elements = [Element() for _ in range(ITERATIONS)]

        # Optimized approach (direct _id access)
        start_time = time.perf_counter()
        for e in elements:
            e.id._id  # Direct access
        optimized_time = time.perf_counter() - start_time

        # String conversion approach (what we avoided)
        start_time = time.perf_counter()
        for e in elements:
            UUID(str(e.id))  # String conversion
        string_conversion_time = time.perf_counter() - start_time

        # Optimized should be significantly faster
        speedup = string_conversion_time / optimized_time

        print(f"\nOptimization Comparison:")
        print(f"Direct access: {optimized_time:.4f}s")
        print(f"String conversion: {string_conversion_time:.4f}s")
        print(f"Speedup: {speedup:.1f}x")

        assert (
            speedup > 2.0
        ), f"Optimized approach should be at least 2x faster, got {speedup:.1f}x"

        # Both should still be reasonably fast
        assert optimized_time < 0.02, "Optimized approach should be very fast"
        assert (
            string_conversion_time < 0.1
        ), "String conversion should be reasonable"


class TestScalabilityCharacteristics:
    """Test behavior under various scales and edge cases."""

    def test_large_scale_protocol_checks(self):
        """Test protocol behavior with large numbers of objects."""
        LARGE_SCALE = 500_000

        # Create elements in batches to avoid memory issues
        batch_size = 10_000
        total_time = 0

        for batch_start in range(0, LARGE_SCALE, batch_size):
            batch_end = min(batch_start + batch_size, LARGE_SCALE)
            batch_elements = [
                Element() for _ in range(batch_end - batch_start)
            ]

            start_time = time.perf_counter()
            for e in batch_elements:
                isinstance(e, Observable)
            total_time += time.perf_counter() - start_time

            # Clean up batch
            del batch_elements

        avg_time_per_check_ns = (total_time / LARGE_SCALE) * 1e9

        print(
            f"\nLarge scale test: {LARGE_SCALE} checks took {total_time:.4f}s. ({avg_time_per_check_ns:.2f} ns/check)"
        )

        # Should scale linearly and remain reasonable (generous for CI)
        assert (
            total_time < 5.0
        ), f"Large scale test took too long: {total_time:.4f}s"
        # 10μs per check is reasonable for Protocol isinstance() checks
        assert (
            avg_time_per_check_ns < 10000
        ), f"Per-check time too high at scale: {avg_time_per_check_ns:.2f} ns"

    def test_memory_efficiency(self):
        """Test that protocol checks don't create excessive temporary objects."""
        import gc
        import sys

        # Force garbage collection and get baseline
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Perform many protocol checks
        elements = [Element() for _ in range(1000)]
        for e in elements:
            isinstance(e, Observable)
            canonical_id(e)
            to_uuid(e.id)

        # Force garbage collection and check object growth
        del elements
        gc.collect()
        final_objects = len(gc.get_objects())

        object_growth = final_objects - initial_objects

        print(f"\nMemory efficiency: Object count grew by {object_growth}")

        # Should not create excessive temporary objects (allow some reasonable growth)
        assert (
            object_growth < 100
        ), f"Protocol operations created too many objects: {object_growth}"
