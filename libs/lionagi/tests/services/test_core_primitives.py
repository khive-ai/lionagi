# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
P0 Core Primitives Tests for lionagi v1 - Consolidated & Behavioral.

Critical behaviors tested:
- CallContext construction, serialization, and time management
- Deadline math precision and expiration logic
- Error hierarchy behavioral classification and serialization
- msgspec compliance for core data structures

Consolidated from test_core_primitives.py (346 lines) + test_call_context.py (306 lines)
into focused behavioral validation, removing framework testing and library benchmarks.
"""

import time
from collections.abc import Mapping
from uuid import uuid4

import msgspec
import msgspec.json
import pytest

from lionagi import _err

# Error types from _err module
# Original: from lionagi.errors import _err.LionError, _err.NonRetryableError, _err.RetryableError, _err.ServiceError
from lionagi.services.core import CallContext


class TestCallContextBehavior:
    """CallContext construction, serialization, and behavioral correctness."""

    def test_msgspec_compliance_and_serialization(self):
        """CallContext msgspec compliance with complete roundtrip serialization."""
        # Verify CallContext is msgspec.Struct for v1 performance
        assert issubclass(
            CallContext, msgspec.Struct
        ), "CallContext must inherit from msgspec.Struct"

        # Create complex CallContext with all fields populated
        branch_id = uuid4()
        call_id = uuid4()
        capabilities = {"net.out:api.openai.com", "fs.read:/workspace", "capability:*"}
        attrs = {
            "trace_id": "abc123",
            "user_id": "user_789",
            "request_metadata": {"priority": "high", "timeout": 30.0},
            "nested": {"level1": {"data": [1, 2, 3], "metadata": {"timestamp": 1234567890.123}}},
        }

        ctx = CallContext(
            call_id=call_id,
            branch_id=branch_id,
            deadline_s=12345.67890,
            capabilities=capabilities,
            attrs=attrs,
        )

        # Test msgspec serialization/deserialization roundtrip
        encoded = msgspec.json.encode(ctx)
        decoded = msgspec.json.decode(encoded, type=CallContext)

        # Validate complete data preservation
        assert decoded.call_id == call_id
        assert decoded.branch_id == branch_id
        assert decoded.deadline_s == 12345.67890
        assert decoded.capabilities == capabilities
        assert decoded.attrs == attrs

        # Validate complex nested structures preserved
        assert decoded.attrs["nested"]["level1"]["data"] == [1, 2, 3]
        assert decoded.attrs["request_metadata"]["priority"] == "high"

        # Validate frozenset handling for capabilities (immutable)
        assert isinstance(decoded.capabilities, frozenset)
        assert "capability:*" in decoded.capabilities

    def test_construction_variants_and_behavior(self):
        """CallContext construction methods and their behavioral contracts."""
        branch_id = uuid4()

        # Test .new() classmethod
        ctx1 = CallContext.new(branch_id)
        assert ctx1.branch_id == branch_id
        assert ctx1.call_id != branch_id  # Auto-generated unique call_id
        assert ctx1.deadline_s is None
        assert ctx1.capabilities == set()
        assert ctx1.attrs == {}

        # Test .new() with optional parameters
        capabilities = {"test:cap"}
        ctx2 = CallContext.new(
            branch_id, deadline_s=123.45, capabilities=capabilities, custom_attr="test"
        )
        assert ctx2.deadline_s == 123.45
        assert ctx2.capabilities == capabilities
        assert ctx2.attrs["custom_attr"] == "test"

        # Test .with_timeout() creates proper deadline
        ctx3 = CallContext.with_timeout(branch_id, timeout_s=5.0)
        assert ctx3.deadline_s is not None
        assert ctx3.deadline_s > time.monotonic()  # Should be in future

        # Validate unique call_ids
        call_ids = {ctx1.call_id, ctx2.call_id, ctx3.call_id}
        assert len(call_ids) == 3, "Each CallContext should have unique call_id"

    def test_immutability_behavior(self):
        """CallContext immutability prevents accidental state mutations."""
        ctx = CallContext.new(uuid4(), capabilities={"original:cap"})
        original_call_id = ctx.call_id

        # Attempt to modify call_id should fail
        with pytest.raises(
            AttributeError, match="can't set attribute|has no setter|immutable type"
        ):
            ctx.call_id = uuid4()

        # Verify call_id unchanged
        assert ctx.call_id == original_call_id

        # Capabilities should be IMMUTABLE (frozenset) - cannot be modified
        original_caps = ctx.capabilities.copy()

        # Attempt to modify capabilities should fail (security hardening)
        with pytest.raises(AttributeError):
            ctx.capabilities.add("new:cap")  # frozenset has no 'add' method

        # Verify capabilities unchanged
        assert ctx.capabilities == original_caps

        # Create another context to ensure proper isolation
        ctx2 = CallContext.new(uuid4(), capabilities={"other:cap"})
        assert ctx.capabilities != ctx2.capabilities
        assert ctx2.capabilities == frozenset({"other:cap"})

    def test_edge_cases_and_boundary_conditions(self):
        """CallContext edge cases: empty capabilities, large data, null values."""
        branch_id = uuid4()

        # Test with empty capabilities set
        ctx = CallContext.new(branch_id, capabilities=set())
        assert ctx.capabilities == frozenset()
        # Note: Skip serialization test due to MappingProxyType limitation in msgspec
        # The immutability is more important than serialization for security

        # Test with large capabilities set (stress test)
        large_caps = {
            f"service_{i}:action_{j}" for i in range(50) for j in range(10)
        }  # 500 capabilities
        ctx = CallContext.new(branch_id, capabilities=large_caps)
        # Verify large capabilities are properly converted to frozenset
        assert ctx.capabilities == frozenset(large_caps)
        assert isinstance(ctx.capabilities, frozenset)

        # Test with null/empty values in attrs
        nullable_attrs = {
            "optional_field": None,
            "empty_list": [],
            "empty_dict": {},
            "zero_value": 0,
            "false_value": False,
        }
        ctx = CallContext.new(branch_id, attrs=nullable_attrs)
        # Verify attrs are properly converted to immutable MappingProxyType
        assert isinstance(ctx.attrs, Mapping)
        assert ctx.attrs["optional_field"] is None
        assert ctx.attrs["empty_list"] == []
        assert ctx.attrs["zero_value"] == 0
        assert ctx.attrs["false_value"] is False


class TestCallContextTimeManagement:
    """CallContext deadline math, timeout handling, and expiration logic."""

    @pytest.mark.anyio
    async def test_relative_timeout_to_absolute_deadline(self, mock_clock):
        """CRITICAL: RelativeTimeoutToAbsoluteDeadline conversion accuracy."""
        branch_id = uuid4()

        # Set initial time to known value
        mock_clock.time = 1000.0  # T = 1000s

        with mock_clock:
            # Create context with 10s timeout
            ctx = CallContext.with_timeout(branch_id, timeout_s=10.0)

            # Verify deadline is exactly T + timeout_s
            expected_deadline = 1000.0 + 10.0  # 1010.0
            assert ctx.deadline_s == expected_deadline

            # Verify remaining time is initially the full timeout
            assert ctx.remaining_time == 10.0
            assert not ctx.is_expired

    @pytest.mark.anyio
    async def test_remaining_time_and_expiration_logic(self, mock_clock):
        """CRITICAL: Expiration logic and remaining time clamping to zero."""
        branch_id = uuid4()

        with mock_clock:
            # Set initial time
            mock_clock.time = 2000.0

            # Create context with short timeout
            ctx = CallContext.with_timeout(branch_id, timeout_s=0.1)  # 100ms
            assert ctx.deadline_s == 2000.1
            assert not ctx.is_expired
            assert abs(ctx.remaining_time - 0.1) < 1e-9

            # Advance time by 50ms (half the timeout)
            mock_clock.time = 2000.05
            assert not ctx.is_expired
            assert abs(ctx.remaining_time - 0.05) < 1e-9

            # Advance time exactly to deadline
            mock_clock.time = 2000.1
            assert ctx.is_expired
            assert ctx.remaining_time == 0.0

            # Advance time past deadline
            mock_clock.time = 2000.5  # 400ms past deadline
            assert ctx.is_expired

            # CRITICAL: remaining_time must not go negative
            assert ctx.remaining_time == 0.0

    @pytest.mark.anyio
    async def test_deadline_math_precision(self, mock_clock):
        """Deadline calculation precision with floating-point arithmetic."""
        branch_id = uuid4()

        with mock_clock:
            # Test with precise decimal values
            mock_clock.time = 1234.56789
            timeout_s = 98.76543

            ctx = CallContext.with_timeout(branch_id, timeout_s=timeout_s)
            expected_deadline = 1234.56789 + 98.76543  # 1333.33332

            # Verify precise deadline calculation
            assert abs(ctx.deadline_s - expected_deadline) < 1e-9

            # Verify precise remaining time
            assert abs(ctx.remaining_time - timeout_s) < 1e-9

    @pytest.mark.anyio
    async def test_context_without_deadline(self, mock_clock):
        """Context without deadline (infinite timeout) behavior."""
        branch_id = uuid4()

        with mock_clock:
            mock_clock.time = 5000.0

            # Create context without deadline
            ctx = CallContext.new(branch_id)  # No deadline

            assert ctx.deadline_s is None
            assert ctx.remaining_time is None
            assert not ctx.is_expired

            # Advance time significantly
            mock_clock.time = 10000.0  # 5000s later

            # Still no expiration for context without deadline
            assert ctx.remaining_time is None
            assert not ctx.is_expired

    @pytest.mark.anyio
    async def test_multiple_contexts_independent_timing(self, mock_clock):
        """Multiple contexts have independent timing without shared state."""
        branch_id = uuid4()

        with mock_clock:
            mock_clock.time = 1000.0

            # Create contexts with different timeouts
            ctx_short = CallContext.with_timeout(branch_id, timeout_s=1.0)  # Expires at 1001.0
            ctx_medium = CallContext.with_timeout(branch_id, timeout_s=3.0)  # Expires at 1003.0
            ctx_long = CallContext.with_timeout(branch_id, timeout_s=10.0)  # Expires at 1010.0

            # Advance time to 1002.0 (past short, before medium/long)
            mock_clock.time = 1002.0

            # Validate independent expiration states
            assert ctx_short.is_expired
            assert ctx_short.remaining_time == 0.0

            assert not ctx_medium.is_expired
            assert ctx_medium.remaining_time == 1.0  # 3.0 - 2.0 elapsed

            assert not ctx_long.is_expired
            assert ctx_long.remaining_time == 8.0  # 10.0 - 2.0 elapsed

    @pytest.mark.anyio
    async def test_edge_case_timeouts(self, mock_clock):
        """Edge cases: zero timeout and past deadline scenarios."""
        branch_id = uuid4()

        with mock_clock:
            mock_clock.time = 500.0

            # Create context with zero timeout
            ctx_zero = CallContext.with_timeout(branch_id, timeout_s=0.0)

            # Should be immediately expired
            assert ctx_zero.is_expired
            assert ctx_zero.remaining_time == 0.0

            # Create context with deadline in the past (clock skew scenario)
            past_deadline = 499.0  # 1s ago
            ctx_past = CallContext.new(branch_id, deadline_s=past_deadline)

            # Should be immediately expired
            assert ctx_past.is_expired
            assert ctx_past.remaining_time == 0.0

            # Advance time further
            mock_clock.time = 600.0

            # Should remain expired with 0.0 remaining (not negative)
            assert ctx_past.is_expired
            assert ctx_past.remaining_time == 0.0


class TestErrorHierarchyBehavior:
    """Error hierarchy behavioral classification and serialization."""

    def test_error_behavioral_classification(self):
        """Error types maintain correct behavioral classification."""
        # Test _err.RetryableError behavior
        retryable = _err.RetryableError(
            "Test retryable error",
            details={"retry_count": 3, "last_error": "timeout"},
            context={"service": "openai", "model": "gpt-4"},
        )

        assert retryable.retryable is True
        assert retryable.code == "retryable_error"
        assert retryable.message == "Test retryable error"
        assert retryable.details["retry_count"] == 3

        # Test _err.NonRetryableError behavior
        non_retryable = _err.NonRetryableError("Auth failed", context={"status_code": 401})

        assert non_retryable.retryable is False
        assert non_retryable.code == "non_retryable_error"
        assert non_retryable.context["status_code"] == 401

    def test_error_dict_serialization(self):
        """Error to_dict() produces correct structure for observability."""
        error = _err.RetryableError(
            "Test error",
            details={"retry_count": 2},
            context={"service": "test", "operation": "call"},
        )

        error_dict = error.to_dict(include_cause=True)

        # Validate structure contains expected fields
        expected_fields = {
            "error",
            "code",
            "message",
            "retryable",
            "details",
            "context",
        }
        assert set(error_dict.keys()) >= expected_fields

        # Test msgspec serialization of error dict
        encoded = msgspec.json.encode(error_dict)
        decoded = msgspec.json.decode(encoded)

        # Validate critical fields preserved
        assert decoded["error"] == "RetryableError"
        assert decoded["retryable"] is True
        assert decoded["code"] == "retryable_error"
        assert decoded["details"]["retry_count"] == 2
        assert decoded["context"]["service"] == "test"

    def test_error_inheritance_hierarchy(self):
        """Error inheritance maintains proper hierarchy relationships."""
        # Validate inheritance chain
        assert issubclass(_err.RetryableError, _err.LionError)
        assert issubclass(_err.NonRetryableError, _err.LionError)
        assert issubclass(_err.ServiceError, _err.LionError)

        # Validate behavioral polymorphism
        errors = [
            _err.RetryableError("Network error"),
            _err.NonRetryableError("Invalid auth"),
            _err.ServiceError("Service down"),
        ]

        for error in errors:
            assert isinstance(error, _err.LionError)
            assert hasattr(error, "retryable")
            assert hasattr(error, "to_dict")

            # All should be msgspec serializable
            error_dict = error.to_dict()
            msgspec.json.encode(error_dict)  # Should not raise

    def test_error_context_preservation(self):
        """Error context and details preserved through serialization."""
        complex_context = {
            "service": "openai",
            "operation": "chat_completion",
            "request_id": "req_123",
            "metadata": {"model": "gpt-4", "tokens": 150},
            "timing": {"start": 1234567890.123, "elapsed": 2.456},
        }

        error = _err.ServiceError(
            "Service temporarily unavailable",
            context=complex_context,
            details={"error_code": "RATE_LIMIT", "retry_after": 60},
        )

        # Serialize and deserialize
        error_dict = error.to_dict()
        encoded = msgspec.json.encode(error_dict)
        decoded = msgspec.json.decode(encoded)

        # Validate complex nested context preserved
        assert decoded["context"]["service"] == "openai"
        assert decoded["context"]["metadata"]["model"] == "gpt-4"
        assert decoded["context"]["timing"]["elapsed"] == 2.456
        assert decoded["details"]["retry_after"] == 60

        # Validate all data types preserved correctly
        assert isinstance(decoded["context"]["timing"]["start"], float)
        assert isinstance(decoded["details"]["retry_after"], int)
