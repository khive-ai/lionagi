# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Consolidated P0 Tests for lionagi v1 services hooks core functionality.

This file consolidates tests from:
- test_hook_registry.py (416 lines) - Execution patterns, timeout behavior
- test_hook_reliability.py (557 lines) - Registration lifecycle, stream chaining

Focus: Core hook functionality including the CRITICAL per-hook timeout flaw validation,
failure isolation with robust gather, and stream hook chaining behavior.

Preserves the essential PerHookSoftTimeout test that validates the hooks.py:582 timeout flaw.
"""

import time
from unittest.mock import patch
from uuid import uuid4

import anyio
import pytest

from lionagi.services.core import CallContext
from lionagi.services.endpoint import RequestModel
from lionagi.services.hooks import (
    HookEvent,
    HookRegistry,
    HookType,
    get_global_hooks,
    hook,
    stream_hook,
)


class TestRequest(RequestModel, frozen=True):
    """Simple test request."""

    content: str = "test"
    model: str = "test-model"


class TestHooksCore:
    """Core P0 tests for hook registry execution and reliability.

    Consolidates essential tests from hook registry and reliability files:
    - Critical timeout flaw validation (MUST PRESERVE)
    - Failure isolation with robust gather
    - Hook registration/deregistration lifecycle
    - Stream hook chaining and transformation
    - Context preservation and error handling
    """

    @pytest.mark.anyio
    async def test_per_hook_soft_timeout_critical_flaw_validation(self):
        """CRITICAL FLAW: hooks.py:582 - Incorrect timeout application in emit().

        Current flaw: HookRegistry.emit applies single fail_after timeout to entire
        group of hooks. One slow hook causes ALL hooks to be cancelled.

        Expected fix: Use per-hook move_on_after soft timeouts with robust gather
        to isolate hook failures.

        GIVEN H_Fast (10ms) and H_Slow (5s), AND registry timeout=1s
        WHEN registry.emit() is called
        THEN H_Fast completes. H_Slow is softly cancelled after 1s.
        AND the emit() call completes in approx 1s (not 5s).

        NOTE: This test is expected to FAIL with current implementation.
        """
        registry = HookRegistry()
        registry._timeout = 1.0  # Set 1 second timeout
        completion_times = []

        # H_Fast - completes quickly (10ms)
        async def hook_fast(event: HookEvent):
            await anyio.sleep(0.01)  # 10ms
            completion_times.append("fast_completed")

        # H_Slow - takes 5 seconds (should be cancelled)
        async def hook_slow(event: HookEvent):
            await anyio.sleep(5.0)  # 5 seconds
            completion_times.append("slow_completed")  # Should NOT reach this

        # Register both hooks
        registry.register(HookType.PRE_CALL, hook_fast)
        registry.register(HookType.PRE_CALL, hook_slow)

        # Create test event
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=10.0)
        event = HookEvent(
            hook_type=HookType.PRE_CALL,
            call_id=ctx.call_id,
            branch_id=ctx.branch_id,
            service_name="test-service",
        )

        # Mock logger to capture timeout warnings
        with patch("lionagi.services.hooks.logger") as mock_logger:
            # Measure emit time - CRITICAL validation
            start_time = time.perf_counter()
            await registry.emit(event)
            end_time = time.perf_counter()

            emit_duration = end_time - start_time

            # Allow any remaining tasks to complete
            await anyio.sleep(0.01)

            # CRITICAL: Verify emit completed in approximately 1 second
            # This validates that per-hook timeouts are working correctly
            assert (
                emit_duration < 1.5
            ), f"Emit should complete in ~1s with per-hook timeouts but took {emit_duration:.3f}s"
            assert (
                emit_duration > 0.9
            ), f"Emit should take close to timeout duration but took {emit_duration:.3f}s"

            # Verify fast hook completed
            assert "fast_completed" in completion_times, "Fast hook should complete successfully"

            # CRITICAL: Verify slow hook was cancelled (per-hook timeout)
            assert (
                "slow_completed" not in completion_times
            ), "Slow hook should be cancelled by per-hook timeout"

            # Verify timeout was logged
            mock_logger.warning.assert_called()
            warning_message = str(mock_logger.warning.call_args)
            assert "timed out" in warning_message.lower()

    @pytest.mark.anyio
    async def test_failure_isolation_robust_gather_critical(self):
        """CRITICAL: Hook failure isolation using robust gather.

        GIVEN hooks H1 (succeeds), H2 (fails), H3 (succeeds)
        WHEN registry.emit() is called (using V1 robust gather)
        THEN H1 and H3 must complete successfully. H2's failure is logged but isolated.
        AND the emit() call must complete resiliently (not fail entirely).

        Validates that hook failures don't break the main service execution.
        """
        registry = HookRegistry()
        successful_completions = []

        # H1 - succeeds
        async def hook_h1_success(event: HookEvent):
            successful_completions.append("h1")

        # H2 - fails with exception
        async def hook_h2_fails(event: HookEvent):
            successful_completions.append("h2_started")  # Should reach this
            raise ValueError("Intentional hook failure for robustness test")

        # H3 - succeeds
        async def hook_h3_success(event: HookEvent):
            successful_completions.append("h3")

        # Register all hooks for PRE_CALL
        registry.register(HookType.PRE_CALL, hook_h1_success)
        registry.register(HookType.PRE_CALL, hook_h2_fails)
        registry.register(HookType.PRE_CALL, hook_h3_success)

        # Create test event
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)
        event = HookEvent(
            hook_type=HookType.PRE_CALL,
            call_id=ctx.call_id,
            branch_id=ctx.branch_id,
            service_name="test-service",
        )

        # Mock logger to capture error logs
        with patch("lionagi.services.hooks.logger") as mock_logger:
            # Emit should complete without raising an exception
            try:
                await registry.emit(event)
            except Exception as e:
                pytest.fail(f"HookRegistry.emit should not raise exception but got: {e}")

            # Allow hooks to complete
            await anyio.sleep(0.01)

            # CRITICAL: Verify successful hooks completed despite failure
            assert "h1" in successful_completions, "Hook H1 should complete successfully"
            assert "h3" in successful_completions, "Hook H3 should complete successfully"
            assert "h2_started" in successful_completions, "Hook H2 should start before failing"

            # Verify error was logged (not raised) - robust gather behavior
            mock_logger.error.assert_called()
            error_call_args = str(mock_logger.error.call_args)
            assert (
                "hook function failed" in error_call_args.lower()
                or "intentional" in error_call_args.lower()
            )

    @pytest.mark.anyio
    async def test_hook_registration_and_deregistration_lifecycle(self):
        """Hook registration/deregistration lifecycle management.

        Validates basic hook registry operations work correctly.
        """
        registry = HookRegistry()
        execution_log = []

        async def test_hook_1(event: HookEvent):
            execution_log.append("hook_1")

        async def test_hook_2(event: HookEvent):
            execution_log.append("hook_2")

        # Initially no hooks
        assert not registry.has_hooks(HookType.PRE_CALL), "Should have no PRE_CALL hooks initially"

        # Register first hook
        registry.register(HookType.PRE_CALL, test_hook_1)
        assert registry.has_hooks(
            HookType.PRE_CALL
        ), "Should have PRE_CALL hooks after registration"

        # Register second hook to same type
        registry.register(HookType.PRE_CALL, test_hook_2)

        # Test emission executes both hooks
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)
        event = HookEvent(
            hook_type=HookType.PRE_CALL,
            call_id=ctx.call_id,
            branch_id=ctx.branch_id,
            service_name="test-service",
        )

        await registry.emit(event)
        await anyio.sleep(0.01)

        assert len(execution_log) == 2, "Both hooks should execute"
        assert "hook_1" in execution_log, "Hook 1 should execute"
        assert "hook_2" in execution_log, "Hook 2 should execute"

        # Unregister first hook
        registry.unregister(HookType.PRE_CALL, test_hook_1)
        execution_log.clear()

        await registry.emit(event)
        await anyio.sleep(0.01)

        assert len(execution_log) == 1, "Only remaining hook should execute"
        assert "hook_2" in execution_log, "Hook 2 should still execute"
        assert "hook_1" not in execution_log, "Hook 1 should not execute after unregistration"

    @pytest.mark.anyio
    async def test_stream_hook_chaining_and_transformation(self):
        """Stream hook chaining with chunk transformation.

        Validates that stream hooks can transform chunks properly and handle
        failures in the chain. Tests the chain properly, failure stops chain
        and logs error as specified in TDD.
        """
        registry = HookRegistry()
        transformation_log = []

        # First transform hook - adds metadata
        async def transform_hook_1(event, chunk_data):
            transformation_log.append("transform_1")
            chunk_data["metadata"] = "added_by_hook_1"
            return chunk_data

        # Second transform hook - modifies content
        async def transform_hook_2(event, chunk_data):
            transformation_log.append("transform_2")
            chunk_data["transformed"] = True
            return chunk_data

        # Third transform hook - fails intentionally
        async def transform_hook_3_fails(event, chunk_data):
            transformation_log.append("transform_3_started")
            raise ValueError("Intentional transform failure")

        # Fourth transform hook - should NOT execute due to failure in chain
        async def transform_hook_4_should_not_run(event, chunk_data):
            transformation_log.append("transform_4")  # Should NOT appear
            return chunk_data

        # Register stream hooks in order
        registry.register_stream_hook(HookType.STREAM_CHUNK, transform_hook_1)
        registry.register_stream_hook(HookType.STREAM_CHUNK, transform_hook_2)
        registry.register_stream_hook(HookType.STREAM_CHUNK, transform_hook_3_fails)
        registry.register_stream_hook(HookType.STREAM_CHUNK, transform_hook_4_should_not_run)

        # Create test chunk data
        original_chunk = {"chunk": 0, "content": "test data"}

        # Create a stream event
        call_id = uuid4()
        branch_id = uuid4()
        stream_event = HookEvent(
            hook_type=HookType.STREAM_CHUNK,
            call_id=call_id,
            branch_id=branch_id,
            service_name="test_service",
            context=CallContext.new(branch_id=branch_id),
        )

        with patch("lionagi.services.hooks.logger") as mock_logger:
            # Process chunk through stream hooks
            processed_chunk = await registry.emit_stream_chunk(stream_event, original_chunk)

            # Allow async operations to complete
            await anyio.sleep(0.01)

            # Validate transformation chain worked until failure
            assert "transform_1" in transformation_log, "First transform should execute"
            assert "transform_2" in transformation_log, "Second transform should execute"
            assert "transform_3_started" in transformation_log, "Third transform should start"
            assert (
                "transform_4" not in transformation_log
            ), "Fourth transform should NOT execute after failure"

            # Validate transformations applied before failure
            assert (
                processed_chunk.get("metadata") == "added_by_hook_1"
            ), "First transformation should be applied"
            assert (
                processed_chunk.get("transformed") is True
            ), "Second transformation should be applied"

            # Verify error was logged for failed transform
            mock_logger.error.assert_called()

    @pytest.mark.anyio
    async def test_hook_context_preservation_call_id_branch_id(self):
        """Hook context preservation with call_id and branch_id.

        Validates that hook events preserve CallContext data correctly
        and timing information is captured properly.
        """
        registry = HookRegistry()
        captured_contexts = []

        async def context_capture_hook(event: HookEvent):
            captured_contexts.append(
                {
                    "call_id": event.call_id,
                    "branch_id": event.branch_id,
                    "service_name": event.service_name,
                    "hook_type": event.hook_type,
                    "timestamp": getattr(event, "timestamp", None),
                }
            )

        registry.register(HookType.POST_CALL, context_capture_hook)

        # Create context with specific IDs
        original_branch_id = uuid4()
        original_call_id = uuid4()

        ctx = CallContext(
            call_id=original_call_id,
            branch_id=original_branch_id,
            deadline_s=None,
            capabilities=set(),
            attrs={},
        )

        event = HookEvent(
            hook_type=HookType.POST_CALL,
            call_id=ctx.call_id,
            branch_id=ctx.branch_id,
            service_name="context-test-service",
        )

        await registry.emit(event)
        await anyio.sleep(0.01)

        # Validate context preservation
        assert len(captured_contexts) == 1, "Should capture one context"
        captured = captured_contexts[0]

        assert captured["call_id"] == original_call_id, "Call ID should be preserved"
        assert captured["branch_id"] == original_branch_id, "Branch ID should be preserved"
        assert (
            captured["service_name"] == "context-test-service"
        ), "Service name should be preserved"
        assert captured["hook_type"] == HookType.POST_CALL, "Hook type should be preserved"

    @pytest.mark.anyio
    async def test_error_hook_triggering_and_context_capture(self):
        """Error hook triggering with proper error context capture.

        Validates that CALL_ERROR and STREAM_ERROR hooks are triggered properly
        and capture comprehensive error metadata for debugging.
        """
        registry = HookRegistry()
        error_events = []

        async def error_capture_hook(event: HookEvent):
            error_events.append(
                {
                    "hook_type": event.hook_type,
                    "call_id": event.call_id,
                    "error_data": getattr(event, "error", None),
                    "context": getattr(event, "context", None),
                }
            )

        # Register error hooks
        registry.register(HookType.CALL_ERROR, error_capture_hook)
        registry.register(HookType.STREAM_ERROR, error_capture_hook)

        ctx = CallContext.new(branch_id=uuid4())

        # Test CALL_ERROR event
        call_error_event = HookEvent(
            hook_type=HookType.CALL_ERROR,
            call_id=ctx.call_id,
            branch_id=ctx.branch_id,
            service_name="error-test-service",
            error="Test call error",
            context={"error_type": "service_failure"},
        )

        await registry.emit(call_error_event)

        # Test STREAM_ERROR event
        stream_error_event = HookEvent(
            hook_type=HookType.STREAM_ERROR,
            call_id=ctx.call_id,
            branch_id=ctx.branch_id,
            service_name="error-test-service",
            error="Test stream error",
            context={"error_type": "stream_failure", "chunk_index": 5},
        )

        await registry.emit(stream_error_event)
        await anyio.sleep(0.01)

        # Validate error events were captured
        assert len(error_events) == 2, "Should capture both error events"

        call_error_captured = next(e for e in error_events if e["hook_type"] == HookType.CALL_ERROR)
        stream_error_captured = next(
            e for e in error_events if e["hook_type"] == HookType.STREAM_ERROR
        )

        assert (
            call_error_captured["error_data"] == "Test call error"
        ), "Call error data should be preserved"
        assert (
            stream_error_captured["error_data"] == "Test stream error"
        ), "Stream error data should be preserved"
        assert (
            stream_error_captured["context"]["chunk_index"] == 5
        ), "Stream error context should include chunk info"

    @pytest.mark.anyio
    async def test_global_hook_decorator_and_registry_integration(self):
        """Global hook decorator integration with registry.

        Tests the @hook and @stream_hook decorators work correctly with
        the global registry and can be accessed properly.
        """
        # Clear global hooks for clean test
        global_registry = get_global_hooks()
        original_hooks = global_registry._hooks.copy()
        global_registry._hooks.clear()

        try:
            execution_log = []

            # Use decorator to register global hook
            @hook(HookType.PRE_CALL)
            async def global_pre_call(event):
                execution_log.append("global_pre_call")

            @stream_hook(HookType.STREAM_CHUNK)
            async def global_stream_transform(event, chunk_data):
                execution_log.append("global_stream_transform")
                chunk_data["global_transform"] = True
                return chunk_data

            # Verify hooks were registered globally
            assert global_registry.has_hooks(
                HookType.PRE_CALL
            ), "Global PRE_CALL hook should be registered"
            assert global_registry.has_hooks(
                HookType.STREAM_CHUNK
            ), "Global STREAM_CHUNK hook should be registered"

            # Test global hook execution
            ctx = CallContext.new(branch_id=uuid4())
            pre_call_event = HookEvent(
                hook_type=HookType.PRE_CALL,
                call_id=ctx.call_id,
                branch_id=ctx.branch_id,
                service_name="global-test",
            )

            await global_registry.emit(pre_call_event)

            # Test global stream hook
            test_chunk = {"chunk": 0, "data": "test"}
            stream_event = HookEvent(
                hook_type=HookType.STREAM_CHUNK,
                call_id=uuid4(),
                branch_id=uuid4(),
                service_name="test_service",
                context=CallContext.new(branch_id=uuid4()),
            )
            transformed_chunk = await global_registry.emit_stream_chunk(stream_event, test_chunk)

            await anyio.sleep(0.01)

            # Validate global hooks executed
            assert "global_pre_call" in execution_log, "Global pre-call hook should execute"
            assert (
                "global_stream_transform" in execution_log
            ), "Global stream transform should execute"
            assert (
                transformed_chunk.get("global_transform") is True
            ), "Global transform should modify chunk"

        finally:
            # Restore original global hooks
            global_registry._hooks = original_hooks
