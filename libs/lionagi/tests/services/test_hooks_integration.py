"""Test suite for V1_Hooks_Integration - Comprehensive P0 Tests.

Focus: HookedMiddleware integration with service pipeline, observability event
emission timing and content, and hook performance impact measurement.
Validates end-to-end integration behavior and production readiness.
"""

import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import anyio
import msgspec
import pytest

from lionagi.services.core import CallContext, Service
from lionagi.services.endpoint import RequestModel
from lionagi.services.hooks import HookedMiddleware, HookEvent, HookRegistry, HookType


class DummyRequest(RequestModel):
    """Dummy request for testing."""

    model: str = "test-model"
    messages: list[dict[str, str]] = msgspec.field(
        default_factory=lambda: [{"role": "user", "content": "test"}]
    )


class MockService(Service[DummyRequest, dict, dict]):
    """Mock service for testing middleware integration."""

    name: str = "mock-service"

    def __init__(self, delay: float = 0.0, should_fail: bool = False):
        self.delay = delay
        self.should_fail = should_fail
        self.call_count = 0
        self.stream_count = 0

    async def call(self, req: DummyRequest, *, ctx: CallContext) -> dict:
        self.call_count += 1
        if self.delay > 0:
            await anyio.sleep(self.delay)
        if self.should_fail:
            raise ValueError("Mock service failure")
        return {"response": f"processed {req.model}", "call_id": str(ctx.call_id)}

    async def stream(self, req: DummyRequest, *, ctx: CallContext) -> AsyncIterator[dict]:
        self.stream_count += 1
        if self.should_fail:
            raise ValueError("Mock streaming failure")
        for i in range(3):
            if self.delay > 0:
                await anyio.sleep(self.delay)
            yield {"chunk": i, "data": f"chunk_{i}", "call_id": str(ctx.call_id)}


class TestV1HooksIntegration:
    """TestSuite: V1_Hooks_Integration - End-to-end integration validation."""

    @pytest.mark.anyio
    async def test_hooked_middleware_integration_with_service_pipeline(self):
        """Test: HookedMiddleware integration with service pipeline.

        CRITICAL: Validates end-to-end hook execution around service calls
        including PRE_CALL, POST_CALL, and proper data flow.
        """
        registry = HookRegistry()
        middleware = HookedMiddleware(registry)
        service = MockService()

        execution_log = []

        # Pre-call hook
        async def pre_call_hook(event: HookEvent):
            execution_log.append(
                {
                    "type": "pre_call",
                    "call_id": str(event.call_id),
                    "service": event.service_name,
                    "request_model": event.request.model if event.request else None,
                }
            )

        # Post-call hook
        async def post_call_hook(event: HookEvent):
            execution_log.append(
                {
                    "type": "post_call",
                    "call_id": str(event.call_id),
                    "service": event.service_name,
                    "result_keys": list(event.result.keys()) if event.result else None,
                }
            )

        registry.register(HookType.PRE_CALL, pre_call_hook)
        registry.register(HookType.POST_CALL, post_call_hook)

        # Create service call context
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)
        ctx = CallContext(
            call_id=ctx.call_id,
            branch_id=ctx.branch_id,
            deadline_s=ctx.deadline_s,
            capabilities=ctx.capabilities,
            attrs={"service_name": "test-service"},  # Required for middleware
        )

        request = DummyRequest(model="test-integration")

        # Execute through middleware
        async def mock_next_call():
            return await service.call(request, ctx=ctx)

        result = await middleware(request, ctx, mock_next_call)
        await anyio.sleep(0.01)  # Allow hooks to complete

        # Verify service was called
        assert service.call_count == 1, "Service should be called once"
        assert (
            result["response"] == "processed test-integration"
        ), "Service result should flow through"

        # Verify hook execution order and data
        assert len(execution_log) == 2, "Both pre and post hooks should execute"

        pre_hook = next(log for log in execution_log if log["type"] == "pre_call")
        post_hook = next(log for log in execution_log if log["type"] == "post_call")

        assert pre_hook["call_id"] == str(ctx.call_id), "Pre-hook should have correct call_id"
        assert pre_hook["service"] == "test-service", "Pre-hook should have correct service name"
        assert pre_hook["request_model"] == "test-integration", "Pre-hook should have request data"

        assert post_hook["call_id"] == str(ctx.call_id), "Post-hook should have correct call_id"
        assert post_hook["result_keys"] == [
            "response",
            "call_id",
        ], "Post-hook should have result data"

    @pytest.mark.anyio
    async def test_hooked_middleware_error_handling_and_error_hooks(self):
        """Test: HookedMiddleware error handling and CALL_ERROR hook triggering."""
        registry = HookRegistry()
        middleware = HookedMiddleware(registry)
        service = MockService(should_fail=True)  # Configure service to fail

        error_captures = []

        async def error_hook(event: HookEvent):
            error_captures.append(
                {
                    "type": "call_error",
                    "call_id": str(event.call_id),
                    "error_type": type(event.error).__name__,
                    "error_message": str(event.error),
                    "service": event.service_name,
                }
            )

        registry.register(HookType.CALL_ERROR, error_hook)

        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)
        ctx = CallContext(
            call_id=ctx.call_id,
            branch_id=ctx.branch_id,
            deadline_s=ctx.deadline_s,
            capabilities=ctx.capabilities,
            attrs={"service_name": "failing-service"},
        )

        request = DummyRequest(model="test-error")

        async def mock_next_call():
            return await service.call(request, ctx=ctx)

        # Service should fail and middleware should propagate error
        with pytest.raises(ValueError, match="Mock service failure"):
            await middleware(request, ctx, mock_next_call)

        await anyio.sleep(0.01)  # Allow error hook to complete

        # Verify error hook was triggered
        assert len(error_captures) == 1, "Error hook should be triggered"

        error_capture = error_captures[0]
        assert error_capture["type"] == "call_error", "Should capture call error"
        assert error_capture["call_id"] == str(ctx.call_id), "Should have correct call_id"
        assert error_capture["error_type"] == "ValueError", "Should capture error type"
        assert (
            "Mock service failure" in error_capture["error_message"]
        ), "Should capture error message"
        assert error_capture["service"] == "failing-service", "Should capture service name"

    @pytest.mark.anyio
    async def test_hooked_middleware_streaming_integration_comprehensive(self):
        """Test: HookedMiddleware streaming integration with comprehensive hook coverage.

        CRITICAL: Validates PRE_STREAM, STREAM_CHUNK, POST_STREAM hook execution
        and chunk transformation in streaming context.
        """
        registry = HookRegistry()
        middleware = HookedMiddleware(registry)
        service = MockService()

        execution_log = []
        chunk_transformations = []

        # Pre-stream hook
        async def pre_stream_hook(event: HookEvent):
            execution_log.append("pre_stream_started")

        # Stream chunk transform hook
        async def chunk_transform_hook(event: HookEvent, chunk: dict) -> dict:
            transformed = {
                **chunk,
                "transformed": True,
                "hook_applied": "middleware_test",
            }
            chunk_transformations.append(f"chunk_{chunk['chunk']}")
            return transformed

        # Post-stream hook
        async def post_stream_hook(event: HookEvent):
            execution_log.append(
                f"post_stream_completed_chunks_{event.metadata.get('chunk_count', 0)}"
            )

        registry.register(HookType.PRE_STREAM, pre_stream_hook)
        registry.register_stream_hook(HookType.STREAM_CHUNK, chunk_transform_hook)
        registry.register(HookType.POST_STREAM, post_stream_hook)

        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)
        ctx = CallContext(
            call_id=ctx.call_id,
            branch_id=ctx.branch_id,
            deadline_s=ctx.deadline_s,
            capabilities=ctx.capabilities,
            attrs={"service_name": "streaming-service"},
        )

        request = DummyRequest(model="test-streaming")

        async def mock_next_stream():
            async for chunk in service.stream(request, ctx=ctx):
                yield chunk

        # Collect streaming results
        stream_results = []
        async for chunk in middleware.stream(request, ctx, mock_next_stream):
            stream_results.append(chunk)

        await anyio.sleep(0.01)  # Allow hooks to complete

        # Verify service was called
        assert service.stream_count == 1, "Service stream should be called once"

        # Verify all chunks were processed and transformed
        assert len(stream_results) == 3, "Should receive all 3 chunks"

        for i, chunk in enumerate(stream_results):
            assert chunk["chunk"] == i, f"Chunk {i} should have correct index"
            assert chunk["transformed"] is True, f"Chunk {i} should be transformed by hook"
            assert (
                chunk["hook_applied"] == "middleware_test"
            ), f"Chunk {i} should have hook metadata"

        # Verify hook execution order
        assert "pre_stream_started" in execution_log, "Pre-stream hook should execute"
        assert (
            "post_stream_completed_chunks_3" in execution_log
        ), "Post-stream hook should execute with correct count"

        # Verify all chunks were transformed
        assert len(chunk_transformations) == 3, "All chunks should be transformed"
        assert chunk_transformations == [
            "chunk_0",
            "chunk_1",
            "chunk_2",
        ], "Chunks should be processed in order"

    @pytest.mark.anyio
    async def test_hooked_middleware_streaming_error_handling(self):
        """Test: HookedMiddleware streaming error handling and STREAM_ERROR hooks."""
        registry = HookRegistry()
        middleware = HookedMiddleware(registry)
        service = MockService(should_fail=True)

        stream_error_captures = []

        async def stream_error_hook(event: HookEvent):
            stream_error_captures.append(
                {
                    "error_type": type(event.error).__name__,
                    "error_message": str(event.error),
                    "call_id": str(event.call_id),
                    "service": event.service_name,
                }
            )

        registry.register(HookType.STREAM_ERROR, stream_error_hook)

        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)
        ctx = CallContext(
            call_id=ctx.call_id,
            branch_id=ctx.branch_id,
            deadline_s=ctx.deadline_s,
            capabilities=ctx.capabilities,
            attrs={"service_name": "failing-stream-service"},
        )

        request = DummyRequest(model="test-stream-error")

        async def mock_failing_stream():
            async for chunk in service.stream(request, ctx=ctx):
                yield chunk

        # Stream should fail and trigger error hook
        with pytest.raises(ValueError, match="Mock streaming failure"):
            async for chunk in middleware.stream(request, ctx, mock_failing_stream):
                pass  # Should not reach this due to failure

        await anyio.sleep(0.01)  # Allow error hook to complete

        # Verify stream error hook was triggered
        assert len(stream_error_captures) == 1, "Stream error hook should be triggered"

        error_capture = stream_error_captures[0]
        assert error_capture["error_type"] == "ValueError", "Should capture correct error type"
        assert (
            "Mock streaming failure" in error_capture["error_message"]
        ), "Should capture error message"
        assert error_capture["service"] == "failing-stream-service", "Should capture service name"

    @pytest.mark.anyio
    async def test_observability_event_emission_timing_and_content(self):
        """Test: Observability event emission timing and content validation.

        CRITICAL: Validates that hook events contain proper observability data
        including timing, context, and structured information for monitoring.
        """
        registry = HookRegistry()
        middleware = HookedMiddleware(registry)
        service = MockService(delay=0.1)  # 100ms delay for timing validation

        observability_data = []

        async def observability_hook(event: HookEvent):
            observability_data.append(
                {
                    "hook_type": event.hook_type.value,
                    "timestamp": event.timestamp,
                    "call_id": str(event.call_id),
                    "branch_id": str(event.branch_id),
                    "service_name": event.service_name,
                    "has_request": event.request is not None,
                    "has_context": event.context is not None,
                    "has_result": event.result is not None,
                    "has_error": event.error is not None,
                    "metadata": event.metadata.copy() if event.metadata else {},
                }
            )

        # Register for multiple event types
        registry.register(HookType.PRE_CALL, observability_hook)
        registry.register(HookType.POST_CALL, observability_hook)

        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)
        ctx = CallContext(
            call_id=ctx.call_id,
            branch_id=ctx.branch_id,
            deadline_s=ctx.deadline_s,
            capabilities={"observability.metrics"},
            attrs={"service_name": "observable-service", "trace_id": "test-trace-123"},
        )

        request = DummyRequest(model="observability-test")

        async def mock_next_call():
            return await service.call(request, ctx=ctx)

        start_time = time.perf_counter()
        result = await middleware(request, ctx, mock_next_call)
        end_time = time.perf_counter()

        await anyio.sleep(0.01)

        # Verify timing
        total_duration = end_time - start_time
        assert total_duration >= 0.1, "Should include service delay time"

        # Verify observability events
        assert len(observability_data) == 2, "Should capture PRE_CALL and POST_CALL events"

        pre_call = next(event for event in observability_data if event["hook_type"] == "pre_call")
        post_call = next(event for event in observability_data if event["hook_type"] == "post_call")

        # Validate PRE_CALL observability data
        assert pre_call["call_id"] == str(ctx.call_id), "Pre-call should have correct call_id"
        assert pre_call["branch_id"] == str(ctx.branch_id), "Pre-call should have correct branch_id"
        assert pre_call["service_name"] == "observable-service", "Pre-call should have service name"
        assert pre_call["has_request"] is True, "Pre-call should have request data"
        assert pre_call["has_context"] is True, "Pre-call should have context data"
        assert pre_call["has_result"] is False, "Pre-call should not have result yet"
        assert pre_call["has_error"] is False, "Pre-call should not have error"

        # Validate POST_CALL observability data
        assert post_call["call_id"] == str(ctx.call_id), "Post-call should have correct call_id"
        assert (
            post_call["service_name"] == "observable-service"
        ), "Post-call should have service name"
        assert post_call["has_request"] is True, "Post-call should have request data"
        assert post_call["has_context"] is True, "Post-call should have context data"
        assert post_call["has_result"] is True, "Post-call should have result data"
        assert post_call["has_error"] is False, "Post-call should not have error"

        # Validate timing order
        assert post_call["timestamp"] > pre_call["timestamp"], "Post-call timestamp should be later"

    @pytest.mark.anyio
    async def test_hook_performance_impact_measurement(self):
        """Test: Hook performance impact measurement and overhead validation.

        CRITICAL: Measures the performance overhead of hooks to ensure they don't
        significantly impact service call performance in production.
        """
        registry_with_hooks = HookRegistry()
        registry_without_hooks = HookRegistry()

        service_fast = MockService(delay=0.01)  # 10ms service
        service_no_hooks = MockService(delay=0.01)

        # Add multiple hooks to measure overhead
        async def lightweight_hook_1(event: HookEvent):
            # Minimal work - just tracking
            pass

        async def lightweight_hook_2(event: HookEvent):
            # Minimal work - just tracking
            pass

        async def lightweight_hook_3(event: HookEvent):
            # Minimal work - just tracking
            pass

        registry_with_hooks.register(HookType.PRE_CALL, lightweight_hook_1)
        registry_with_hooks.register(HookType.PRE_CALL, lightweight_hook_2)
        registry_with_hooks.register(HookType.PRE_CALL, lightweight_hook_3)
        registry_with_hooks.register(HookType.POST_CALL, lightweight_hook_1)
        registry_with_hooks.register(HookType.POST_CALL, lightweight_hook_2)

        middleware_with_hooks = HookedMiddleware(registry_with_hooks)
        middleware_without_hooks = HookedMiddleware(registry_without_hooks)

        # Benchmark without hooks
        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)
        ctx = CallContext(
            call_id=ctx.call_id,
            branch_id=ctx.branch_id,
            deadline_s=ctx.deadline_s,
            capabilities=ctx.capabilities,
            attrs={"service_name": "performance-test"},
        )

        request = DummyRequest(model="perf-test")

        # Multiple runs for accuracy
        runs_per_test = 10

        # Benchmark without hooks
        async def mock_next_call_no_hooks():
            return await service_no_hooks.call(request, ctx=ctx)

        start_time = time.perf_counter()
        for _ in range(runs_per_test):
            await middleware_without_hooks(request, ctx, mock_next_call_no_hooks)
        no_hooks_duration = time.perf_counter() - start_time

        # Benchmark with hooks
        async def mock_next_call_with_hooks():
            return await service_fast.call(request, ctx=ctx)

        start_time = time.perf_counter()
        for _ in range(runs_per_test):
            await middleware_with_hooks(request, ctx, mock_next_call_with_hooks)
        with_hooks_duration = time.perf_counter() - start_time

        await anyio.sleep(0.01)  # Allow any remaining hooks to complete

        # Calculate overhead
        no_hooks_avg = no_hooks_duration / runs_per_test
        with_hooks_avg = with_hooks_duration / runs_per_test
        overhead_ms = (with_hooks_avg - no_hooks_avg) * 1000
        overhead_percent = ((with_hooks_avg - no_hooks_avg) / no_hooks_avg) * 100

        # Performance assertions - hooks should add minimal overhead
        assert overhead_ms < 5.0, f"Hook overhead should be < 5ms but was {overhead_ms:.3f}ms"
        assert (
            overhead_percent < 50.0
        ), f"Hook overhead should be < 50% but was {overhead_percent:.1f}%"

        # Verify both services were called correctly
        assert (
            service_no_hooks.call_count == runs_per_test
        ), "No-hooks service should be called for each run"
        assert (
            service_fast.call_count == runs_per_test
        ), "With-hooks service should be called for each run"

    @pytest.mark.anyio
    async def test_hooked_middleware_context_attribute_injection(self):
        """Test: HookedMiddleware context attribute handling and service name injection."""
        registry = HookRegistry()
        middleware = HookedMiddleware(registry)
        service = MockService()

        context_captures = []

        async def context_hook(event: HookEvent):
            context_captures.append(
                {
                    "service_name": event.service_name,
                    "context_attrs": dict(event.context.attrs) if event.context else {},
                }
            )

        registry.register(HookType.PRE_CALL, context_hook)

        # Test with explicit service_name in attrs
        ctx_explicit = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)
        ctx_explicit = CallContext(
            call_id=ctx_explicit.call_id,
            branch_id=ctx_explicit.branch_id,
            deadline_s=ctx_explicit.deadline_s,
            capabilities=ctx_explicit.capabilities,
            attrs={"service_name": "explicit-service", "custom_attr": "test_value"},
        )

        request = DummyRequest(model="context-test")

        async def mock_next_call():
            return await service.call(request, ctx=ctx_explicit)

        await middleware(request, ctx_explicit, mock_next_call)
        await anyio.sleep(0.01)

        assert len(context_captures) == 1, "Should capture context data"

        capture = context_captures[0]
        assert capture["service_name"] == "explicit-service", "Should use explicit service name"
        assert (
            capture["context_attrs"]["service_name"] == "explicit-service"
        ), "Context attrs should be preserved"
        assert (
            capture["context_attrs"]["custom_attr"] == "test_value"
        ), "Custom attributes should be preserved"

    @pytest.mark.anyio
    async def test_hooked_middleware_concurrent_execution_safety(self):
        """Test: HookedMiddleware thread safety under concurrent execution."""
        registry = HookRegistry()
        middleware = HookedMiddleware(registry)

        execution_counts = {"calls": []}

        async def concurrent_safe_hook(event: HookEvent):
            # Record each call - list append is thread-safe in Python
            execution_counts["calls"].append(str(event.call_id))
            await anyio.sleep(0.001)  # Small delay to test concurrency

        registry.register(HookType.PRE_CALL, concurrent_safe_hook)

        # Create multiple services and contexts for concurrent execution
        services = [MockService() for _ in range(10)]
        contexts = []

        for i in range(10):
            ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)
            ctx = CallContext(
                call_id=ctx.call_id,
                branch_id=ctx.branch_id,
                deadline_s=ctx.deadline_s,
                capabilities=ctx.capabilities,
                attrs={"service_name": f"concurrent-service-{i}"},
            )
            contexts.append(ctx)

        request = DummyRequest(model="concurrent-test")

        # Create concurrent service calls
        async def concurrent_call(service: MockService, ctx: CallContext):
            async def mock_next_call():
                return await service.call(request, ctx=ctx)

            return await middleware(request, ctx, mock_next_call)

        # Execute all calls concurrently
        from lionagi.ln.concurrency import gather

        results = await gather(
            *(concurrent_call(services[i], contexts[i]) for i in range(10)),
            return_exceptions=True,
        )

        await anyio.sleep(0.01)  # Allow hooks to complete

        # Verify all calls completed successfully
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Should have no exceptions but got: {exceptions}"

        # Verify all hooks were executed (list append is thread-safe)
        assert (
            len(execution_counts["calls"]) == 10
        ), f"Should execute all 10 hooks but got {len(execution_counts['calls'])}"

        # Verify all call_ids are unique (no duplicate hook executions)
        unique_calls = set(execution_counts["calls"])
        assert len(unique_calls) == 10, "Each hook should be called exactly once"

    @pytest.mark.anyio
    async def test_hooked_middleware_service_name_fallback_behavior(self):
        """Test: HookedMiddleware service name fallback when not in context attrs."""
        registry = HookRegistry()
        middleware = HookedMiddleware(registry)
        service = MockService()

        service_name_captures = []

        async def service_name_hook(event: HookEvent):
            service_name_captures.append(event.service_name)

        registry.register(HookType.PRE_CALL, service_name_hook)

        # Test with context that has no service_name in attrs
        ctx_no_service = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)
        # Don't add service_name to attrs

        request = DummyRequest(model="fallback-test")

        async def mock_next_call():
            return await service.call(request, ctx=ctx_no_service)

        await middleware(request, ctx_no_service, mock_next_call)
        await anyio.sleep(0.01)

        assert len(service_name_captures) == 1, "Should capture service name"
        assert (
            service_name_captures[0] == "unknown"
        ), "Should use 'unknown' as fallback service name"

    @pytest.mark.anyio
    async def test_hooked_middleware_empty_registry_behavior(self):
        """Test: HookedMiddleware behavior with empty hook registry."""
        empty_registry = HookRegistry()
        middleware = HookedMiddleware(empty_registry)
        service = MockService()

        ctx = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)
        ctx = CallContext(
            call_id=ctx.call_id,
            branch_id=ctx.branch_id,
            deadline_s=ctx.deadline_s,
            capabilities=ctx.capabilities,
            attrs={"service_name": "empty-registry-test"},
        )

        request = DummyRequest(model="empty-test")

        async def mock_next_call():
            return await service.call(request, ctx=ctx)

        # Should work normally even with no hooks
        start_time = time.perf_counter()
        result = await middleware(request, ctx, mock_next_call)
        end_time = time.perf_counter()

        # Verify normal operation
        assert result["response"] == "processed empty-test", "Service should execute normally"
        assert service.call_count == 1, "Service should be called once"

        # Should be very fast with no hooks
        duration = end_time - start_time
        assert duration < 0.1, f"Empty registry should add minimal overhead, took {duration:.3f}s"
