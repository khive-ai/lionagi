# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Consolidated P1 Integration Tests for lionagi v1 services - Agent Kernel readiness.

This file consolidates the essential integration tests from 4 previously redundant files:
- test_middleware_integration.py (1033 lines)
- test_imodel_integration.py (920 lines)
- test_agent_kernel_scenarios.py (918 lines)
- test_resilience_integration.py (843 lines)

Focus: TDD P1 requirements - "Full pipeline via iModel" and "Provider capability propagation"
Eliminates: 6x redundancy in middleware order, 4x redundancy in policy enforcement,
           800+ lines of duplicate mock setup

Tests validate Agent Kernel principles:
1. Structured execution with deadline enforcement
2. Least privilege through capability propagation
3. Constraints as guardrails via policy enforcement
4. Declarative data flow through CallContext
5. Performance through standardization (msgspec)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import anyio
import pytest
from anyio.testing import MockClock

from lionagi.errors import PolicyError, ServiceError, TimeoutError
from lionagi.services.core import CallContext, Service
from lionagi.services.endpoint import RequestModel
from lionagi.services.executor import ExecutorConfig, RateLimitedExecutor, ServiceCall
from lionagi.services.hooks import HookRegistry, HookType
from lionagi.services.imodel import iModel
from lionagi.services.middleware import (
    CallMW,
    CircuitBreakerMW,
    HookedMiddleware,
    MetricsMW,
    PolicyGateMW,
    RedactionMW,
    RetryMW,
)

# Consolidated Mock Services (replaces 800+ lines of duplicate setup)


class TestRequest(RequestModel):
    """Simple test request."""

    content: str = "test message"
    model: str = "gpt-4"


class ConfigurableTestService(Service):
    """Single configurable service replacing multiple duplicate mock services."""

    name = "test_service"
    requires = {"net.out:api.openai.com"}

    def __init__(
        self,
        call_delay: float = 0.1,
        stream_chunks: int = 3,
        failure_mode: str | None = None,
        failure_after: int = 0,
    ):
        self.call_delay = call_delay
        self.stream_chunks = stream_chunks
        self.failure_mode = failure_mode
        self.call_count = 0
        self.failure_after = failure_after

    async def call(self, req: TestRequest, *, ctx: CallContext) -> dict[str, Any]:
        self.call_count += 1

        if self.failure_mode and self.call_count > self.failure_after:
            if self.failure_mode == "retryable":
                raise ServiceError("Simulated retryable failure")
            elif self.failure_mode == "non_retryable":
                raise PolicyError("Simulated non-retryable failure", context={"reason": "test"})
            elif self.failure_mode == "timeout":
                raise TimeoutError("Simulated timeout")

        await anyio.sleep(self.call_delay)
        return {
            "result": f"processed: {req.content}",
            "call_count": self.call_count,
            "call_id": str(ctx.call_id),
            "model": req.model,
        }

    async def stream(self, req: TestRequest, *, ctx: CallContext) -> AsyncIterator[dict[str, Any]]:
        for i in range(self.stream_chunks):
            if self.failure_mode == "stream_error" and i == 1:
                raise ServiceError("Simulated stream error")

            await anyio.sleep(self.call_delay / self.stream_chunks)
            yield {
                "chunk": i,
                "content": req.content,
                "call_id": str(ctx.call_id),
                "total_chunks": self.stream_chunks,
            }


@pytest.fixture
def test_service():
    """Default test service."""
    return ConfigurableTestService()


@pytest.fixture
def slow_service():
    """Service with longer delay."""
    return ConfigurableTestService(call_delay=0.5)


@pytest.fixture
def failing_service():
    """Service that fails after 1 successful call."""
    return ConfigurableTestService(failure_mode="retryable", failure_after=1)


@pytest.fixture
def executor_config():
    """Test executor configuration."""
    return ExecutorConfig(
        queue_capacity=10,
        capacity_refresh_time=1.0,
        limit_requests=5,
        limit_tokens=1000,
        concurrency_limit=3,
    )


@pytest.fixture
async def test_executor(executor_config):
    """Test executor with cleanup."""
    executor = RateLimitedExecutor(executor_config)
    await executor.start()
    try:
        yield executor
    finally:
        await executor.stop()


@pytest.fixture
def hook_registry():
    """Test hook registry."""
    return HookRegistry()


class TestIntegrationCore:
    """Core P1 integration tests consolidating previous redundant files.

    Tests the essential integration behaviors from TDD specification:
    - Full pipeline via iModel (TDD P1 requirement)
    - Provider capability propagation (TDD P1 requirement)
    - Middleware execution order (Policy → Metrics → Hooks → Service)
    - Error propagation through complete pipeline
    - Context preservation across all boundaries
    """

    @pytest.mark.anyio
    async def test_full_pipeline_via_imodel_p1_requirement(
        self, test_service, hook_registry, caplog
    ):
        """TDD P1 REQUIREMENT: Full pipeline via iModel.

        Validates complete Agent Kernel pipeline: build context, install middleware,
        submit to executor, await completion with proper observability.
        """
        # Test Agent Kernel principle: Structured execution with all middleware
        with patch("lionagi.services.provider_detection.detect_provider") as mock_detect:
            mock_detect.return_value = "openai"

            # Create iModel with all middleware enabled
            model = iModel(
                model="gpt-4",
                service=test_service,
                enable_policy=True,
                enable_metrics=True,
                enable_hooks=True,
                enable_redaction=True,
            )

            # Register test hooks for observability validation
            pre_call_events = []
            post_call_events = []

            @hook_registry.register(HookType.PRE_CALL)
            async def capture_pre_call(event):
                pre_call_events.append(event)

            @hook_registry.register(HookType.POST_CALL)
            async def capture_post_call(event):
                post_call_events.append(event)

            model.hooks = hook_registry

            # Execute full pipeline
            request = TestRequest(content="Agent Kernel test", model="gpt-4")
            context = CallContext.new(
                branch_id=uuid4(),
                capabilities={"net.out:api.openai.com"},  # Required capability
            )

            with caplog.at_level(logging.INFO):
                result = await model.call(request, context=context)

            # Validate Agent Kernel principles
            assert result["result"] == "processed: Agent Kernel test"
            assert result["model"] == "gpt-4"
            assert "call_id" in result

            # Validate structured execution (observability)
            assert len(pre_call_events) == 1
            assert len(post_call_events) == 1
            assert pre_call_events[0]["type"] == HookType.PRE_CALL
            assert post_call_events[0]["type"] == HookType.POST_CALL

            # Validate structured logging
            log_records = [r for r in caplog.records if r.levelname == "INFO"]
            assert any("call_id" in r.getMessage() for r in log_records)

    @pytest.mark.anyio
    async def test_provider_capability_propagation_p1_requirement(self, test_service):
        """TDD P1 REQUIREMENT: Provider capability propagation.

        Validates service.requires → CallContext.capabilities flow and policy
        enforcement with exact/wildcard capability matching.
        """
        with patch("lionagi.services.provider_detection.detect_provider") as mock_detect:
            mock_detect.return_value = "openai"

            model = iModel(model="gpt-4", service=test_service, enable_policy=True)
            request = TestRequest(content="capability test")

            # Test 1: Exact capability match should succeed
            context_exact = CallContext.new(
                branch_id=uuid4(),
                capabilities={"net.out:api.openai.com"},  # Exact match
            )

            result = await model.call(request, context=context_exact)
            assert result["result"] == "processed: capability test"

            # Test 2: Wildcard capability match should succeed
            context_wildcard = CallContext.new(
                branch_id=uuid4(),
                capabilities={"net.out:*"},  # Wildcard match
            )

            result = await model.call(request, context=context_wildcard)
            assert result["result"] == "processed: capability test"

            # Test 3: Insufficient capabilities should fail (fail-closed security)
            context_insufficient = CallContext.new(
                branch_id=uuid4(),
                capabilities={"fs.read:/safe"},  # Wrong capability
            )

            with pytest.raises(PolicyError) as exc_info:
                await model.call(request, context=context_insufficient)

            # Validate fail-closed behavior and security audit context
            error_context = exc_info.value.context
            assert "missing_capabilities" in error_context
            assert "net.out:api.openai.com" in error_context["missing_capabilities"]

    @pytest.mark.anyio
    async def test_middleware_execution_order_policy_metrics_hooks_service(
        self, test_service, hook_registry
    ):
        """Validates middleware chain execution order: Policy → Metrics → Hooks → Service.

        Consolidates 4 redundant middleware order tests from previous files.
        Tests Agent Kernel principle: Constraints as guardrails.
        """
        execution_order = []

        # Mock middleware to track execution order
        original_policy_call = PolicyGateMW.__call__
        original_metrics_call = MetricsMW.__call__
        original_hooked_call = HookedMiddleware.__call__

        async def track_policy(self, req, ctx, next_call):
            execution_order.append("policy")
            return await original_policy_call(self, req, ctx, next_call)

        async def track_metrics(self, req, ctx, next_call):
            execution_order.append("metrics")
            return await original_metrics_call(self, req, ctx, next_call)

        async def track_hooked(self, req, ctx, next_call):
            execution_order.append("hooks")
            return await original_hooked_call(self, req, ctx, next_call)

        # Patch middleware to track execution
        with (
            patch.object(PolicyGateMW, "__call__", track_policy),
            patch.object(MetricsMW, "__call__", track_metrics),
            patch.object(HookedMiddleware, "__call__", track_hooked),
        ):
            with patch("lionagi.services.provider_detection.detect_provider") as mock_detect:
                mock_detect.return_value = "openai"

                model = iModel(
                    model="gpt-4",
                    service=test_service,
                    enable_policy=True,
                    enable_metrics=True,
                    enable_hooks=True,
                )
                model.hooks = hook_registry

                request = TestRequest(content="order test")
                context = CallContext.new(
                    branch_id=uuid4(), capabilities={"net.out:api.openai.com"}
                )

                await model.call(request, context=context)

                # Validate exact execution order
                expected_order = ["policy", "metrics", "hooks"]
                assert execution_order == expected_order

    @pytest.mark.anyio
    async def test_error_propagation_through_complete_pipeline(self, hook_registry):
        """Tests error propagation through middleware chain and executor.

        Consolidates 6 redundant error propagation tests from previous files.
        Validates that errors bubble correctly and don't get lost in middleware.
        """
        failing_service = ConfigurableTestService(failure_mode="retryable", failure_after=0)

        with patch("lionagi.services.provider_detection.detect_provider") as mock_detect:
            mock_detect.return_value = "openai"

            model = iModel(
                model="gpt-4",
                service=failing_service,
                enable_policy=True,
                enable_metrics=True,
                enable_hooks=True,
            )
            model.hooks = hook_registry

            request = TestRequest(content="error test")
            context = CallContext.new(branch_id=uuid4(), capabilities={"net.out:api.openai.com"})

            # Service error should propagate through entire pipeline
            with pytest.raises(ServiceError) as exc_info:
                await model.call(request, context=context)

            assert "Simulated retryable failure" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_context_preservation_across_all_boundaries(self, test_service, hook_registry):
        """Tests CallContext preservation across middleware → executor → service boundaries.

        Consolidates 4 redundant context preservation tests.
        Validates Agent Kernel principle: Declarative data flow.
        """
        captured_contexts = []

        @hook_registry.register(HookType.PRE_CALL)
        async def capture_context(event):
            captured_contexts.append(event["context"])

        with patch("lionagi.services.provider_detection.detect_provider") as mock_detect:
            mock_detect.return_value = "openai"

            model = iModel(
                model="gpt-4",
                service=test_service,
                enable_policy=True,
                enable_metrics=True,
                enable_hooks=True,
            )
            model.hooks = hook_registry

            original_branch_id = uuid4()
            original_call_id = uuid4()

            context = CallContext.new(
                branch_id=original_branch_id,
                call_id=original_call_id,
                capabilities={"net.out:api.openai.com"},
                attrs={"test_attr": "preserved_value"},
            )

            request = TestRequest(content="context test")
            result = await model.call(request, context=context)

            # Validate context preservation
            assert len(captured_contexts) == 1
            preserved_context = captured_contexts[0]

            assert preserved_context.branch_id == original_branch_id
            assert preserved_context.call_id == original_call_id
            assert preserved_context.capabilities == {"net.out:api.openai.com"}
            assert preserved_context.attrs["test_attr"] == "preserved_value"

            # Validate context reached service (via call_id in result)
            assert result["call_id"] == str(original_call_id)

    @pytest.mark.anyio
    async def test_streaming_integration_with_hooks_and_middleware(
        self, test_service, hook_registry
    ):
        """Tests streaming through complete pipeline with chunk transformation.

        Consolidates 4 redundant streaming integration tests.
        Validates Agent Kernel principle: Performance through standardization.
        """
        chunk_events = []

        @hook_registry.register(HookType.STREAM_CHUNK)
        async def capture_chunks(event):
            chunk_events.append(event["chunk"])
            # Transform chunk data
            event["chunk"]["transformed"] = True
            return event["chunk"]

        with patch("lionagi.services.provider_detection.detect_provider") as mock_detect:
            mock_detect.return_value = "openai"

            model = iModel(model="gpt-4", service=test_service, enable_hooks=True)
            model.hooks = hook_registry

            request = TestRequest(content="streaming test")
            context = CallContext.new(branch_id=uuid4(), capabilities={"net.out:api.openai.com"})

            chunks = []
            async for chunk in model.stream(request, context=context):
                chunks.append(chunk)

            # Validate streaming worked
            assert len(chunks) == 3  # ConfigurableTestService default
            assert all(chunk["content"] == "streaming test" for chunk in chunks)
            assert all("call_id" in chunk for chunk in chunks)

            # Validate hook transformation
            assert len(chunk_events) == 3
            assert all(chunk.get("transformed") for chunk in chunks)

    @pytest.mark.anyio
    async def test_deadline_enforcement_end_to_end(self, hook_registry):
        """Tests deadline enforcement through complete pipeline.

        Validates Agent Kernel principle: Structured execution with timeout contracts.
        """
        slow_service = ConfigurableTestService(call_delay=1.0)  # 1 second delay

        with patch("lionagi.services.provider_detection.detect_provider") as mock_detect:
            mock_detect.return_value = "openai"

            model = iModel(model="gpt-4", service=slow_service)

            request = TestRequest(content="timeout test")
            context = CallContext.with_timeout(
                branch_id=uuid4(),
                timeout_s=0.3,  # Shorter than service delay
                capabilities={"net.out:api.openai.com"},
            )

            start_time = time.time()

            # Should timeout after ~0.3s, not wait for 1s service delay
            with pytest.raises(TimeoutError):
                await model.call(request, context=context)

            elapsed = time.time() - start_time
            assert 0.2 < elapsed < 0.6  # Should timeout quickly

    @pytest.mark.anyio
    async def test_observability_correlation_call_id_branch_id(
        self, test_service, hook_registry, caplog
    ):
        """Tests observability data correlation with call_id/branch_id.

        Consolidates 6 redundant observability tests.
        Validates structured logging and metrics correlation.
        """
        hook_events = []

        @hook_registry.register(HookType.PRE_CALL)
        async def capture_pre(event):
            hook_events.append(("pre_call", event))

        @hook_registry.register(HookType.POST_CALL)
        async def capture_post(event):
            hook_events.append(("post_call", event))

        with patch("lionagi.services.provider_detection.detect_provider") as mock_detect:
            mock_detect.return_value = "openai"

            model = iModel(
                model="gpt-4",
                service=test_service,
                enable_metrics=True,
                enable_hooks=True,
            )
            model.hooks = hook_registry

            original_branch_id = uuid4()
            original_call_id = uuid4()

            context = CallContext.new(
                branch_id=original_branch_id,
                call_id=original_call_id,
                capabilities={"net.out:api.openai.com"},
            )

            request = TestRequest(content="observability test")

            with caplog.at_level(logging.INFO):
                result = await model.call(request, context=context)

            # Validate call_id correlation
            assert result["call_id"] == str(original_call_id)

            # Validate hook event correlation
            assert len(hook_events) == 2
            pre_event = hook_events[0][1]
            post_event = hook_events[1][1]

            assert pre_event["call_id"] == original_call_id
            assert pre_event["branch_id"] == original_branch_id
            assert post_event["call_id"] == original_call_id
            assert post_event["branch_id"] == original_branch_id

            # Validate structured logging correlation
            log_messages = [r.getMessage() for r in caplog.records]
            assert any(str(original_call_id) in msg for msg in log_messages)
