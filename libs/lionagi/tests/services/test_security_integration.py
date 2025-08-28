# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive end-to-end security integration tests for lionagi v1 services.

Tests security enforcement through the complete iModel pipeline, validates
attack vector prevention, and ensures proper security audit logging.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any
from types import MappingProxyType
from uuid import uuid4

import pytest

from lionagi.errors import PolicyError, ServiceError
from lionagi.services.core import CallContext, Service
from lionagi.services.endpoint import RequestModel
from lionagi.services.executor import ExecutorConfig, RateLimitedExecutor
from lionagi.services.middleware import MetricsMW, PolicyGateMW, RedactionMW


class MockService(Service):
    """Mock service for testing security integration."""

    name = "mock_service"

    def __init__(self, requires: set[str] | None = None, *, should_fail: bool = False):
        self.requires = requires or set()
        self.should_fail = should_fail
        self.call_count = 0
        self.stream_count = 0

    async def call(self, req: RequestModel, *, ctx: CallContext) -> dict[str, Any]:
        """Mock service call implementation."""
        self.call_count += 1

        if self.should_fail:
            raise ServiceError("Mock service failure", context={"call_id": str(ctx.call_id)})

        return {
            "status": "success",
            "call_id": str(ctx.call_id),
            "branch_id": str(ctx.branch_id),
            "model": getattr(req, "model", "test-model"),
            "call_count": self.call_count,
        }

    async def stream(self, req: RequestModel, *, ctx: CallContext) -> AsyncIterator[dict[str, Any]]:
        """Mock service streaming implementation."""
        self.stream_count += 1

        if self.should_fail:
            raise ServiceError("Mock streaming failure", context={"call_id": str(ctx.call_id)})

        for i in range(3):
            yield {
                "chunk": i,
                "call_id": str(ctx.call_id),
                "stream_count": self.stream_count,
            }


class MockRequest(RequestModel):
    """Mock request for testing."""

    model: str = "test-model"
    messages: list[dict] = []
    _extra_requires: set[str] | None = None


class TestEndToEndSecurityEnforcement:
    """Test security enforcement through the complete service pipeline."""

    @pytest.mark.anyio
    async def test_imodel_security_pipeline_policy_denial(self):
        """Test that PolicyGateMW prevents execution at the iModel level."""
        # Create mock service that requires admin access
        mock_service = MockService(requires={"admin:delete"})

        # Create executor config
        config = ExecutorConfig(queue_capacity=10)
        executor = RateLimitedExecutor(config)
        await executor.start()

        try:
            # Create middleware stack with policy enforcement
            policy_mw = PolicyGateMW()
            metrics_mw = MetricsMW()

            # Context with insufficient capabilities
            ctx = CallContext.new(
                branch_id=uuid4(),
                capabilities={"user:read"},  # Insufficient - needs admin:delete
                service_requires=mock_service.requires,
            )

            req = MockRequest()

            # Test policy enforcement at middleware level
            executed = False

            async def attempt_call():
                nonlocal executed
                executed = True
                return await mock_service.call(req, ctx=ctx)

            # Policy middleware should block the call
            with pytest.raises(PolicyError) as exc_info:
                await policy_mw(req, ctx, attempt_call)

            # Verify service was never called
            assert not executed
            assert mock_service.call_count == 0

            # Verify policy error context
            error_ctx = exc_info.value.context
            assert error_ctx["policy_check"] == "capability_enforcement"
            assert "admin:delete" in error_ctx["missing_capabilities"]

        finally:
            await executor.stop()

    @pytest.mark.anyio
    async def test_imodel_security_pipeline_success_flow(self):
        """Test successful security enforcement when capabilities are sufficient."""
        mock_service = MockService(requires={"fs:read:/data"})

        config = ExecutorConfig(queue_capacity=10)
        executor = RateLimitedExecutor(config)
        await executor.start()

        try:
            # Create middleware stack
            policy_mw = PolicyGateMW()
            metrics_mw = MetricsMW()

            # Context with sufficient capabilities
            ctx = CallContext.new(
                branch_id=uuid4(),
                capabilities={"fs:read:/data", "net:out:api.openai.com"},  # Sufficient
                service_requires=mock_service.requires,
            )

            req = MockRequest()

            # Chain middleware (policy -> metrics -> service)
            async def metrics_wrapped():
                return await metrics_mw(req, ctx, lambda: mock_service.call(req, ctx=ctx))

            # Should succeed through policy gate
            result = await policy_mw(req, ctx, metrics_wrapped)

            # Verify successful execution
            assert result["status"] == "success"
            assert mock_service.call_count == 1

        finally:
            await executor.stop()

    @pytest.mark.anyio
    async def test_streaming_security_enforcement(self):
        """Test security enforcement for streaming operations."""
        mock_service = MockService(requires={"stream:admin"})

        # Test insufficient capabilities
        ctx_insufficient = CallContext.new(
            branch_id=uuid4(),
            capabilities={"stream:user"},  # Insufficient
            service_requires=mock_service.requires,
        )

        policy_mw = PolicyGateMW()
        req = MockRequest()

        # Should fail at policy gate
        with pytest.raises(PolicyError):
            async for chunk in policy_mw.stream(
                req,
                ctx_insufficient,
                lambda: mock_service.stream(req, ctx=ctx_insufficient),
            ):
                pass

        assert mock_service.stream_count == 0, "Stream should not have started"

        # Test sufficient capabilities
        ctx_sufficient = CallContext.new(
            branch_id=uuid4(),
            capabilities={"stream:admin", "other:cap"},  # Sufficient
            service_requires=mock_service.requires,
        )

        chunks = []
        async for chunk in policy_mw.stream(
            req, ctx_sufficient, lambda: mock_service.stream(req, ctx=ctx_sufficient)
        ):
            chunks.append(chunk)

        assert len(chunks) == 3  # Mock service yields 3 chunks
        assert mock_service.stream_count == 1


class TestAttackVectorValidation:
    """Test validation against various attack vectors."""

    @pytest.mark.anyio
    async def test_capability_injection_attack_prevention(self):
        """Test prevention of capability injection through request manipulation."""
        mock_service = MockService(requires={"admin:root"})
        policy_mw = PolicyGateMW()

        # Attacker has limited capabilities
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities={"user:read"},  # Limited user access
            service_requires=mock_service.requires,  # Requires admin:root
        )

        # Various injection attempts through request fields
        attack_vectors = [
            # Try to inject through model field
            MockRequest(model="admin:root"),
            # Try to inject through extra requirements (legitimate field but shouldn't bypass)
            MockRequest(_extra_requires={"admin:root"}),
            # Try complex injection patterns
            MockRequest(model="system/../../admin:root"),
        ]

        for attack_req in attack_vectors:
            executed = False

            async def track_execution():
                nonlocal executed
                executed = True
                return {"attack": "successful"}

            with pytest.raises(PolicyError) as exc_info:
                await policy_mw(attack_req, ctx, track_execution)

            assert not executed, f"Injection attack succeeded with {attack_req.__dict__}"
            assert "admin:root" in exc_info.value.context["missing_capabilities"]

    @pytest.mark.anyio
    async def test_wildcard_abuse_prevention(self):
        """Test prevention of wildcard pattern abuse for privilege escalation."""
        mock_service = MockService(requires={"admin:delete", "system:shutdown"})
        policy_mw = PolicyGateMW()

        # Attacker attempts to use wildcard patterns
        wildcard_abuse_attempts = [
            # User has limited wildcard but tries to access admin functions
            {"user:*"},  # Should not cover admin:delete
            # Partial wildcard that doesn't match
            {"ad*"},  # Malformed or partial wildcard
            # Similar but insufficient wildcards
            {"admin:read*"},  # admin:read* doesn't cover admin:delete exactly
        ]

        for attacker_caps in wildcard_abuse_attempts:
            ctx = CallContext.new(
                branch_id=uuid4(),
                capabilities=attacker_caps,
                service_requires=mock_service.requires,
            )

            req = MockRequest()
            executed = False

            async def track_execution():
                nonlocal executed
                executed = True
                return {"escalation": "successful"}

            with pytest.raises(PolicyError):
                await policy_mw(req, ctx, track_execution)

            assert not executed, f"Wildcard abuse succeeded with {attacker_caps}"

    @pytest.mark.anyio
    async def test_context_manipulation_attack_prevention(self):
        """Test prevention of attacks through context manipulation."""
        mock_service = MockService(requires={"secure:access"})
        policy_mw = PolicyGateMW()

        # Attempt to manipulate context after creation
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities={"public:read"},
            attrs={"service_requires": mock_service.requires},
        )

        # Various context manipulation attempts (these should not work due to msgspec.Struct)
        original_caps = ctx.capabilities.copy()

        # Try to modify capabilities directly (this might not work with msgspec frozen structs)
        try:
            ctx.capabilities.add("secure:access")  # Attempt direct modification
        except (AttributeError, TypeError):
            pass  # Expected with frozen structs

        # Try to modify attrs
        try:
            ctx.attrs["service_requires"] = set()  # Attempt to clear requirements
        except (AttributeError, TypeError):
            pass  # Expected with frozen/immutable structures

        req = MockRequest()
        executed = False

        async def track_execution():
            nonlocal executed
            executed = True
            return {"manipulation": "successful"}

        # Should still fail with original insufficient capabilities
        with pytest.raises(PolicyError):
            await policy_mw(req, ctx, track_execution)

        assert not executed, "Context manipulation attack succeeded"

    @pytest.mark.anyio
    async def test_race_condition_attack_prevention(self):
        """Test prevention of race condition attacks in capability checking."""
        mock_service = MockService(requires={"race:sensitive"})
        policy_mw = PolicyGateMW()

        # Simulate concurrent access attempts with different capability levels
        attack_tasks = []
        results = []

        async def attempt_access(caps: set[str], should_succeed: bool):
            ctx = CallContext.new(
                branch_id=uuid4(),
                capabilities=caps,
                service_requires=mock_service.requires,
            )

            req = MockRequest()

            try:
                result = await policy_mw(req, ctx, lambda: {"access": "granted"})
                results.append(("success", should_succeed, caps))
                return result
            except PolicyError:
                results.append(("denied", should_succeed, caps))
                if should_succeed:
                    raise AssertionError(f"Expected success but was denied: {caps}")

        # Launch concurrent attempts with different capability levels
        tasks = [
            attempt_access({"race:sensitive"}, True),  # Should succeed
            attempt_access({"user:read"}, False),  # Should fail
            attempt_access({"race:sensitive"}, True),  # Should succeed
            attempt_access(set(), False),  # Should fail
            attempt_access({"race:*"}, True),  # Should succeed (wildcard)
        ]

        # Run concurrently to test for race conditions
        await asyncio.gather(*tasks, return_exceptions=True)

        # Verify results are consistent with expected behavior
        for result_status, should_succeed, caps in results:
            if should_succeed:
                assert result_status == "success", f"Race condition: {caps} should have succeeded"
            else:
                assert result_status == "denied", f"Race condition: {caps} should have been denied"


class TestSecurityAuditLogging:
    """Test security audit logging and observability."""

    @pytest.mark.anyio
    async def test_policy_error_audit_logging(self, caplog):
        """Test that policy errors are properly logged for security auditing."""
        caplog.set_level(logging.DEBUG)

        mock_service = MockService(requires={"audit:target"})
        policy_mw = PolicyGateMW()

        call_id = uuid4()
        branch_id = uuid4()

        ctx = CallContext(
            call_id=call_id,
            branch_id=branch_id,
            capabilities=frozenset({"user:basic"}),
            attrs=MappingProxyType({"service_requires": mock_service.requires}),
        )

        req = MockRequest()

        # Attempt unauthorized access
        with pytest.raises(PolicyError) as exc_info:
            await policy_mw(req, ctx, lambda: {"unauthorized": True})

        # Verify comprehensive security context in error
        error_context = exc_info.value.context

        # Essential audit fields
        assert error_context["call_id"] == str(call_id)
        assert error_context["branch_id"] == str(branch_id)
        assert error_context["policy_check"] == "capability_enforcement"
        assert error_context["operation"] == "call"

        # Security analysis fields
        assert set(error_context["available_capabilities"]) == {"user:basic"}
        assert set(error_context["required_capabilities"]) == {"audit:target"}
        assert set(error_context["missing_capabilities"]) == {"audit:target"}

    @pytest.mark.anyio
    async def test_redaction_middleware_integration(self, caplog):
        """Test that RedactionMW properly redacts sensitive data in security contexts."""
        caplog.set_level(logging.DEBUG)

        mock_service = MockService()
        redaction_mw = RedactionMW()

        # Create context with sensitive data
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities={"test:cap"},
            # Add sensitive data to attrs
            authorization="Bearer secret_token_123",
            api_key="sk-secret-api-key-456",
            password="super_secret_password",
            normal_field="public_data",
        )

        req = MockRequest()

        # Execute through redaction middleware
        async def mock_next():
            return {"redaction": "test"}
        
        result = await redaction_mw(req, ctx, mock_next)

        # Check that sensitive fields were redacted in logs
        log_records = [
            record.message for record in caplog.records if "Service call starting" in record.message
        ]

        # Should have logged the start with redacted sensitive fields
        assert len(log_records) > 0

        # Verify sensitive data doesn't appear in logs (check log extras)
        for record in caplog.records:
            if hasattr(record, "attrs"):
                attrs = record.attrs
                assert "secret_token_123" not in str(attrs), "Authorization token leaked in logs"
                assert "secret-api-key-456" not in str(attrs), "API key leaked in logs"
                assert "super_secret_password" not in str(attrs), "Password leaked in logs"

    @pytest.mark.anyio
    async def test_metrics_middleware_security_context(self, caplog):
        """Test that MetricsMW captures security-relevant metrics."""
        caplog.set_level(logging.INFO)

        mock_service = MockService()
        metrics_mw = MetricsMW()

        ctx = CallContext.new(branch_id=uuid4(), capabilities={"test:metric"})

        req = MockRequest(model="security-test-model")

        # Execute successful call
        result = await metrics_mw(req, ctx, lambda: mock_service.call(req, ctx=ctx))

        # Check metrics logging
        success_logs = [r for r in caplog.records if "Service call completed" in r.message]
        assert len(success_logs) == 1

        success_log = success_logs[0]
        assert success_log.call_id == str(ctx.call_id)
        assert success_log.branch_id == str(ctx.branch_id)
        assert success_log.status == "success"
        assert success_log.model == "security-test-model"
        assert hasattr(success_log, "duration_s")

        # Test error case
        caplog.clear()
        mock_service.should_fail = True

        with pytest.raises(ServiceError):
            await metrics_mw(req, ctx, lambda: mock_service.call(req, ctx=ctx))

        # Check error metrics
        error_logs = [r for r in caplog.records if "Service call failed" in r.message]
        assert len(error_logs) == 1

        error_log = error_logs[0]
        assert error_log.status == "error"
        assert error_log.error_type == "ServiceError"


class TestSecurityMiddlewareComposition:
    """Test security when multiple middleware components are composed together."""

    @pytest.mark.anyio
    async def test_policy_metrics_redaction_middleware_stack(self, caplog):
        """Test complete middleware stack: Policy -> Metrics -> Redaction -> Service."""
        caplog.set_level(logging.DEBUG)

        mock_service = MockService(requires={"full:stack"})

        # Create middleware stack
        policy_mw = PolicyGateMW()
        metrics_mw = MetricsMW()
        redaction_mw = RedactionMW()

        # Test case 1: Insufficient capabilities (should fail at policy gate)
        ctx_fail = CallContext.new(
            branch_id=uuid4(),
            capabilities={"user:basic"},
            service_requires=mock_service.requires,
            sensitive_data="should_be_redacted",
        )

        req = MockRequest()

        # Compose middleware stack
        async def full_stack():
            return await redaction_mw(
                req,
                ctx_fail,
                lambda: metrics_mw(req, ctx_fail, lambda: mock_service.call(req, ctx=ctx_fail)),
            )

        # Should fail at policy gate before reaching other middleware
        with pytest.raises(PolicyError):
            await policy_mw(req, ctx_fail, full_stack)

        # Verify service was never called
        assert mock_service.call_count == 0

        # Test case 2: Sufficient capabilities (should succeed through full stack)
        caplog.clear()
        ctx_success = CallContext.new(
            branch_id=uuid4(),
            capabilities={"full:stack", "extra:cap"},
            service_requires=mock_service.requires,
            api_key="secret_key_789",
        )

        async def full_stack_success():
            return await redaction_mw(
                req,
                ctx_success,
                lambda: metrics_mw(
                    req, ctx_success, lambda: mock_service.call(req, ctx=ctx_success)
                ),
            )

        result = await policy_mw(req, ctx_success, full_stack_success)

        # Verify successful execution
        assert result["status"] == "success"
        assert mock_service.call_count == 1

        # Verify metrics were captured
        success_logs = [r for r in caplog.records if "Service call completed" in r.message]
        assert len(success_logs) == 1

        # Verify redaction occurred (sensitive data should not appear in logs)
        all_log_text = " ".join([r.message for r in caplog.records])
        assert "secret_key_789" not in all_log_text, "Sensitive data leaked in logs"

    @pytest.mark.anyio
    async def test_middleware_execution_order_security(self):
        """Test that middleware execution order maintains security properties."""
        mock_service = MockService(requires={"order:test"})

        execution_order = []

        class TrackingPolicyMW(PolicyGateMW):
            async def _enforce_policy(self, req, ctx, next_call):
                execution_order.append("policy_start")
                try:
                    result = await super()._enforce_policy(req, ctx, next_call)
                    execution_order.append("policy_success")
                    return result
                except Exception as e:
                    execution_order.append("policy_fail")
                    raise

        class TrackingMetricsMW(MetricsMW):
            async def __call__(self, req, ctx, next_call):
                execution_order.append("metrics_start")
                try:
                    result = await super().__call__(req, ctx, next_call)
                    execution_order.append("metrics_success")
                    return result
                except Exception as e:
                    execution_order.append("metrics_fail")
                    raise

        tracking_policy = TrackingPolicyMW()
        tracking_metrics = TrackingMetricsMW()

        # Test failure case - policy should block before metrics
        ctx_fail = CallContext.new(
            branch_id=uuid4(),
            capabilities={"insufficient"},
            service_requires=mock_service.requires,
        )

        req = MockRequest()
        execution_order.clear()

        with pytest.raises(PolicyError):
            await tracking_policy(
                req,
                ctx_fail,
                lambda: tracking_metrics(
                    req, ctx_fail, lambda: mock_service.call(req, ctx=ctx_fail)
                ),
            )

        # Policy should fail before metrics even starts
        assert "policy_start" in execution_order
        assert "policy_fail" in execution_order
        assert "metrics_start" not in execution_order, "Metrics started despite policy failure"

        # Test success case - proper execution order
        ctx_success = CallContext.new(
            branch_id=uuid4(),
            capabilities={"order:test"},
            service_requires=mock_service.requires,
        )

        execution_order.clear()

        result = await tracking_policy(
            req,
            ctx_success,
            lambda: tracking_metrics(
                req, ctx_success, lambda: mock_service.call(req, ctx=ctx_success)
            ),
        )

        # Verify proper execution order
        expected_order = [
            "policy_start",
            "metrics_start",
            "metrics_success",
            "policy_success",
        ]
        assert execution_order == expected_order, f"Wrong execution order: {execution_order}"
