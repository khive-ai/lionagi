# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Middleware system for service pipeline composition."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, TypeVar

from lionagi import _err

from .core import CallContext
from .endpoint import RequestModel

# Type variables for middleware
Req = TypeVar("Req", bound=RequestModel)
Res = TypeVar("Res")
Chunk = TypeVar("Chunk")

# Middleware type definitions
CallMW = Callable[[Req, CallContext, Callable[[], Awaitable[Res]]], Awaitable[Res]]
StreamMW = Callable[[Req, CallContext, Callable[[], AsyncIterator[Chunk]]], AsyncIterator[Chunk]]

logger = logging.getLogger(__name__)


# Core middleware implementations


class PolicyGateMW:
    """Policy enforcement middleware - checks capabilities against requirements.

    Enforces capability-based security by validating that the call context
    has sufficient capabilities for the endpoint requirements. Fails closed
    on any policy check errors.
    """

    def __init__(self, *, strict: bool = True):
        self.strict = strict

    def __call__(
        self,
        req: RequestModel,
        ctx: CallContext,
        next_call: Callable[[], Awaitable[Any]],
    ) -> Awaitable[Any]:
        """Enforce policy gate for call operations."""
        return self._enforce_policy(req, ctx, next_call)

    def stream(
        self,
        req: RequestModel,
        ctx: CallContext,
        next_stream: Callable[[], AsyncIterator[Any]],
    ) -> AsyncIterator[Any]:
        """Enforce policy gate for streaming operations."""
        return self._enforce_policy_stream(req, ctx, next_stream)

    async def _enforce_policy(
        self,
        req: RequestModel,
        ctx: CallContext,
        next_call: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Check policy before allowing call to proceed."""
        # Get requirements from service (passed via context) and optional request extras
        required = self._get_required_capabilities(req, ctx)

        if not self._check_capabilities(ctx.capabilities, required):
            # Calculate missing capabilities for better debugging
            missing_capabilities = required - ctx.capabilities
            raise _err.PolicyError(
                f"Insufficient capabilities for operation. Missing: {missing_capabilities}",
                context={
                    "call_id": str(ctx.call_id),
                    "branch_id": str(ctx.branch_id),
                    "required_capabilities": sorted(required),
                    "available_capabilities": sorted(ctx.capabilities),
                    "missing_capabilities": sorted(missing_capabilities),
                    "operation": "call",
                    "policy_check": "capability_enforcement",
                },
            )

        return await next_call()

    async def _enforce_policy_stream(
        self,
        req: RequestModel,
        ctx: CallContext,
        next_stream: Callable[[], AsyncIterator[Any]],
    ) -> AsyncIterator[Any]:
        """Check policy before allowing stream to proceed."""
        # Get requirements from service (passed via context) and optional request extras
        required = self._get_required_capabilities(req, ctx)

        if not self._check_capabilities(ctx.capabilities, required):
            # Calculate missing capabilities for better debugging
            missing_capabilities = required - ctx.capabilities
            raise _err.PolicyError(
                f"Insufficient capabilities for streaming. Missing: {missing_capabilities}",
                context={
                    "call_id": str(ctx.call_id),
                    "branch_id": str(ctx.branch_id),
                    "required_capabilities": sorted(required),
                    "available_capabilities": sorted(ctx.capabilities),
                    "missing_capabilities": sorted(missing_capabilities),
                    "operation": "streaming",
                    "policy_check": "capability_enforcement",
                },
            )

        async for chunk in next_stream():
            yield chunk

    def _get_required_capabilities(self, req: RequestModel, ctx: CallContext) -> set[str]:
        """Get required capabilities from service declaration and optional request additions.

        Service requirements are authoritative (from ctx.attrs["service_requires"]).
        Request can add extra requirements but cannot replace service requirements.
        """
        # Service-declared requirements (source of truth)
        service_requires = set()
        if isinstance(ctx.attrs, (dict, Mapping)):
            raw_service_requires = ctx.attrs.get("service_requires", set())
            # Handle JSON serialization converting sets to lists
            if isinstance(raw_service_requires, (list, tuple)):
                service_requires = set(raw_service_requires)
            else:
                service_requires = set(raw_service_requires)

        # Optional additional requirements from request
        request_extras = getattr(req, "_extra_requires", None)
        if request_extras is None:
            request_extras = set()
        else:
            # Handle lists/tuples from serialization
            request_extras = set(request_extras)

        # Union - request can only add, not replace
        return service_requires | request_extras

    def _check_capabilities(self, available: set[str], required: set[str]) -> bool:
        """Check if available capabilities satisfy requirements.

        Supports wildcard matching on the available side only.
        """
        if not required:
            return True

        for req_cap in required:
            if not self._capability_covers(available, req_cap):
                return False
        return True

    def _capability_covers(self, available: set[str], required: str) -> bool:
        """Check if any available capability covers the required one."""
        # Exact match
        if required in available:
            return True

        # Wildcard matching - only on available side
        for avail_cap in available:
            if avail_cap.endswith("*"):
                # Handle malformed wildcards by stripping trailing *s
                wildcard_pattern = avail_cap.rstrip("*")

                # Special case: if pattern ends with separator (., :, /), match hierarchical capabilities
                # "api.*" should ONLY match "api.something", NOT bare "api" (strict matching)
                if wildcard_pattern.endswith((".", ":", "/")):
                    # Strict hierarchical matching - separator must be present
                    if required.startswith(wildcard_pattern):
                        return True
                else:
                    # Standard prefix matching
                    if required == wildcard_pattern or required.startswith(wildcard_pattern):
                        return True

        return False


class MetricsMW:
    """Metrics collection middleware.

    Collects call latency, success/failure rates, and other observability data.
    """

    def __init__(self, *, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)

    async def __call__(
        self,
        req: RequestModel,
        ctx: CallContext,
        next_call: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Collect metrics around call execution."""
        import time

        start_time = time.perf_counter()

        try:
            result = await next_call()
            duration = time.perf_counter() - start_time

            self.logger.info(
                "Service call completed",
                extra={
                    "call_id": str(ctx.call_id),
                    "branch_id": str(ctx.branch_id),
                    "duration_s": duration,
                    "status": "success",
                    "model": getattr(req, "model", None),
                },
            )

            return result

        except Exception as e:
            duration = time.perf_counter() - start_time

            self.logger.error(
                "Service call failed",
                extra={
                    "call_id": str(ctx.call_id),
                    "branch_id": str(ctx.branch_id),
                    "duration_s": duration,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "model": getattr(req, "model", None),
                },
                exc_info=True,
            )

            raise

    async def stream(
        self,
        req: RequestModel,
        ctx: CallContext,
        next_stream: Callable[[], AsyncIterator[Any]],
    ) -> AsyncIterator[Any]:
        """Collect metrics around streaming execution."""
        import time

        start_time = time.perf_counter()
        chunk_count = 0
        total_bytes = 0

        try:
            async for chunk in next_stream():
                chunk_count += 1
                if isinstance(chunk, (bytes, str)):
                    total_bytes += len(chunk)
                yield chunk

            duration = time.perf_counter() - start_time
            self.logger.info(
                "Service stream completed",
                extra={
                    "call_id": str(ctx.call_id),
                    "branch_id": str(ctx.branch_id),
                    "duration_s": duration,
                    "status": "success",
                    "chunk_count": chunk_count,
                    "total_bytes": total_bytes,
                    "model": getattr(req, "model", None),
                },
            )

        except Exception as e:
            duration = time.perf_counter() - start_time

            self.logger.error(
                "Service stream failed",
                extra={
                    "call_id": str(ctx.call_id),
                    "branch_id": str(ctx.branch_id),
                    "duration_s": duration,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "chunk_count": chunk_count,
                    "total_bytes": total_bytes,
                    "model": getattr(req, "model", None),
                },
                exc_info=True,
            )

            raise


class RedactionMW:
    """Redaction middleware for sensitive data.

    Redacts secrets and PII from logs and metrics to prevent accidental exposure.
    """

    SENSITIVE_HEADERS = {
        "authorization",
        "x-api-key",
        "api-key",
        "api_key",  # Support both dash and underscore variants
        "x-auth-token",
        "bearer",
    }

    def __init__(self, *, redact_model_inputs: bool = False):
        self.redact_model_inputs = redact_model_inputs

    async def __call__(
        self,
        req: RequestModel,
        ctx: CallContext,
        next_call: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Apply redaction to call context and request."""
        # Create redacted context for logging
        redacted_attrs = self._redact_dict(ctx.attrs)

        # Log redacted version
        logger.debug(
            "Service call starting",
            extra={
                "call_id": str(ctx.call_id),
                "branch_id": str(ctx.branch_id),
                "attrs": redacted_attrs,
                "model": (
                    getattr(req, "model", None) if not self.redact_model_inputs else "[REDACTED]"
                ),
            },
        )

        return await next_call()

    async def stream(
        self,
        req: RequestModel,
        ctx: CallContext,
        next_stream: Callable[[], AsyncIterator[Any]],
    ) -> AsyncIterator[Any]:
        """Apply redaction to streaming context."""
        redacted_attrs = self._redact_dict(ctx.attrs)

        logger.debug(
            "Service stream starting",
            extra={
                "call_id": str(ctx.call_id),
                "branch_id": str(ctx.branch_id),
                "attrs": redacted_attrs,
                "model": (
                    getattr(req, "model", None) if not self.redact_model_inputs else "[REDACTED]"
                ),
            },
        )

        async for chunk in next_stream():
            yield chunk

    def _redact_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact sensitive values from dictionary."""
        redacted = {}
        for key, value in data.items():
            if self._is_sensitive_key(key):
                redacted[key] = "[REDACTED]"
            elif isinstance(value, dict):
                redacted[key] = self._redact_dict(value)
            else:
                redacted[key] = value
        return redacted

    def _is_sensitive_key(self, key: str) -> bool:
        """Check if key contains sensitive data."""
        key_lower = key.lower()
        return (
            any(sensitive in key_lower for sensitive in self.SENSITIVE_HEADERS)
            or "password" in key_lower
            or "secret" in key_lower
            or "token" in key_lower
        )
