# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Unified error hierarchy for the entire lionagi ecosystem.

Based on v0 foundation with ChatGPT/Gemini enhancements for v1.
Provides structured error handling with behavioral classification,
rich context, machine-readable codes, and observability integration.
"""

from __future__ import annotations

from typing import Any, ClassVar

__all__ = (
    "LionError",
    "ServiceError",
    "RetryableError",
    "NonRetryableError",
    "TimeoutError",
    "TransportError",
    "PolicyError",
    "ValidationError",
    "NotFoundError",
    "ExistsError",
    "ObservationError",
    "ResourceError",
    "RateLimitError",
    "IDError",
    "RelationError",
    "OperationError",
    "ExecutionError",
    "ItemNotFoundError",
    "ItemExistsError",
)


class LionError(Exception):
    """Unified base for all lionagi errors.

    Provides structured error handling with:
    - Behavioral classification (retryable/non-retryable)
    - Rich context and observability
    - Machine-readable error codes
    - Standard serialization for logging/monitoring

    Based on v0 foundation with ChatGPT/Gemini enhancements for v1.
    """

    default_message: ClassVar[str] = "LionAGI error"
    default_status_code: ClassVar[int] = 500
    retryable: ClassVar[bool] = False  # Behavioral classification for resilience middleware
    code: ClassVar[str] = "lion_error"  # Machine-readable error code
    severity: ClassVar[str] = "error"  # Log severity level

    __slots__ = ("message", "details", "status_code", "context", "__cause__")

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message or self.default_message)
        if cause:
            self.__cause__ = cause  # Preserves traceback chain
        self.message = message or self.default_message
        self.details = details or {}
        self.status_code = status_code or type(self).default_status_code
        self.context = context or {}

    def to_dict(self, *, include_cause: bool = False) -> dict[str, Any]:
        """Serialize error to structured dictionary for logging/monitoring."""
        data = {
            "error": self.__class__.__name__,
            "code": type(self).code,
            "message": self.message,
            "status_code": self.status_code,
            "retryable": type(self).retryable,
            "severity": type(self).severity,
            **({"details": self.details} if self.details else {}),
            **({"context": self.context} if self.context else {}),
        }
        if include_cause and (cause := self.get_cause()):
            data["cause"] = repr(cause)
        return data

    def get_cause(self) -> Exception | None:
        """Get the cause of this error, if any."""
        return self.__cause__ if hasattr(self, "__cause__") else None

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        expected: str | None = None,
        message: str | None = None,
        cause: Exception | None = None,
        **extra: Any,
    ):
        """Create error from a value with optional expected type and message."""
        details = {
            "value": value,
            "type": type(value).__name__,
            **({"expected": expected} if expected else {}),
            **extra,
        }
        return cls(message=message, details=details, cause=cause)


# Behavioral base classes - used by resilience middleware
class ServiceError(LionError):
    """Base service error - general service operation failure."""

    code = "service_error"


class RetryableError(LionError):
    """Error that can be retried (5xx, network, rate limits, timeouts)."""

    retryable = True
    severity = "warning"  # Often transient
    code = "retryable_error"


class NonRetryableError(LionError):
    """Error that should not be retried (4xx except 429, auth failures, validation errors)."""

    retryable = False
    severity = "error"
    code = "non_retryable_error"


# Service-specific errors
class TimeoutError(RetryableError):
    """Operation timed out - retryable with backoff."""

    default_status_code = 504
    code = "timeout"


class TransportError(RetryableError):
    """Transport-level error (network, HTTP, parsing) - usually retryable."""

    code = "transport_error"

    def __init__(self, message: str, status_code: int | None = None, **kwargs):
        super().__init__(message, status_code=status_code, **kwargs)


class PolicyError(NonRetryableError):
    """Policy check failed - insufficient capabilities, non-retryable."""

    default_status_code = 403
    code = "policy_denied"


# Domain-specific errors - classified for behavioral patterns
class ValidationError(NonRetryableError):
    """Validation failed - non-retryable client error."""

    default_message = "Validation failed"
    default_status_code = 422
    code = "validation_failed"


class NotFoundError(NonRetryableError):
    """Resource not found - non-retryable client error."""

    default_message = "Resource not found"
    default_status_code = 404
    code = "not_found"


class ExistsError(NonRetryableError):
    """Resource already exists - non-retryable conflict."""

    default_message = "Resource already exists"
    default_status_code = 409
    code = "resource_exists"


class ResourceError(RetryableError):
    """Resource access error - often retryable."""

    default_message = "Resource access error"
    default_status_code = 429
    code = "resource_error"


class RateLimitError(RetryableError):
    """Rate limit exceeded - retryable with backoff."""

    default_message = "Rate limit exceeded"
    default_status_code = 429
    code = "rate_limited"

    __slots__ = ("retry_after",)

    def __init__(self, retry_after: float, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "retry_after", retry_after)


class OperationError(NonRetryableError):
    """Operation failed - non-retryable business logic error."""

    code = "operation_failed"


class ExecutionError(NonRetryableError):
    """Execution failed - non-retryable runtime error."""

    code = "execution_failed"


class ObservationError(LionError):
    """Observation failed - logging/monitoring error."""

    default_message = "Observation failed"
    code = "observation_failed"


class IDError(NonRetryableError):
    """Invalid or missing ID - non-retryable."""

    code = "invalid_id"


class RelationError(NonRetryableError):
    """Relationship constraint violation - non-retryable."""

    code = "relation_error"


# Aliases for backwards compatibility
ItemNotFoundError = NotFoundError
ItemExistsError = ExistsError
