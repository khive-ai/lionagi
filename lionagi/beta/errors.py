# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Lion exception hierarchy with structured details and retryability.

All exceptions inherit from LionError and include:
    - message: Human-readable description
    - details: Structured context dict
    - retryable: Whether retry might succeed

Naming: LionConnectionError/LionTimeoutError avoid shadowing builtins.
"""

from __future__ import annotations

from typing import Any, Literal

from .protocols import Serializable, implements

__all__ = (
    "AccessError",
    "ConfigurationError",
    "ExecutionError",
    "ExistsError",
    "LionConnectionError",
    "LionTimeoutError",
    "LionError",
    "NotAllowedError",
    "NotFoundError",
    "OperationError",
    "QueueFullError",
    "ValidationError",
)


@implements(Serializable)
class LionError(Exception):
    """Base exception for lion. Serializable with structured details.

    Subclasses set default_message and default_retryable.
    Use cause= to chain exceptions with preserved traceback.

    Attributes:
        message: Human-readable description.
        details: Structured context for debugging/logging.
        retryable: Whether retry might succeed.
    """

    default_message: str = "lion error"
    default_retryable: bool = True

    def __init__(
        self,
        message: str | None = None,
        *,
        retryable: bool | None = None,
        cause: Exception | None = None,
        **details: dict[str, Any],
    ):
        """Initialize with optional message, details, retryability, and cause."""
        self.message = message or self.default_message
        self.details = details or {}
        self.retryable = retryable if retryable is not None else self.default_retryable

        if cause:
            self.__cause__ = cause  # Preserve traceback

        super().__init__(self.message)

    def to_dict(self, **kw: Any) -> dict[str, Any]:
        """Serialize to dict: {error, message, retryable, details?}."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "retryable": self.retryable,
            **({"details": self.details} if self.details else {}),
        }


class ValidationError(LionError):
    """Data validation failure. Raise when input fails schema/constraint checks."""

    default_message = "Validation failed"
    default_retryable = False


class AccessError(LionError):
    """Permission denied. Raise when capability/resource access is blocked."""

    default_message = "Access denied"
    default_retryable = False


class ConfigurationError(LionError):
    """Invalid configuration. Raise when setup/config is incorrect."""

    default_message = "Configuration error"
    default_retryable = False


class ExecutionError(LionError):
    """Execution failure. Raise when Event/Calling invoke fails (often transient)."""

    default_message = "Execution failed"
    default_retryable = True


class LionConnectionError(LionError):
    """Network/connection failure. Named to avoid shadowing builtins."""

    default_message = "Connection error"
    default_retryable = True


class LionTimeoutError(LionError):
    """Operation timeout. Named to avoid shadowing builtins."""

    default_message = "Operation timed out"
    default_retryable = True


class NotFoundError(LionError):
    """Resource/item not found. Raise when lookup fails."""

    default_message = "Item not found"
    default_retryable = False


class NotAllowedError(LionError):
    """Type/value not permitted by container constraints.

    Raise when an item fails a container's type or membership rules
    (e.g., Pile.item_type mismatch, capability not in allowed set).
    Distinct from AccessError (permission) and ValidationError (schema).
    """

    default_message = "Item not allowed"
    default_retryable = False


class ExistsError(LionError):
    """Duplicate item. Raise when creating item that already exists."""

    default_message = "Item already exists"
    default_retryable = False


class QueueFullError(LionError):
    """Capacity exceeded. Raise when queue/buffer is full."""

    default_message = "Queue is full"
    default_retryable = True


class OperationError(LionError):
    """Generic operation failure. Use for unclassified operation errors."""

    default_message = "Operation failed"
    default_retryable = False
