# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, ClassVar


class LionError(Exception):
    default_message: ClassVar[str] = "LionAGI error"
    status_code: ClassVar[int] = 500
    __slots__ = ("message", "details", "status_code")

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message or self.default_message)
        if cause:
            self.__cause__ = cause  # preserves traceback
        self.message = message or self.default_message
        self.details = details or {}
        self.status_code = status_code or self.status_code

    def to_dict(self, *, include_cause: bool = False) -> dict[str, Any]:
        data = {
            "error": self.__class__.__name__,
            "message": self.message,
            "status_code": self.status_code,
            **({"details": self.details} if self.details else {}),
        }
        if include_cause and (cause := self.get_cause()):
            data["cause"] = repr(cause)
        return data

    def get_cause(self) -> Exception | None:
        """Get the cause of this error, if any."""
        return self.__cause__ if hasattr(self, "__cause__") else None


class ValidationError(LionError):
    """Exception raised when validation fails."""

    default_message = "Validation failed"
    status_code = 422  # Unprocessable Entity
    __slots__ = ()  # no new attrs

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
        """Create a ValidationError from a value with optional expected type and message."""
        details = {
            "value": value,
            "type": type(value).__name__,
            **({"expected": expected} if expected else {}),
            **extra,
        }
        return cls(message=message, details=details, cause=cause)


class ItemNotFoundError(LionError):
    pass


class ItemExistsError(LionError):
    pass


class IDError(LionError):
    pass


class RelationError(LionError):
    pass


class RateLimitError(LionError):
    pass


class OperationError(LionError):
    pass


class ExecutionError(LionError):
    pass
