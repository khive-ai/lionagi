# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import sys
from enum import Enum as _Enum
from typing import Any

# ExceptionGroup is available in Python 3.11+; use exceptiongroup backport for 3.10
if sys.version_info < (3, 11):  # noqa: UP036
    try:
        from exceptiongroup import ExceptionGroup  # noqa: A004
    except ImportError:
        ExceptionGroup = None  # type: ignore[misc,assignment]  # noqa: A001
else:
    ExceptionGroup = ExceptionGroup  # noqa: PLW0127, A001

from pydantic import Field, field_serializer

from lionagi import ln
from lionagi.utils import Unset, to_dict

from .element import Element

__all__ = (
    "EventStatus",
    "Execution",
    "Event",
)


_SIMPLE_TYPE = (str, bytes, bytearray, int, float, type(None), _Enum)


class EventStatus(str, ln.types.Enum):
    """Status states for tracking action execution progress.

    Attributes:
        PENDING: Initial state before execution starts.
        PROCESSING: Action is currently being executed.
        COMPLETED: Action completed successfully.
        FAILED: Action failed during execution.
        SKIPPED: Action was skipped due to unmet conditions.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    ABORTED = "aborted"


class Execution:
    """Represents the execution state of an event.

    Attributes:
        status (`EventStatus`): The current status of the event execution.
        duration (float | None): Time (in seconds) the execution took,
            if known.
        response (Any): The result or output of the execution, if any.
        error (str | None): An error message if the execution failed.
        retryable (bool | None): Whether a retry is safe after failure.
    """

    __slots__ = ("status", "duration", "response", "error", "retryable")

    def __init__(
        self,
        duration: float | None = None,
        response: Any = None,
        status: EventStatus = EventStatus.PENDING,
        error: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        """Initializes an execution instance.

        Args:
            duration (float | None): The duration of the execution.
            response (Any): The result or output of the execution.
            status (EventStatus): The current status (default is PENDING).
            error (str | None): An optional error message.
            retryable (bool | None): Whether retry is safe (default None).
        """
        self.status = status
        self.duration = duration
        self.response = response
        self.error = error
        self.retryable = retryable

    def __str__(self) -> str:
        """Returns a string representation of the execution state.

        Returns:
            str: A descriptive string indicating status, duration, response,
            error, and retryable.
        """
        return (
            f"Execution(status={self.status.value}, duration={self.duration}, "
            f"response={self.response}, error={self.error}, "
            f"retryable={self.retryable})"
        )

    def to_dict(self) -> dict:
        """Converts the execution state to a dictionary.

        Returns:
            dict: A dictionary representation of the execution state.
        """
        res_ = Unset
        json_serializable = True

        if not isinstance(self.response, _SIMPLE_TYPE):
            json_serializable = False
            try:
                # check whether response is JSON serializable
                ln.json_dumps(self.response)
                res_ = self.response
                json_serializable = True
            except Exception:
                with contextlib.suppress(Exception):
                    # attempt to force convert to dict
                    d_ = to_dict(
                        self.response,
                        recursive=True,
                        recursive_python_only=False,
                        use_enum_values=True,
                    )
                    ln.json_dumps(d_)
                    res_ = d_
                    json_serializable = True

        if res_ is Unset and not json_serializable:
            res_ = "<unserializable>"

        error_value = self.error
        if isinstance(self.error, BaseException):
            if ExceptionGroup is not None and isinstance(self.error, ExceptionGroup):
                error_value = self._serialize_exception_group(self.error)
            else:
                error_value = {
                    "error": type(self.error).__name__,
                    "message": str(self.error),
                }

        return {
            "status": self.status.value,
            "duration": self.duration,
            "response": res_ or self.response,
            "error": error_value,
            "retryable": self.retryable,
        }

    def _serialize_exception_group(
        self,
        eg: ExceptionGroup,
        depth: int = 0,
        _seen: set[int] | None = None,
    ) -> dict[str, Any]:
        """Recursively serialize ExceptionGroup with depth limit and cycle detection.

        Args:
            eg: ExceptionGroup to serialize.
            depth: Current recursion depth (internal).
            _seen: Object IDs already visited for cycle detection (internal).

        Returns:
            Dict with error type, message, and nested exceptions.
        """
        max_depth = 100
        if depth > max_depth:
            return {
                "error": "ExceptionGroup",
                "message": f"Max nesting depth ({max_depth}) exceeded",
                "nested_count": len(eg.exceptions) if hasattr(eg, "exceptions") else 0,
            }

        if _seen is None:
            _seen = set()

        eg_id = id(eg)
        if eg_id in _seen:
            return {
                "error": "ExceptionGroup",
                "message": "Circular reference detected",
            }

        _seen.add(eg_id)

        try:
            exceptions = []
            for exc in eg.exceptions:
                if isinstance(exc, ExceptionGroup):
                    exceptions.append(self._serialize_exception_group(exc, depth + 1, _seen))
                else:
                    exceptions.append(
                        {
                            "error": type(exc).__name__,
                            "message": str(exc),
                        }
                    )

            return {
                "error": type(eg).__name__,
                "message": str(eg),
                "exceptions": exceptions,
            }
        finally:
            _seen.discard(eg_id)

    def add_error(self, exc: BaseException) -> None:
        """Add error; creates ExceptionGroup if multiple errors accumulated.

        On Python 3.10 without the ``exceptiongroup`` backport, multiple
        errors are stored as a plain list in a wrapper Exception.

        Args:
            exc: The exception to add.
        """
        if self.error is None:
            self.error = exc
        elif ExceptionGroup is not None and isinstance(self.error, ExceptionGroup):
            self.error = ExceptionGroup(
                "multiple errors",
                [*self.error.exceptions, exc],
            )
        elif isinstance(self.error, BaseException):
            if ExceptionGroup is not None:
                self.error = ExceptionGroup(
                    "multiple errors",
                    [self.error, exc],
                )
            else:
                # Fallback for Python 3.10 without exceptiongroup
                self.error = Exception(f"multiple errors: {self.error}, {exc}")
        else:
            # error is a string or other non-exception type
            self.error = exc


class Event(Element):
    """Extends Element with an execution state.

    Attributes:
        execution (Execution): The execution state of this event.
    """

    execution: Execution = Field(default_factory=Execution)
    streaming: bool = Field(False, exclude=True)

    @field_serializer("execution")
    def _serialize_execution(self, val: Execution) -> dict:
        """Serializes the Execution object into a dictionary."""
        return val.to_dict()

    @property
    def response(self) -> Any:
        """Gets or sets the execution response."""
        return self.execution.response

    @response.setter
    def response(self, val: Any) -> None:
        """Sets the execution response."""
        self.execution.response = val

    @property
    def status(self) -> EventStatus:
        """Gets or sets the event status."""
        return self.execution.status

    @status.setter
    def status(self, val: EventStatus | str) -> None:
        """Sets the event status."""
        if isinstance(val, str):
            if val not in EventStatus.allowed():
                raise ValueError(f"Invalid status: {val}")
            val = EventStatus(val)
        if isinstance(val, EventStatus):
            self.execution.status = val
        else:
            raise ValueError(f"Invalid status type: {type(val)}. Expected EventStatus or str.")

    @property
    def request(self) -> dict:
        """Gets the request for this event. Override in subclasses"""
        return {}

    async def invoke(self) -> None:
        """Performs the event action asynchronously."""
        raise NotImplementedError("Override in subclass.")

    async def stream(self) -> None:
        """Performs the event action asynchronously, streaming results."""
        raise NotImplementedError("Override in subclass.")

    @classmethod
    def from_dict(cls, data: dict) -> Event:
        """Not implemented. Events cannot be fully recreated once done."""
        raise NotImplementedError("Cannot recreate an event once it's done.")

    def assert_completed(self) -> None:
        """Assert the event completed successfully.

        Raises:
            RuntimeError: If the event status is not COMPLETED, with
                execution details in the message.
        """
        if self.execution.status != EventStatus.COMPLETED:
            exec_dict = self.execution.to_dict()
            exec_dict.pop("response", None)
            raise RuntimeError(f"Event did not complete successfully: {exec_dict}")

    def as_fresh_event(self, copy_meta: bool = False) -> Event:
        """Creates a clone of this event with the same execution state."""
        d_ = self.to_dict()
        for i in ["execution", "created_at", "id", "metadata"]:
            d_.pop(i, None)
        fresh = self.__class__(**d_)
        if copy_meta:
            meta = self.metadata.copy()
            fresh.metadata = meta
        fresh.metadata["original"] = {
            "id": str(self.id),
            "created_at": self.created_at,
        }
        return fresh


# File: lionagi/protocols/generic/event.py
