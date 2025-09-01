# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from enum import Enum as _Enum
from typing import Any

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
    """

    __slots__ = ("status", "duration", "response", "error")

    def __init__(
        self,
        duration: float | None = None,
        response: Any = None,
        status: EventStatus = EventStatus.PENDING,
        error: str | None = None,
    ) -> None:
        """Initializes an execution instance.

        Args:
            duration (float | None): The duration of the execution.
            response (Any): The result or output of the execution.
            status (EventStatus): The current status (default is PENDING).
            error (str | None): An optional error message.
        """
        self.status = status
        self.duration = duration
        self.response = response
        self.error = error

    def __str__(self) -> str:
        """Returns a string representation of the execution state.

        Returns:
            str: A descriptive string indicating status, duration, response,
            and error.
        """
        return (
            f"Execution(status={self.status.value}, duration={self.duration}, "
            f"response={self.response}, error={self.error})"
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

        return {
            "status": self.status.value,
            "duration": self.duration,
            "response": res_ or self.response,
            "error": self.error,
        }


class Event(Element):
    """Extends Element with an execution state.

    Attributes:
        execution (Execution): The execution state of this event.
    """

    execution: Execution = Field(default_factory=Execution)
    streaming: bool = False

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
            raise ValueError(
                f"Invalid status type: {type(val)}. Expected EventStatus or str."
            )

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
