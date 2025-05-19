# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import Field, field_serializer

from lionagi.core_defs import Execution, ExecutionStatus # Import new Pydantic versions
from .element import Element

__all__ = (
    "Event",
    "Execution",       # Exporting the Pydantic version
    "ExecutionStatus", # Exporting the Pydantic version
)

# Old EventStatus and Execution class definitions are removed.

class Event(Element):
    """Extends Element with an execution state.

    Attributes:
        execution (Execution): The execution state of this event.
    """

    execution: Execution = Field(default_factory=Execution) # Now defaults to Pydantic Execution model
    streaming: bool = False

    @field_serializer("execution")
    def _serialize_execution(self, val: Execution) -> dict:
        """Serializes the Pydantic Execution object into a dictionary."""
        # Pydantic Execution model already serializes status enum to its value.
        # model_dump() is a standard way to get dict from Pydantic model.
        return val.model_dump(exclude_none=True) # exclude_none=True is often useful

    @property
    def response(self) -> Any: # response in Pydantic Execution is Optional[Dict]
        """Gets or sets the execution response.

        Returns:
            Any: The current response for this event.
        """
        return self.execution.response

    @response.setter
    def response(self, val: Any) -> None:
        """Sets the execution response.

        Args:
            val (Any): The new response value for this event.
        """
        self.execution.response = val

    @property
    def status(self) -> ExecutionStatus: # Changed to ExecutionStatus
        """Gets or sets the event status.

        Returns:
            ExecutionStatus: The current status of this event.
        """
        return self.execution.status

    @status.setter
    def status(self, val: ExecutionStatus) -> None: # Changed to ExecutionStatus
        """Sets the event status.

        Args:
            val (ExecutionStatus): The new status for the event.
        """
        self.execution.status = val

    @property
    def request(self) -> dict:
        """Gets the request for this event.

        Returns:
            dict: An empty dictionary by default. Override in subclasses
            if needed.
        """
        return {}

    async def invoke(self) -> None:
        """Performs the event action asynchronously.

        Raises:
            NotImplementedError: This base method must be overridden by
            subclasses.
        """
        raise NotImplementedError("Override in subclass.")

    async def stream(self) -> None:
        """Performs the event action asynchronously, streaming results.

        Raises:
            NotImplementedError: This base method must be overridden by
            subclasses.
        """
        raise NotImplementedError("Override in subclass.")

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        """Not implemented. Events cannot be fully recreated once done.

        Args:
            data (dict): Event data (unused).

        Raises:
            NotImplementedError: Always, because recreating an event is
            disallowed.
        """
        raise NotImplementedError("Cannot recreate an event once it's done.")


# File: lionagi/protocols/generic/event.py
