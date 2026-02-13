# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

from pydantic import field_validator

from .message import MessageContent, MessageRole, RoledMessage


@dataclass(slots=True)
class ActionResponseContent(MessageContent):
    """Content for action/function call responses.

    Fields:
        function: Function name that was invoked
        arguments: Arguments used in the function call
        output: Result returned from the function
        action_request_id: Link to the original request
    """

    function: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    action_request_id: str | None = None

    @property
    def rendered(self) -> str:
        """Render action response as YAML."""
        from lionagi.libs.schema.minimal_yaml import minimal_yaml

        doc = {
            "Function": self.function,
            "Arguments": self.arguments,
            "Output": self.output,
        }
        return minimal_yaml(doc).strip()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionResponseContent":
        """Construct ActionResponseContent from dictionary."""
        # Handle nested structure from old format
        if "action_response" in data:
            resp = data["action_response"]
            function = resp.get("function", "")
            arguments = resp.get("arguments", {})
            output = resp.get("output")
        else:
            function = data.get("function", "")
            arguments = data.get("arguments", {})
            output = data.get("output")

        action_request_id = data.get("action_request_id")
        if action_request_id:
            action_request_id = str(action_request_id)

        return cls(
            function=function,
            arguments=arguments,
            output=output,
            action_request_id=action_request_id,
        )


class ActionResponse(RoledMessage):
    """Message containing the result of an action/function execution."""

    role: MessageRole = MessageRole.ACTION
    content: ActionResponseContent

    @field_validator("content", mode="before")
    def _validate_content(cls, v):
        if v is None:
            return ActionResponseContent()
        if isinstance(v, dict):
            return ActionResponseContent.from_dict(v)
        if isinstance(v, ActionResponseContent):
            return v
        raise TypeError("content must be dict or ActionResponseContent instance")

    @property
    def function(self) -> str:
        """Access the function name."""
        return self.content.function

    @property
    def arguments(self) -> dict[str, Any]:
        """Access the function arguments."""
        return self.content.arguments

    @property
    def output(self) -> Any:
        """Access the function output."""
        return self.content.output
