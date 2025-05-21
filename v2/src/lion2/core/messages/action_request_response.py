# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import json
from uuid import UUID

import lionfuncs as ln
from pydantic import field_serializer, field_validator
from pydapter.protocols.utils import validate_uuid

from ..types import MessageContent, ToolRequest, ToolResponse


class ActionRequestContent(MessageContent):

    action_response_id: UUID | None = None
    tool_requests: list[ToolRequest]

    @field_validator("action_response_id", mode="before")
    def _validate_action_response_id(cls, v: UUID | str | None) -> UUID | None:
        return validate_uuid(v) if v else None

    @field_serializer("action_response_id")
    def _serialize_action_response_id(self, v) -> str | None:
        return str(v) if v else None

    @field_validator("tool_requests", mode="before")
    def _validate_tool_requests(
        cls, v: list | dict | str | ToolRequest
    ) -> list[ToolRequest]:
        if isinstance(v, ToolRequest):
            return [v]

        # attempt to parse JSON into a list of dictionaries
        if isinstance(v, str):
            v: list[dict] | dict = ln.to_json(v)

        # If it's a dictionary, we assume it's a single tool request
        if isinstance(v, dict):
            return [ToolRequest.model_validate(v)]

        # If it's a list, we assume it contains multiple tool requests
        if isinstance(v, list):
            validated_list = []
            for item in v:
                if isinstance(item, dict):
                    validated_list.append(ToolRequest.model_validate(item))
                elif isinstance(item, ToolRequest):
                    validated_list.append(item)
                else:
                    raise ValueError(
                        f"Invalid item type in 'requests' list: {type(item)}"
                    )
            return validated_list

        # This case should ideally not be reached if type hints are respected,
        raise ValueError(f"Invalid input type for 'requests': {type(v)}")

    @property
    def rendered(self) -> dict:
        str_ = "## Action Requests\n"
        for idx, item in enumerate(self.tool_requests):
            str_ += f"- Request {idx + 1}:\n"
            str_ += f"  - Function: {item.function}\n"
            str_ += (
                f"  - Arguments: {json.dumps(item.arguments, indent=2)}\n\n"
            )

        return str_

    def update(
        self, requests: list[ToolRequest] | dict | str, append: bool = False
    ) -> None:
        requests = self._validate_requests(requests)
        if append:
            self.tool_requests.extend(requests)
        else:
            self.tool_requests = requests


class ActionResponseContent(MessageContent):
    action_request_id: UUID
    tool_responses: list[ToolResponse]

    @field_validator("action_request_id", mode="before")
    def _validate_action_request_id(cls, v: UUID | str) -> UUID:
        return validate_uuid(v)

    @field_serializer("action_request_id")
    def _serialize_action_request_id(self, v) -> str:
        return str(v)

    def update(
        self, responses: list[ToolResponse], append: bool = False
    ) -> None:
        if append:
            self.tool_responses.extend(responses)
        else:
            self.tool_responses = responses

    @property
    def rendered(self) -> dict:
        str_ = "## Action Responses\n"
        str_ += f"- Action Request ID: {str(self.action_request_id)}\n"
        for idx, item in enumerate(self.tool_responses):
            str_ += f"- Response {idx + 1}:\n"
            str_ += f"  - Function: {item.function}\n"
            str_ += (
                f"  - Arguments: {json.dumps(item.arguments, indent=2)}\n\n"
            )

        return str_
