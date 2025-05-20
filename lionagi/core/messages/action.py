# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import json
from uuid import UUID

from pydantic import field_serializer, field_validator
from pydapter.protocols.utils import validate_uuid

from .types import MessageContent, MessageRole, ToolRequest, ToolResponse


class ActionRequestContent(MessageContent):
    action_response_id: UUID | None = None
    role: MessageRole = MessageRole.ACTION
    requests: list[ToolRequest]

    @field_validator("action_response_id", mode="before")
    def _validate_action_response_id(cls, v: UUID | str | None) -> UUID | None:
        return validate_uuid(v) if v else None

    @field_serializer("action_response_id")
    def _serialize_action_response_id(self, v) -> str | None:
        return str(v) if v else None

    @field_validator("requests", mode="before")
    def _validate_requests(cls, v: list | dict | str) -> list[ToolRequest]:
        v = [v] if not isinstance(v, list) else v
        requests = []
        for item in v:
            if isinstance(item, dict):
                requests.append(ToolRequest.model_validate(item))
            elif isinstance(item, ToolRequest):
                requests.append(item)
            elif isinstance(item, str):
                requests.append(ToolRequest.model_validate_json(item))
            else:
                raise ValueError("Invalid tool request format.")
        return requests

    @property
    def rendered(self) -> dict:
        str_ = "## Action Requests\n"
        for idx, item in enumerate(self.requests):
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
            self.requests.extend(requests)
        else:
            self.requests = requests


class ActionResponseContent(MessageContent):
    role: MessageRole = MessageRole.ACTION
    action_request_id: UUID
    responses: list[ToolResponse]

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
            self.responses.extend(responses)
        else:
            self.responses = responses

    @property
    def rendered(self) -> dict:
        str_ = "## Action Responses\n"
        str_ += f"- Action Request ID: {str(self.action_request_id)}\n"
        for idx, item in enumerate(self.responses):
            str_ += f"- Response {idx + 1}:\n"
            str_ += f"  - Function: {item.function}\n"
            str_ += (
                f"  - Arguments: {json.dumps(item.arguments, indent=2)}\n\n"
            )

        return str_
