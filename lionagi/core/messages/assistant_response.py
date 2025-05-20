# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .types import MessageContent

__all__ = ("AssistantResponseContent",)


class AssistantResponseContent(MessageContent):
    created_at: str | None = Field(None, exclude=True)
    name: str | None = None
    time: str | None = None
    text_response: str | None = None
    model_response: dict | list[dict] | None = None

    def __init__(
        self,
        assistant_response: BaseModel | list[BaseModel] | dict | str | Any,
        name: str | None = None,
        created_at: str | None = None,
    ):
        text_response, model_response = _prepare_assistant_response(
            assistant_response
        )
        super().__init__(
            text_response=text_response,
            model_response=model_response,
            name=name,
            created_at=created_at,
        )

    @field_validator("created_at")
    def _validate_created_at_to_str(
        cls, v: str | datetime | int | None
    ) -> str:
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, datetime):
            return v.isoformat(timespec="seconds")
        if isinstance(v, int):
            try:
                return datetime.fromtimestamp(v).isoformat(timespec="seconds")
            except Exception:
                raise ValueError(
                    f"Invalid timestamp: {v}. Must be a valid datetime or timestamp."
                )

    def update(
        self, assistant_response: Any = None, name=None, created_at: Any = None
    ) -> None:
        if assistant_response is not None:
            text_response, model_response = _prepare_assistant_response(
                assistant_response
            )
            self.text_response = text_response
            self.model_response = model_response
        if name is not None:
            self.name = name
        if created_at is not None:
            self.created_at = self._validate_created_at_to_str(created_at)

    @property
    def rendered(self) -> str:
        str_ = "## Assistant Message\n"
        if self.name:
            str_ += f"- name: {self.name}\n"
        if self.created_at:
            str_ += f"- time: {self.created_at}\n"

        rendered += f"- content:\n\n{self.text_response}\n"
        return rendered


def _prepare_assistant_response(
    assistant_response: BaseModel | list[BaseModel] | dict | str | Any, /
) -> dict:
    assistant_response = (
        [assistant_response]
        if not isinstance(assistant_response, list)
        else assistant_response
    )

    text_contents = []
    model_responses = []

    for i in assistant_response:
        if isinstance(i, BaseModel):
            i = i.model_dump(exclude_none=True, exclude_unset=True)

        model_responses.append(i)

        if isinstance(i, dict):
            # anthropic standard
            if "content" in i:
                content = i["content"]
                content = (
                    [content] if not isinstance(content, list) else content
                )
                for j in content:
                    if isinstance(j, dict):
                        if j.get("type") == "text":
                            text_contents.append(j["text"])
                    elif isinstance(j, str):
                        text_contents.append(j)

            # openai standard
            elif "choices" in i:
                choices = i["choices"]
                choices = (
                    [choices] if not isinstance(choices, list) else choices
                )
                for j in choices:
                    if "message" in j:
                        text_contents.append(j["message"]["content"] or "")
                    elif "delta" in j:
                        text_contents.append(j["delta"]["content"] or "")

        elif isinstance(i, str):
            text_contents.append(i)

    text_contents = "".join(text_contents)
    model_responses = (
        model_responses[0] if len(model_responses) == 1 else model_responses
    )
    return (text_contents, model_responses)
