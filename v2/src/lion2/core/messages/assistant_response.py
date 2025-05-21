# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import BaseModel, Field

from ..types import MessageContent

__all__ = ("AssistantResponseContent",)


class AssistantResponseContent(MessageContent):

    name: str | None = None
    text_response: str | None = None

    # Exclude model response, API calls gets logged separately
    model_response: dict | list[dict] | None = Field(None, exclude=True)

    def __init__(
        self,
        model_response: Any,
        name: str | None = None,
    ):
        text_response, model_response = _prepare_assistant_response(
            model_response
        )
        super().__init__(
            text_response=text_response,
            model_response=model_response,
            name=name,
        )

    def update(self, model_response: Any = None, name=None) -> None:
        if model_response is not None:
            text_response, model_response = _prepare_assistant_response(
                model_response
            )
            self.text_response = text_response
            self.model_response = model_response
        if name is not None:
            self.name = name

    @property
    def rendered(self) -> str:
        str_ = "## Assistant Message\n"
        if self.name:
            str_ += f"- name: {self.name}\n"

        str_ += f"- content:\n\n{self.text_response}\n"
        return str_


def _prepare_assistant_response(
    model_response: Any,
) -> tuple[str, dict | list[dict] | None]:
    model_response = (
        [model_response]
        if not isinstance(model_response, list)
        else model_response
    )

    text_contents = []
    model_responses = []

    for i in model_response:
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

    # model_responses is a list that might contain dicts, lists of dicts, or strings (if original input was simple string)

    final_structured_response = None
    if not model_responses:
        final_structured_response = None

    elif len(model_responses) == 1:
        if not isinstance(model_responses[0], str):
            final_structured_response = model_responses[0]
        # If it is a string, final_structured_response remains None, which is correct.
    else:
        # If there are multiple items, it should be a list of dicts.
        # Filter out any stray strings, though ideally model_responses should only contain structured items here.
        structured_items = [
            item for item in model_responses if isinstance(item, (dict, list))
        ]
        if structured_items:
            final_structured_response = (
                structured_items  # Return as list if multiple structured items
            )
        # If, after filtering, structured_items is empty (e.g. original was list of strings), it remains None.

    return (text_contents, final_structured_response)
