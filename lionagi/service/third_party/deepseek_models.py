# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek OpenAI-compatible chat models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from lionagi.service.third_party.openai_models import OpenAIChatCompletionsRequest

DeepseekThinkingType = Literal["enabled", "disabled"]
DeepseekReasoningEffort = Literal["low", "medium", "high", "xhigh", "max"]


class DeepseekThinking(BaseModel):
    """DeepSeek thinking-mode switch."""

    type: DeepseekThinkingType = "enabled"


class DeepseekChatCompletionsRequest(OpenAIChatCompletionsRequest):
    """Request body for DeepSeek chat completions.

    DeepSeek is OpenAI-compatible for the common chat-completions surface, but
    supports additional thinking-mode parameters and a different highest
    reasoning effort value.
    """

    thinking: DeepseekThinking | None = Field(
        default=None,
        description="DeepSeek thinking-mode switch.",
    )
    reasoning_effort: DeepseekReasoningEffort | None = Field(
        default=None,
        description="DeepSeek reasoning effort; common effort values are mapped.",
    )

    @model_validator(mode="after")
    def _normalize_deepseek_reasoning(self):
        if self.reasoning_effort in {"low", "medium"}:
            self.reasoning_effort = "high"
        elif self.reasoning_effort in {"high", "xhigh"}:
            self.reasoning_effort = "max"
        return self


def normalize_deepseek_usage(response: Any) -> Any:
    """Expose DeepSeek reasoning usage as ``thinking_tokens``.

    DeepSeek responses may report reasoning usage under
    ``usage.completion_tokens_details.reasoning_tokens``. Older review notes and
    callers refer to this as ``thinking_tokens``; add that alias without
    discarding the provider-native fields.
    """
    if not isinstance(response, dict):
        return response

    usage = response.get("usage")
    if not isinstance(usage, dict):
        return response

    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        details = {}

    thinking_tokens = None
    for source, key in (
        (usage, "thinking_tokens"),
        (usage, "reasoning_tokens"),
        (details, "thinking_tokens"),
        (details, "reasoning_tokens"),
    ):
        if key in source:
            thinking_tokens = source[key]
            break
    if thinking_tokens is not None:
        usage["thinking_tokens"] = thinking_tokens
        usage.setdefault("reasoning_tokens", thinking_tokens)
        if isinstance(usage.get("completion_tokens_details"), dict):
            usage["completion_tokens_details"].setdefault(
                "thinking_tokens",
                thinking_tokens,
            )

    return response
