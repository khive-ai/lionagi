# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""OpenRouter models — extends OpenAI-compatible request with reasoning control."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from lionagi.providers.openai.chat.models import OpenAIChatCompletionsRequest

__all__ = ("ReasoningConfig", "OpenRouterRequest")


class ReasoningConfig(BaseModel):
    effort: Literal["none", "low", "medium", "high"] = "none"


class OpenRouterRequest(OpenAIChatCompletionsRequest):
    reasoning: ReasoningConfig | dict[str, Any] | None = Field(
        default=None,
        description="Reasoning/thinking config. Set {'effort':'none'} to disable thinking.",
    )
    include_reasoning: bool | None = Field(
        default=None,
        description="Whether to include reasoning tokens in response.",
    )
