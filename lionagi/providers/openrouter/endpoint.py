# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from lionagi.providers.openai.chat.models import OpenAIChatCompletionsRequest
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig

from ._config import OpenRouterConfigs

__all__ = ("OpenRouterEndpoint",)


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


@OpenRouterConfigs.CHAT.register
class OpenRouterEndpoint(Endpoint):
    """OpenRouter API endpoint with reasoning control.

    Extends the standard OpenAI-compatible endpoint with:
    - reasoning effort control (none/low/medium/high)
    - reasoning token inclusion in response metadata
    """

    def __init__(
        self,
        config: EndpointConfig | None = None,
        **kwargs,
    ):
        if config is None:
            from lionagi.config import settings

            kwargs.setdefault(
                "api_key", settings.OPENROUTER_API_KEY or "dummy-key-for-testing"
            )
            kwargs.setdefault("kwargs", {"model": "google/gemini-2.5-flash"})
        super().__init__(config=config, **kwargs)
