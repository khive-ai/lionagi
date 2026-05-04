# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class OpenAIResponsesRequest(BaseModel):
    """Request body for OpenAI Responses API.

    The endpoint accepts a broad set of structured fields; keep the schema
    explicit so provider/internal kwargs are filtered before the HTTP call.
    """

    model: str = Field(..., description="Model name.")
    input: str | list[Any] = Field(..., description="Input text or structured items.")
    instructions: str | None = None
    previous_response_id: str | None = None
    store: bool | None = None
    stream: bool | None = None
    include: list[str] | None = None
    metadata: dict[str, Any] | None = None
    user: str | None = None

    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    truncation: Literal["auto", "disabled"] | None = None

    reasoning: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None


__all__ = ("OpenAIResponsesRequest",)
