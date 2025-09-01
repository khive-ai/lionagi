# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Endpoint protocol for request building and wire-shape determination."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any, Protocol

import msgspec


class RequestModel(msgspec.Struct, kw_only=True):
    """Base request model with common fields for service endpoints.

    Using msgspec for v1 performance requirements - orders of magnitude
    faster than Pydantic for serialization/deserialization.
    """

    model: str | None = None
    stream: bool = False

    # Allow extra fields via forbid=False (msgspec default behavior)


class Endpoint(Protocol):
    """Request builder for deterministic wire-shape construction.

    Keep build() pure—deterministic and testable. Avoid side effects
    or environment reads here. This ensures request construction is
    reproducible and easy to test.
    """

    name: str
    method: str  # HTTP method like "POST"
    url: str  # fully resolved URL
    default_headers: Mapping[str, str]
    requires: set[str]  # capability requirements like {"net.out:api.openai.com"}

    def build(self, req: RequestModel) -> tuple[MutableMapping[str, str], dict]:
        """Build HTTP request from typed request model.

        Returns:
            tuple: (headers, json-payload) ready for transport

        This is a pure function - no IO, no side effects, deterministic.
        """
        ...

    def required_rights(self, **payload_params: Any) -> set[str]:
        """Compute dynamic capability requirements from payload parameters.

        For most endpoints this returns self.requires, but some endpoints
        like file operations may need dynamic rights based on paths.
        """
        return self.requires.copy()


# Standard request models for common service types


class ChatRequestModel(RequestModel, kw_only=True):
    """Request model for chat completion endpoints using msgspec."""

    messages: list[dict[str, Any]]
    model: str | None = None  # Model to use for completion
    temperature: float = 1.0
    max_tokens: int | None = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: str | list[str] | None = None


class CompletionRequestModel(RequestModel, kw_only=True):
    """Request model for text completion endpoints using msgspec."""

    prompt: str
    model: str | None = None  # Model to use for completion
    temperature: float = 1.0
    max_tokens: int | None = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: str | list[str] | None = None


class EmbeddingRequestModel(RequestModel, kw_only=True):
    """Request model for embedding endpoints using msgspec."""

    input: str | list[str]
    model: str | None = None  # Model to use for embeddings
    encoding_format: str = "float"
