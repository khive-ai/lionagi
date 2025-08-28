# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""OpenAI SDK-based service implementation - works with any OpenAI-compatible API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import openai
from openai import AsyncOpenAI

from lionagi.ln.concurrency import fail_at

from lionagi.errors import (
    NonRetryableError,
    RateLimitError,
    RetryableError,
    ServiceError,
    TimeoutError,
)
from ..core import CallContext
from ..endpoint import RequestModel
from ..middleware import CallMW, StreamMW


@dataclass(slots=True)
class OpenAICompatibleService:
    """OpenAI SDK-based service that works with any OpenAI-compatible API.

    Supports:
    - OpenAI (api.openai.com)
    - Anthropic via OpenAI compatibility (api.anthropic.com)
    - OpenRouter, Together, Fireworks, etc.
    - Local APIs like Ollama, LM Studio, vLLM

    Features:
    - Deadline-aware operations
    - Capability-based security via middleware
    - Clean error handling and retries
    - Proper streaming with backpressure
    """

    client: AsyncOpenAI
    name: str = "openai_compatible"
    requires: set[str] = frozenset({"net.out:*"})  # Override for specific hosts
    call_mw: tuple[CallMW, ...] = ()
    stream_mw: tuple[StreamMW, ...] = ()

    async def call(self, req: RequestModel, *, ctx: CallContext) -> dict:
        """Execute single chat completion call."""

        async def do_call() -> dict:
            """Core call operation using OpenAI SDK."""
            # Convert our request model to SDK parameters
            kwargs = self._build_call_kwargs(req, ctx)

            try:
                response = await self.client.chat.completions.create(**kwargs)
                return response.model_dump()

            except TimeoutError as e:
                # Handle both lionagi.errors.TimeoutError and asyncio.TimeoutError
                from lionagi.errors import TimeoutError as LionagiTimeoutError

                raise LionagiTimeoutError(
                    f"OpenAI API call timed out: {e}",
                    context={
                        "call_id": str(ctx.call_id),
                        "service": self.name,
                        "model": kwargs.get("model"),
                        "timeout": kwargs.get("timeout"),
                    },
                    cause=e,
                )
            except openai.RateLimitError as e:
                # Use specific RateLimitError for better handling
                retry_after = getattr(e, "retry_after", None) or 60.0
                raise RateLimitError(
                    retry_after=retry_after,
                    message=f"OpenAI API rate limited: {e}",
                    context={
                        "call_id": str(ctx.call_id),
                        "service": self.name,
                        "model": kwargs.get("model"),
                        "retry_after": retry_after,
                        "status_code": getattr(e, "status_code", None),
                    },
                    cause=e,
                )
            except openai.APITimeoutError as e:
                # Handle OpenAI SDK timeout errors
                from lionagi.errors import TimeoutError as LionagiTimeoutError

                raise LionagiTimeoutError(
                    f"OpenAI API timeout error: {e}",
                    context={
                        "call_id": str(ctx.call_id),
                        "service": self.name,
                        "model": kwargs.get("model"),
                        "error_type": "api_timeout",
                        "timeout": kwargs.get("timeout"),
                    },
                    cause=e,
                )
            except openai.APIConnectionError as e:
                raise RetryableError(
                    f"OpenAI API connection error: {e}",
                    context={
                        "call_id": str(ctx.call_id),
                        "service": self.name,
                        "model": kwargs.get("model"),
                        "error_type": "connection",
                    },
                    cause=e,
                )
            except openai.InternalServerError as e:
                raise RetryableError(
                    f"OpenAI API server error: {e}",
                    context={
                        "call_id": str(ctx.call_id),
                        "service": self.name,
                        "model": kwargs.get("model"),
                        "error_type": "server_error",
                        "status_code": getattr(e, "status_code", None),
                    },
                    cause=e,
                )
            except openai.BadRequestError as e:
                raise NonRetryableError(
                    f"OpenAI API bad request: {e}",
                    context={
                        "call_id": str(ctx.call_id),
                        "service": self.name,
                        "model": kwargs.get("model"),
                        "error_type": "bad_request",
                        "status_code": getattr(e, "status_code", None),
                    },
                    cause=e,
                )
            except openai.AuthenticationError as e:
                raise NonRetryableError(
                    f"OpenAI API authentication failed: {e}",
                    context={
                        "call_id": str(ctx.call_id),
                        "service": self.name,
                        "model": kwargs.get("model"),
                        "error_type": "authentication",
                        "status_code": getattr(e, "status_code", None),
                    },
                    cause=e,
                )
            except openai.PermissionDeniedError as e:
                raise NonRetryableError(
                    f"OpenAI API permission denied: {e}",
                    context={
                        "call_id": str(ctx.call_id),
                        "service": self.name,
                        "model": kwargs.get("model"),
                        "error_type": "permission_denied",
                        "status_code": getattr(e, "status_code", None),
                    },
                    cause=e,
                )
            except openai.NotFoundError as e:
                raise NonRetryableError(
                    f"OpenAI API resource not found: {e}",
                    context={
                        "call_id": str(ctx.call_id),
                        "service": self.name,
                        "model": kwargs.get("model"),
                        "error_type": "not_found",
                        "status_code": getattr(e, "status_code", None),
                    },
                    cause=e,
                )
            except openai.OpenAIError as e:
                raise ServiceError(
                    f"OpenAI API error: {e}",
                    context={
                        "call_id": str(ctx.call_id),
                        "service": self.name,
                        "model": kwargs.get("model"),
                        "error_type": "openai_api",
                        "status_code": getattr(e, "status_code", None),
                    },
                    cause=e,
                )

        # Apply middleware chain
        async def invoke(i: int = 0) -> dict:
            if i >= len(self.call_mw):
                return await do_call()
            return await self.call_mw[i](req, ctx, lambda: invoke(i + 1))

        # Execute with deadline enforcement
        if ctx.deadline_s is None:
            return await invoke()
        else:
            with fail_at(ctx.deadline_s):
                return await invoke()

    async def stream(self, req: RequestModel, *, ctx: CallContext) -> AsyncIterator[dict]:
        """Execute streaming chat completion call."""

        async def do_stream() -> AsyncIterator[dict]:
            """Core streaming operation using OpenAI SDK."""
            # Convert our request model to SDK parameters (force stream=True)
            kwargs = self._build_call_kwargs(req, ctx, stream=True)

            try:
                stream = await self.client.chat.completions.create(**kwargs)

                async for chunk in stream:
                    yield chunk.model_dump()

            except TimeoutError as e:
                # Handle both lionagi.errors.TimeoutError and asyncio.TimeoutError
                from lionagi.errors import TimeoutError as LionagiTimeoutError

                raise LionagiTimeoutError(
                    f"OpenAI API stream timed out: {e}",
                    context={
                        "call_id": str(ctx.call_id),
                        "service": self.name,
                        "model": kwargs.get("model"),
                        "operation": "streaming",
                        "timeout": kwargs.get("timeout"),
                    },
                    cause=e,
                )
            except openai.RateLimitError as e:
                retry_after = getattr(e, "retry_after", None) or 60.0
                raise RateLimitError(
                    retry_after=retry_after,
                    message=f"OpenAI API stream rate limited: {e}",
                    context={
                        "call_id": str(ctx.call_id),
                        "service": self.name,
                        "model": kwargs.get("model"),
                        "operation": "streaming",
                        "retry_after": retry_after,
                    },
                    cause=e,
                )
            except openai.APITimeoutError as e:
                # Handle OpenAI SDK timeout errors in streaming
                from lionagi.errors import TimeoutError as LionagiTimeoutError

                raise LionagiTimeoutError(
                    f"OpenAI API stream timeout error: {e}",
                    context={
                        "call_id": str(ctx.call_id),
                        "service": self.name,
                        "model": kwargs.get("model"),
                        "operation": "streaming",
                        "error_type": "api_timeout",
                        "timeout": kwargs.get("timeout"),
                    },
                    cause=e,
                )
            except openai.APIConnectionError as e:
                raise RetryableError(
                    f"OpenAI API stream connection error: {e}",
                    context={
                        "call_id": str(ctx.call_id),
                        "service": self.name,
                        "model": kwargs.get("model"),
                        "operation": "streaming",
                        "error_type": "connection",
                    },
                    cause=e,
                )
            except openai.InternalServerError as e:
                raise RetryableError(
                    f"OpenAI API stream server error: {e}",
                    context={
                        "call_id": str(ctx.call_id),
                        "service": self.name,
                        "model": kwargs.get("model"),
                        "operation": "streaming",
                        "error_type": "server_error",
                        "status_code": getattr(e, "status_code", None),
                    },
                    cause=e,
                )
            except openai.BadRequestError as e:
                raise NonRetryableError(
                    f"OpenAI API stream bad request: {e}",
                    context={
                        "call_id": str(ctx.call_id),
                        "service": self.name,
                        "model": kwargs.get("model"),
                        "operation": "streaming",
                        "error_type": "bad_request",
                        "status_code": getattr(e, "status_code", None),
                    },
                    cause=e,
                )
            except openai.OpenAIError as e:
                raise ServiceError(
                    f"OpenAI API stream error: {e}",
                    context={
                        "call_id": str(ctx.call_id),
                        "service": self.name,
                        "model": kwargs.get("model"),
                        "operation": "streaming",
                        "error_type": "openai_api",
                        "status_code": getattr(e, "status_code", None),
                    },
                    cause=e,
                )

        # Apply middleware chain for streaming
        async def invoke_stream(i: int = 0) -> AsyncIterator[dict]:
            if i >= len(self.stream_mw):
                async for chunk in do_stream():
                    yield chunk
                return

            async for chunk in self.stream_mw[i](req, ctx, lambda: invoke_stream(i + 1)):
                yield chunk

        # Execute with deadline enforcement
        if ctx.deadline_s is None:
            async for chunk in invoke_stream():
                yield chunk
        else:
            with fail_at(ctx.deadline_s):
                async for chunk in invoke_stream():
                    yield chunk

    def _build_call_kwargs(
        self, req: RequestModel, ctx: CallContext, *, stream: bool = False
    ) -> dict[str, Any]:
        """Build OpenAI SDK call parameters from request model."""
        kwargs = {}

        # Core parameters
        if hasattr(req, "model") and req.model:
            kwargs["model"] = req.model
        if hasattr(req, "messages") and req.messages:
            kwargs["messages"] = req.messages

        # Streaming
        kwargs["stream"] = stream or getattr(req, "stream", False)

        # Optional parameters - only include if explicitly set
        for field, default in [
            ("temperature", 1.0),
            ("top_p", 1.0),
            ("frequency_penalty", 0.0),
            ("presence_penalty", 0.0),
        ]:
            value = getattr(req, field, default)
            if value != default:
                kwargs[field] = value

        # Parameters that should be included if present
        for field in ["max_tokens", "stop"]:
            if hasattr(req, field):
                value = getattr(req, field)
                if value is not None:
                    kwargs[field] = value

        # Add timeout from call context
        if ctx.remaining_time is not None:
            kwargs["timeout"] = max(1.0, ctx.remaining_time)

        # Handle extra fields from pydantic models
        if hasattr(req, "__pydantic_extra__"):
            kwargs.update(req.__pydantic_extra__)

        return kwargs


# Factory functions for common providers


def create_openai_service(
    api_key: str,
    *,
    organization: str | None = None,
    call_mw: tuple[CallMW, ...] = (),
    stream_mw: tuple[StreamMW, ...] = (),
    **client_kwargs,
) -> OpenAICompatibleService:
    """Create service for OpenAI API."""
    client = AsyncOpenAI(api_key=api_key, organization=organization, **client_kwargs)

    return OpenAICompatibleService(
        client=client,
        name="openai",
        requires=frozenset({"net.out:api.openai.com"}),
        call_mw=call_mw,
        stream_mw=stream_mw,
    )


def create_anthropic_service(
    api_key: str,
    *,
    call_mw: tuple[CallMW, ...] = (),
    stream_mw: tuple[StreamMW, ...] = (),
    **client_kwargs,
) -> OpenAICompatibleService:
    """Create service for Anthropic API via OpenAI compatibility layer."""
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.anthropic.com/v1/messages",  # Anthropic's OpenAI-compatible endpoint
        **client_kwargs,
    )

    return OpenAICompatibleService(
        client=client,
        name="anthropic",
        requires=frozenset({"net.out:api.anthropic.com"}),
        call_mw=call_mw,
        stream_mw=stream_mw,
    )


def create_openrouter_service(
    api_key: str,
    *,
    call_mw: tuple[CallMW, ...] = (),
    stream_mw: tuple[StreamMW, ...] = (),
    **client_kwargs,
) -> OpenAICompatibleService:
    """Create service for OpenRouter API."""
    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", **client_kwargs)

    return OpenAICompatibleService(
        client=client,
        name="openrouter",
        requires=frozenset({"net.out:openrouter.ai"}),
        call_mw=call_mw,
        stream_mw=stream_mw,
    )


def create_together_service(
    api_key: str,
    *,
    call_mw: tuple[CallMW, ...] = (),
    stream_mw: tuple[StreamMW, ...] = (),
    **client_kwargs,
) -> OpenAICompatibleService:
    """Create service for Together AI API."""
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.together.xyz/v1", **client_kwargs)

    return OpenAICompatibleService(
        client=client,
        name="together",
        requires=frozenset({"net.out:api.together.xyz"}),
        call_mw=call_mw,
        stream_mw=stream_mw,
    )


def create_ollama_service(
    *,
    base_url: str = "http://localhost:11434/v1",
    call_mw: tuple[CallMW, ...] = (),
    stream_mw: tuple[StreamMW, ...] = (),
    **client_kwargs,
) -> OpenAICompatibleService:
    """Create service for local Ollama API."""
    client = AsyncOpenAI(
        api_key="ollama",  # Ollama doesn't need real API key
        base_url=base_url,
        **client_kwargs,
    )

    return OpenAICompatibleService(
        client=client,
        name="ollama",
        requires=frozenset({"net.out:localhost:11434"}),  # Adjust port as needed
        call_mw=call_mw,
        stream_mw=stream_mw,
    )


def create_generic_service(
    api_key: str,
    base_url: str,
    *,
    name: str = "generic",
    host_capability: str | None = None,
    call_mw: tuple[CallMW, ...] = (),
    stream_mw: tuple[StreamMW, ...] = (),
    **client_kwargs,
) -> OpenAICompatibleService:
    """Create service for any OpenAI-compatible API."""
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, **client_kwargs)

    # Infer capability from base_url if not provided
    if host_capability is None:
        from urllib.parse import urlparse

        parsed = urlparse(base_url)
        host_capability = f"net.out:{parsed.netloc}"

    return OpenAICompatibleService(
        client=client,
        name=name,
        requires=frozenset({host_capability}),
        call_mw=call_mw,
        stream_mw=stream_mw,
    )
