# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging

from anyio import get_cancelled_exc_class
from pydantic import Field, PrivateAttr, model_validator
from typing_extensions import Self

from lionagi.protocols.generic.event import Event, EventStatus
from lionagi.protocols.types import Log
from lionagi.service.hooks import HookEvent, HookEventTypes, global_hook_logger
from lionagi.service.token_calculator import TokenCalculator

from .endpoint import Endpoint

logger = logging.getLogger(__name__)


class APICalling(Event):
    """Handles asynchronous API calls with automatic token usage tracking.

    This class manages API calls through endpoints, handling both regular
    and streaming responses with optional token usage tracking.
    """

    endpoint: Endpoint = Field(
        ...,
        description="Endpoint instance for making the API call",
        exclude=True,
    )

    payload: dict = Field(
        ..., description="Request payload to send to the API"
    )

    headers: dict = Field(
        default_factory=dict,
        description="Additional headers for the request",
        exclude=True,
    )

    cache_control: bool = Field(
        default=False,
        description="Whether to use cache control for this request",
        exclude=True,
    )

    include_token_usage_to_model: bool = Field(
        default=False,
        description="Whether to include token usage information in messages",
        exclude=True,
    )

    _pre_invoke_hook_event: HookEvent = PrivateAttr(None)
    _post_invoke_hook_event: HookEvent = PrivateAttr(None)

    @model_validator(mode="after")
    def _validate_streaming(self) -> Self:
        """Validate streaming configuration and add token usage if requested."""
        if self.payload.get("stream") is True:
            self.streaming = True

        # Add token usage information to the last message if requested
        if (
            self.include_token_usage_to_model
            and self.endpoint.config.requires_tokens
        ):
            # Handle both messages format (chat completions) and input format (responses API)
            if "messages" in self.payload and isinstance(
                self.payload["messages"][-1], dict
            ):
                required_tokens = self.required_tokens
                content = self.payload["messages"][-1]["content"]
                # Model token limit mapping
                TOKEN_LIMITS = {
                    # OpenAI models
                    "gpt-4": 128_000,
                    "gpt-4-turbo": 128_000,
                    "o1-mini": 128_000,
                    "o1-preview": 128_000,
                    "o1": 200_000,
                    "o3": 200_000,
                    "gpt-4.1": 1_000_000,
                    # Anthropic models
                    "sonnet": 200_000,
                    "haiku": 200_000,
                    "opus": 200_000,
                    # Google models
                    "gemini": 1_000_000,
                    # Alibaba models
                    "qwen-turbo": 1_000_000,
                }

                token_msg = (
                    f"\n\nEstimated Current Token Usage: {required_tokens}"
                )

                # Find matching token limit
                if "model" in self.payload:
                    model = self.payload["model"]
                    for model_prefix, limit in TOKEN_LIMITS.items():
                        if model_prefix in model.lower():
                            token_msg += f"/{limit:,}"
                            break

                # Update content based on its type
                if isinstance(content, str):
                    content += token_msg
                elif isinstance(content, dict) and "text" in content:
                    content["text"] += token_msg
                elif isinstance(content, list):
                    for item in reversed(content):
                        if isinstance(item, dict) and "text" in item:
                            item["text"] += token_msg
                            break

                self.payload["messages"][-1]["content"] = content

        return self

    @property
    def required_tokens(self) -> int | None:
        """Calculate the number of tokens required for this request."""
        if not self.endpoint.config.requires_tokens:
            return None

        # Handle chat completions format
        if "messages" in self.payload:
            return TokenCalculator.calculate_message_tokens(
                self.payload["messages"], **self.payload
            )
        # Handle responses API format
        elif "input" in self.payload:
            # Convert input to messages format for token calculation
            input_val = self.payload["input"]
            if isinstance(input_val, str):
                messages = [{"role": "user", "content": input_val}]
            elif isinstance(input_val, list):
                # Handle array input format
                messages = []
                for item in input_val:
                    if isinstance(item, str):
                        messages.append({"role": "user", "content": item})
                    elif isinstance(item, dict) and "type" in item:
                        # Handle structured input items
                        if item["type"] == "message":
                            messages.append(item)
            else:
                return None
            return TokenCalculator.calculate_message_tokens(
                messages, **self.payload
            )
        # Handle embeddings endpoint
        elif "embed" in self.endpoint.config.endpoint:
            return TokenCalculator.calculate_embed_token(**self.payload)

        return None

    async def invoke(self) -> None:
        """Execute the API call through the endpoint.

        Updates execution status and stores the response or error.
        """
        start = asyncio.get_event_loop().time()

        try:
            self.execution.status = EventStatus.PROCESSING
            if h_ev := self._pre_invoke_hook_event:
                await h_ev.invoke()
                if h_ev._should_exit:
                    raise h_ev._exit_cause or RuntimeError(
                        "Pre-invocation hook requested exit without a cause"
                    )
                await global_hook_logger.alog(Log.create(h_ev))

            # Make the API call with skip_payload_creation=True since payload is already prepared
            response = await self.endpoint.call(
                request=self.payload,
                cache_control=self.cache_control,
                skip_payload_creation=True,
                extra_headers=self.headers if self.headers else None,
            )

            if h_ev := self._post_invoke_hook_event:
                await h_ev.invoke()
                if h_ev._should_exit:
                    raise h_ev._exit_cause or RuntimeError(
                        "Post-invocation hook requested exit without a cause"
                    )
                await global_hook_logger.alog(Log.create(h_ev))

            self.execution.response = response
            self.execution.status = EventStatus.COMPLETED

        except get_cancelled_exc_class():
            self.execution.error = "API call cancelled"
            self.execution.status = EventStatus.CANCELLED
            raise

        except Exception as e:
            self.execution.error = str(e)
            self.execution.status = EventStatus.FAILED
            logger.error(f"API call failed: {e}")

        finally:
            self.execution.duration = asyncio.get_event_loop().time() - start

    async def stream(self):
        """Stream the API response through the endpoint.

        Yields:
            Streaming chunks from the API.
        """
        start = asyncio.get_event_loop().time()
        response = []

        try:
            self.execution.status = EventStatus.PROCESSING

            async for chunk in self.endpoint.stream(
                request=self.payload,
                extra_headers=self.headers if self.headers else None,
            ):
                response.append(chunk)
                yield chunk

            self.execution.response = response
            self.execution.status = EventStatus.COMPLETED

        except Exception as e:
            self.execution.error = str(e)
            self.execution.status = EventStatus.FAILED
            logger.error(f"Streaming failed: {e}")

        finally:
            self.execution.duration = asyncio.get_event_loop().time() - start

    @property
    def request(self) -> dict:
        """Get request information including token usage."""
        return {
            "required_tokens": self.required_tokens,
        }

    @property
    def response(self):
        """Get the response from the execution."""
        return self.execution.response if self.execution else None

    def create_pre_invoke_hook(
        self,
        hook_registry,
        exit_hook: bool = None,
        hook_timeout: float = 30.0,
        hook_params: dict = None,
    ):
        h_ev = HookEvent(
            hook_type=HookEventTypes.PreInvokation,
            event_like=self,
            registry=hook_registry,
            exit=exit_hook,
            timeout=hook_timeout,
            params=hook_params or {},
        )
        self._pre_invoke_hook_event = h_ev

    def create_post_invoke_hook(
        self,
        hook_registry,
        exit_hook: bool = None,
        hook_timeout: float = 30.0,
        hook_params: dict = None,
    ):
        h_ev = HookEvent(
            hook_type=HookEventTypes.PostInvokation,
            event_like=self,
            registry=hook_registry,
            exit=exit_hook,
            timeout=hook_timeout,
            params=hook_params or {},
        )
        self._post_invoke_hook_event = h_ev
