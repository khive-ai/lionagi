# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

from lionagi.protocols.generic.event import EventStatus

from .api_calling import APICalling
from .endpoint import Endpoint
from .match_endpoint import match_endpoint
from .rate_limited_processor import RateLimitedAPIExecutor


class iModel:
    
    def __init__(
        self,
        provider: str = None,
        endpoint: Endpoint | str = None,
        queue_capacity: int = 100,
        capacity_refresh_time: float = 60,
        interval: float | None = None,
        limit_requests: int = None,
        limit_tokens: int = None,
        concurrency_limit: int | None = None,
        api_key: str = None,
        **kwargs,
    ):

        model = kwargs.get("model", None)
        if model:
            if not provider:
                if "/" in model:
                    provider = model.split("/")[0]
                    model = model.replace(provider + "/", "")
                    kwargs["model"] = model
                else:
                    raise ValueError("Provider must be provided")

        if api_key is None:
            provider = str(provider or "").strip().lower()
            match provider:
                case "openai":
                    api_key = "OPENAI_API_KEY"
                case "anthropic":
                    api_key = "ANTHROPIC_API_KEY"
                case "openrouter":
                    api_key = "OPENROUTER_API_KEY"
                case "perplexity":
                    api_key = "PERPLEXITY_API_KEY"
                case "groq":
                    api_key = "GROQ_API_KEY"
                case "exa":
                    api_key = "EXA_API_KEY"
                case "ollama":
                    api_key = "ollama"
                case _:
                    api_key = f"{provider.upper()}_API_KEY"

        if os.getenv(api_key, None) is not None:
            self.api_key_scheme = api_key
            api_key = os.getenv(api_key)

        kwargs["api_key"] = api_key
        if isinstance(endpoint, Endpoint):
            self.endpoint = endpoint
        else:
            self.endpoint = match_endpoint(
                provider=provider,
                endpoint=endpoint,
            )

        self.kwargs = kwargs
        self.executor = RateLimitedAPIExecutor(
            queue_capacity=queue_capacity,
            capacity_refresh_time=capacity_refresh_time,
            interval=interval,
            limit_requests=limit_requests,
            limit_tokens=limit_tokens,
            concurrency_limit=concurrency_limit,
        )

    async def invoke(self, api_call: APICalling) -> APICalling | None:
        """Invokes a rate-limited API call with the given arguments.

        Args:
            **kwargs:
                Arguments for the request, merged with self.kwargs.

        Returns:
            APICalling | None:
                The `APICalling` object if successfully invoked and
                completed; otherwise None.

        Raises:
            ValueError:
                If the call fails or if an error occurs during invocation.
        """
        try:
            if (
                self.executor.processor is None
                or self.executor.processor.is_stopped()
            ):
                await self.executor.start()

            await self.executor.append(api_call)
            await self.executor.forward()
            ctr = 0
            while api_call.execution.status not in (
                EventStatus.COMPLETED,
                EventStatus.FAILED,
            ):
                if ctr > 100:
                    break
                await self.executor.forward()
                ctr += 1
                await asyncio.sleep(0.1)
            return self.executor.pile.pop(api_call.id)
        except Exception as e:
            raise RuntimeError(f"Failed to invoke API call: {e}")
