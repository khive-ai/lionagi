# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator, Callable

from pydantic import BaseModel

from lionagi.protocols.generic.event import EventStatus
from lionagi.utils import is_coro_func

from .connections.api_calling import APICalling
from .connections.endpoint import Endpoint
from .connections.match_endpoint import match_endpoint
from .rate_limited_processor import RateLimitedAPIExecutor


class iModel:
    """Manages API calls for a specific provider with optional rate-limiting.

    The iModel class encapsulates a specific endpoint configuration (e.g.,
    chat or completion endpoints). It determines and sets the necessary
    API key based on the provider and uses a RateLimitedAPIExecutor to
    handle queuing and throttling requests.

    Attributes:
        endpoint (Endpoint):
            The chosen endpoint object (constructed via `match_endpoint` if
            none is provided).
        executor (RateLimitedAPIExecutor):
            The rate-limited executor that queues and runs API calls in a
            controlled fashion.
    """

    def __init__(
        self,
        provider: str = None,
        base_url: str = None,
        endpoint: str | Endpoint = "chat",
        api_key: str = None,
        queue_capacity: int = 100,
        capacity_refresh_time: float = 60,
        interval: float | None = None,
        limit_requests: int = None,
        limit_tokens: int = None,
        concurrency_limit: int | None = None,
        streaming_process_func: Callable = None,
        **kwargs,
    ) -> None:
        """Initializes the iModel instance.

        Args:
            provider (str, optional):
                Name of the provider (e.g., 'openai', 'anthropic').
            base_url (str, optional):
                Base URL for the API (if a custom endpoint is needed).
            endpoint (str | Endpoint, optional):
                Either a string representing the endpoint type (e.g., 'chat')
                or an `Endpoint` instance.
            endpoint_params (list[str] | None, optional):
                Additional parameters for the endpoint (e.g., 'v1' or other).
            api_key (str, optional):
                An explicit API key. If not given, tries to load one from
                environment variables based on the provider.
            queue_capacity (int, optional):
                Maximum number of requests allowed in the queue before
                executing them.
            capacity_refresh_time (float, optional):
                Time interval (in seconds) after which the queue capacity
                is refreshed.
            interval (float | None, optional):
                Interval in seconds to check or process requests in
                the queue. If None, defaults to capacity_refresh_time.
            limit_requests (int | None, optional):
                Maximum number of requests allowed per cycle, if any.
            limit_tokens (int | None, optional):
                Maximum number of tokens allowed per cycle, if any.
            concurrency_limit (int | None, optional):
                Maximum number of streaming concurrent requests allowed.
                only applies to streaming requests.
            **kwargs:
                Additional keyword arguments, such as `model`, or any other
                provider-specific fields.
        """
        model = kwargs.get("model", None)
        if model:
            if not provider:
                if "/" in model:
                    provider = model.split("/")[0]
                    model = model.replace(provider + "/", "")
                    kwargs["model"] = model
                else:
                    raise ValueError("Provider must be provided")

        # Pass api_key to endpoint if provided
        if api_key is not None:
            kwargs["api_key"] = api_key
        if isinstance(endpoint, Endpoint):
            self.endpoint = endpoint
        else:
            self.endpoint = match_endpoint(
                provider=provider,
                endpoint=endpoint,
                **kwargs,
            )
        if provider:
            self.endpoint.config.provider = provider
        if base_url:
            self.endpoint.config.base_url = base_url

        self.executor = RateLimitedAPIExecutor(
            queue_capacity=queue_capacity,
            capacity_refresh_time=capacity_refresh_time,
            interval=interval,
            limit_requests=limit_requests,
            limit_tokens=limit_tokens,
            concurrency_limit=concurrency_limit,
        )
        # Use provided streaming_process_func or default to None
        self.streaming_process_func = streaming_process_func

    def create_api_calling(
        self, include_token_usage_to_model: bool = False, **kwargs
    ) -> APICalling:
        """Constructs an `APICalling` object from endpoint-specific payload.

        Args:
            **kwargs:
                Additional arguments used to generate the payload.

        Returns:
            APICalling:
                An `APICalling` instance with the constructed payload,
                headers, and the selected endpoint.
        """
        # The new Endpoint.create_payload returns (payload, headers)
        payload, headers = self.endpoint.create_payload(request=kwargs)

        # Extract cache_control if provided
        cache_control = kwargs.pop("cache_control", False)

        return APICalling(
            payload=payload,
            headers=headers,
            endpoint=self.endpoint,
            cache_control=cache_control,
            include_token_usage_to_model=include_token_usage_to_model,
        )

    async def process_chunk(self, chunk) -> None:
        """Processes a chunk of streaming data.

        Override this method in subclasses if you need custom handling
        of streaming responses from the API.

        Args:
            chunk:
                A portion of the streamed data returned by the API.
        """
        if self.streaming_process_func and not isinstance(chunk, APICalling):
            if is_coro_func(self.streaming_process_func):
                return await self.streaming_process_func(chunk)
            return self.streaming_process_func(chunk)

    async def stream(
        self,
        api_call=None,
        include_token_usage_to_model: bool = False,
        **kwargs,
    ) -> AsyncGenerator:
        """Performs a streaming API call with the given arguments.

        Args:
            **kwargs:
                Arguments for the request, merged with self.kwargs.

        Returns:
            `APICalling` | None:
                An APICalling instance upon success, or None if something
                goes wrong.
        """
        if api_call is None:
            kwargs["stream"] = True
            api_call = self.create_api_calling(
                include_token_usage_to_model=include_token_usage_to_model,
                **kwargs,
            )
            await self.executor.append(api_call)

        if (
            self.executor.processor is None
            or self.executor.processor.is_stopped()
        ):
            await self.executor.start()

        if self.executor.processor._concurrency_sem:
            async with self.executor.processor._concurrency_sem:
                try:
                    async for i in api_call.stream():
                        result = await self.process_chunk(i)
                        if result:
                            yield result
                except Exception as e:
                    raise ValueError(f"Failed to stream API call: {e}")
                finally:
                    yield self.executor.pile.pop(api_call.id)
        else:
            try:
                async for i in api_call.stream():
                    result = await self.process_chunk(i)
                    if result:
                        yield result
            except Exception as e:
                raise ValueError(f"Failed to stream API call: {e}")
            finally:
                yield self.executor.pile.pop(api_call.id)

    async def invoke(
        self, api_call: APICalling = None, **kwargs
    ) -> APICalling | None:
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
            if api_call is None:
                kwargs.pop("stream", None)
                api_call = self.create_api_calling(**kwargs)
            if (
                self.executor.processor is None
                or self.executor.processor.is_stopped()
            ):
                await self.executor.start()

            await self.executor.append(api_call)
            await self.executor.forward()
            ctr = 0
            while api_call.status not in (
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
            raise ValueError(f"Failed to invoke API call: {e}")

    @property
    def allowed_roles(self) -> set[str]:
        """list[str]: Roles that are permissible for this endpoint.

        Returns:
            If the endpoint has an `allowed_roles` attribute, returns that;
            otherwise, defaults to `{"system", "user", "assistant"}`.
        """
        if hasattr(self.endpoint, "allowed_roles"):
            return self.endpoint.allowed_roles
        return {"system", "user", "assistant"}

    @property
    def model_name(self) -> str:
        """str: The name of the model used by the endpoint.

        Returns:
            The model name if available; otherwise, an empty string.
        """
        return self.endpoint.config.kwargs.get("model", "")

    @property
    def request_options(self) -> type[BaseModel] | None:
        """type[BaseModel] | None: The request options model for the endpoint.

        Returns:
            The request options model if available; otherwise, None.
        """
        return self.endpoint.request_options

    def to_dict(self):
        return {
            "endpoint": self.endpoint.to_dict(),
            "processor_config": self.executor.config,
        }

    @classmethod
    def from_dict(cls, data: dict):
        endpoint = Endpoint.from_dict(data.get("endpoint", {}))

        e1 = match_endpoint(
            provider=endpoint.config.provider,
            endpoint=endpoint.config.endpoint,
        )

        e1.config = endpoint.config

        return cls(
            endpoint=e1,
            **data.get("processor_config", {}),
        )
