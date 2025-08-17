# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging

import aiohttp
import backoff
from aiocache import cached
from pydantic import BaseModel

from lionagi.config import settings
from lionagi.service.resilience import (
    CircuitBreaker,
    RetryConfig,
    retry_with_backoff,
)
from lionagi.utils import to_dict

from .endpoint_config import EndpointConfig
from .header_factory import HeaderFactory

logger = logging.getLogger(__name__)


class Endpoint:
    def __init__(
        self,
        config: dict | EndpointConfig,
        circuit_breaker: CircuitBreaker | None = None,
        retry_config: RetryConfig | None = None,
        **kwargs,
    ):
        """
        Initialize the endpoint.

        This endpoint is designed to be stateless and thread-safe for parallel operations.
        Each API call will create its own client session to avoid conflicts.

        Args:
            config: The endpoint configuration.
            circuit_breaker: Optional circuit breaker for resilience.
            retry_config: Optional retry configuration for resilience.
            **kwargs: Additional keyword arguments to update the configuration.
        """
        if isinstance(config, dict):
            _config = EndpointConfig(**config, **kwargs)
        elif isinstance(config, EndpointConfig):
            _config = config.model_copy(deep=True)
            _config.update(**kwargs)
        else:
            raise ValueError(
                "Config must be a dict or EndpointConfig instance"
            )
        self.config = _config
        self.circuit_breaker = circuit_breaker
        self.retry_config = retry_config

        logger.debug(
            f"Initialized Endpoint with provider={self.config.provider}, "
            f"endpoint={self.config.endpoint}, circuit_breaker={circuit_breaker is not None}, "
            f"retry_config={retry_config is not None}"
        )

    def _create_http_session(self):
        """Create a new HTTP session (not thread-safe, create new for each request)."""
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(self.config.timeout),
            **self.config.client_kwargs,
        )

    @property
    def request_options(self):
        return self.config.request_options

    @request_options.setter
    def request_options(self, value):
        self.config.request_options = EndpointConfig._validate_request_options(
            value
        )

    def create_payload(
        self,
        request: dict | BaseModel,
        extra_headers: dict | None = None,
        **kwargs,
    ):
        # First, create headers
        headers = HeaderFactory.get_header(
            auth_type=self.config.auth_type,
            content_type=self.config.content_type,
            api_key=self.config._api_key,
            default_headers=self.config.default_headers,
        )
        if extra_headers:
            headers.update(extra_headers)

        # Convert request to dict if it's a BaseModel
        request = (
            request
            if isinstance(request, dict)
            else request.model_dump(exclude_none=True)
        )

        # Start with config defaults
        payload = self.config.kwargs.copy()

        # Update with request data
        payload.update(request)

        # Update with additional kwargs
        if kwargs:
            payload.update(kwargs)

        # If we have request_options, use the model's fields to filter valid params
        if self.config.request_options is not None:
            # Get valid field names from the model
            valid_fields = set(self.config.request_options.model_fields.keys())

            # Filter payload to only include valid fields
            filtered_payload = {
                k: v for k, v in payload.items() if k in valid_fields
            }

            # Validate the filtered payload
            payload = self.config.validate_payload(filtered_payload)
        else:
            # If no request_options, we still need to remove obvious non-API params
            # These are parameters that are never part of any API payload
            non_api_params = {
                "task",
                "provider",
                "base_url",
                "endpoint",
                "endpoint_params",
                "api_key",
                "queue_capacity",
                "capacity_refresh_time",
                "interval",
                "limit_requests",
                "limit_tokens",
                "invoke_with_endpoint",
                "extra_headers",
                "headers",
                "cache_control",
                "include_token_usage_to_model",
                "chat_model",
                "imodel",
                "branch",
                "aggregation_sources",
                "aggregation_count",
            }
            payload = {
                k: v for k, v in payload.items() if k not in non_api_params
            }

        return (payload, headers)

    async def _call(self, payload: dict, headers: dict, **kwargs):
        return await self._call_aiohttp(
            payload=payload, headers=headers, **kwargs
        )

    async def call(
        self,
        request: dict | BaseModel,
        cache_control: bool = False,
        skip_payload_creation: bool = False,
        **kwargs,
    ):
        """
        Make a call to the endpoint.

        Args:
            request: The request parameters or model.
            cache_control: Whether to use cache control.
            skip_payload_creation: Whether to skip create_payload and treat request as ready payload.
            **kwargs: Additional keyword arguments for the request.

        Returns:
            The response from the endpoint.
        """
        # Extract extra_headers before passing to create_payload
        extra_headers = kwargs.pop("extra_headers", None)

        payload, headers = None, None
        if skip_payload_creation:
            # Treat request as the ready payload
            payload = (
                request if isinstance(request, dict) else request.model_dump()
            )
            headers = extra_headers or {}
        else:
            payload, headers = self.create_payload(
                request, extra_headers=extra_headers, **kwargs
            )

        # Apply resilience patterns if configured
        call_func = self._call

        # Apply retry if configured
        if self.retry_config:

            async def call_func(p, h, **kw):
                return await retry_with_backoff(
                    self._call, p, h, **kw, **self.retry_config.as_kwargs()
                )

        # Apply circuit breaker if configured
        if self.circuit_breaker:
            if self.retry_config:
                # If both are configured, apply circuit breaker to the retry-wrapped function
                if not cache_control:
                    return await self.circuit_breaker.execute(
                        call_func, payload, headers, **kwargs
                    )
            else:
                # If only circuit breaker is configured, apply it directly
                if not cache_control:
                    return await self.circuit_breaker.execute(
                        self._call, payload, headers, **kwargs
                    )

        # Handle caching if requested
        if cache_control:

            @cached(**settings.aiocache_config.as_kwargs())
            async def _cached_call(payload: dict, headers: dict, **kwargs):
                # Apply resilience patterns to cached call if configured
                if self.circuit_breaker and self.retry_config:
                    return await self.circuit_breaker.execute(
                        call_func, payload, headers, **kwargs
                    )
                if self.circuit_breaker:
                    return await self.circuit_breaker.execute(
                        self._call, payload, headers, **kwargs
                    )
                if self.retry_config:
                    return await call_func(payload, headers, **kwargs)

                return await self._call(payload, headers, **kwargs)

            return await _cached_call(payload, headers, **kwargs)

        # No caching, apply resilience patterns directly
        if self.retry_config:
            return await call_func(payload, headers, **kwargs)

        return await self._call(payload, headers, **kwargs)

    async def _call_aiohttp(self, payload: dict, headers: dict, **kwargs):
        """
        Make a call using aiohttp with a fresh session for each request.

        Args:
            payload: The request payload.
            headers: The request headers.
            **kwargs: Additional keyword arguments for the request.

        Returns:
            The response from the endpoint.
        """

        async def _make_request_with_backoff():
            # Create a new session for this request
            async with self._create_http_session() as session:
                response = None
                try:
                    response = await session.request(
                        method=self.config.method,
                        url=self.config.full_url,
                        headers=headers,
                        json=payload,
                        **kwargs,
                    )

                    # Check for rate limit or server errors that should be retried
                    if response.status == 429 or response.status >= 500:
                        response.raise_for_status()  # This will be caught by backoff
                    elif response.status != 200:
                        # Try to get error details from response body
                        try:
                            error_body = await response.json()
                            error_message = f"Request failed with status {response.status}: {error_body}"
                        except:
                            error_message = (
                                f"Request failed with status {response.status}"
                            )

                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=error_message,
                            headers=response.headers,
                        )

                    # Extract and return the JSON response
                    return await response.json()
                finally:
                    # Ensure response is properly released if coroutine is cancelled between retries
                    if response is not None and not response.closed:
                        await response.release()

        # Define a giveup function for backoff
        def giveup_on_client_error(e):
            # Don't retry on 4xx errors except 429 (rate limit)
            if isinstance(e, aiohttp.ClientResponseError):
                return 400 <= e.status < 500 and e.status != 429
            return False

        # Use backoff for retries with exponential backoff and jitter
        # Moved inside the method to reference runtime config
        backoff_handler = backoff.on_exception(
            backoff.expo,
            (aiohttp.ClientError, asyncio.TimeoutError),
            max_tries=self.config.max_retries,
            giveup=giveup_on_client_error,
            jitter=backoff.full_jitter,
        )

        # Apply the decorator at runtime
        return await backoff_handler(_make_request_with_backoff)()

    async def stream(
        self,
        request: dict | BaseModel,
        extra_headers: dict | None = None,
        **kwargs,
    ):
        """
        Stream responses from the endpoint.

        Args:
            request: The request parameters or model.
            extra_headers: Additional headers for the request.
            **kwargs: Additional keyword arguments for the request.

        Yields:
            Streaming chunks from the API.
        """
        payload, headers = self.create_payload(
            request, extra_headers, **kwargs
        )

        # Direct streaming without context manager
        async for chunk in self._stream_aiohttp(
            payload=payload, headers=headers, **kwargs
        ):
            yield chunk

    async def _stream_aiohttp(self, payload: dict, headers: dict, **kwargs):
        """
        Stream responses using aiohttp with a fresh session.

        Args:
            payload: The request payload.
            headers: The request headers.
            **kwargs: Additional keyword arguments for the request.

        Yields:
            Streaming chunks from the API.
        """
        # Ensure stream is enabled
        payload["stream"] = True

        # Create a new session for streaming
        async with self._create_http_session() as session:
            async with session.request(
                method=self.config.method,
                url=self.config.full_url,
                headers=headers,
                json=payload,
                **kwargs,
            ) as response:
                if response.status != 200:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"Request failed with status {response.status}",
                        headers=response.headers,
                    )

                async for line in response.content:
                    if line:
                        yield line.decode("utf-8")

    def to_dict(self):
        return {
            "retry_config": (
                self.retry_config.to_dict() if self.retry_config else None
            ),
            "circuit_breaker": (
                self.circuit_breaker.to_dict()
                if self.circuit_breaker
                else None
            ),
            "config": self.config.model_dump(exclude_none=True),
        }

    @classmethod
    def from_dict(cls, data: dict):
        data = to_dict(data, recursive=True)
        retry_config = data.get("retry_config")
        circuit_breaker = data.get("circuit_breaker")
        config = data.get("config")

        if retry_config:
            retry_config = RetryConfig(**retry_config)
        if circuit_breaker:
            circuit_breaker = CircuitBreaker(**circuit_breaker)
        if config:
            config = EndpointConfig(**config)

        return cls(
            config=config,
            circuit_breaker=circuit_breaker,
            retry_config=retry_config,
        )
