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
from lionagi.ln.concurrency.patterns import retry
from lionagi.utils import to_dict

from .endpoint_config import EndpointConfig
from .header_factory import HeaderFactory

logger = logging.getLogger(__name__)


class Endpoint:
    def __init__(
        self,
        config: dict | EndpointConfig,
        retry_attempts: int = 3,
        retry_delay: float = 0.1,
        retry_max_delay: float = 2.0,
        retry_jitter: float = 0.1,
        retry_on: tuple[type[BaseException], ...] = (Exception,),
        **kwargs,
    ):
        """
        Initialize the endpoint.

        This endpoint is designed to be stateless and thread-safe for parallel operations.
        Each API call will create its own client session to avoid conflicts.

        Args:
            config: The endpoint configuration.
            retry_attempts: Maximum retry attempts for failed requests.
            retry_delay: Base delay between retries in seconds.
            retry_max_delay: Maximum delay between retries in seconds.
            retry_jitter: Random jitter factor (0.0 to 1.0) for retry delays.
            retry_on: Exception types that trigger retry.
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
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.retry_max_delay = retry_max_delay
        self.retry_jitter = retry_jitter
        self.retry_on = retry_on

        logger.debug(
            f"Initialized Endpoint with provider={self.config.provider}, "
            f"endpoint={self.config.endpoint}, retry_attempts={retry_attempts}"
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
                "action_strategy",
                "parse_model",
                "reason",
                "actions",
                "return_operative",
                "operative_model",
                "request_model",
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

        # Apply retry if configured (using our concurrency patterns)
        async def call_with_retry():
            return await retry(
                lambda: self._call(payload, headers, **kwargs),
                attempts=self.retry_attempts,
                base_delay=self.retry_delay,
                max_delay=self.retry_max_delay,
                retry_on=self.retry_on,
                jitter=self.retry_jitter,
            )

        # Handle caching if requested
        if cache_control:

            @cached(**settings.aiocache_config.as_kwargs())
            async def _cached_call(payload: dict, headers: dict, **kwargs):
                return await retry(
                    lambda: self._call(payload, headers, **kwargs),
                    attempts=self.retry_attempts,
                    base_delay=self.retry_delay,
                    max_delay=self.retry_max_delay,
                    retry_on=self.retry_on,
                    jitter=self.retry_jitter,
                )

            return await _cached_call(payload, headers, **kwargs)

        # No caching, apply retry directly
        return await call_with_retry()

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
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay,
            "retry_max_delay": self.retry_max_delay,
            "retry_jitter": self.retry_jitter,
            "config": self.config.model_dump(exclude_none=True),
        }

    @classmethod
    def from_dict(cls, data: dict):
        data = to_dict(data, recursive=True)
        config = data.get("config")

        if config:
            config = EndpointConfig(**config)

        return cls(
            config=config,
            retry_attempts=data.get("retry_attempts", 3),
            retry_delay=data.get("retry_delay", 0.1),
            retry_max_delay=data.get("retry_max_delay", 2.0),
            retry_jitter=data.get("retry_jitter", 0.1),
        )
