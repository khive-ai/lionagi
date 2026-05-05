# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""HTTP endpoint backend for beta resources.

Provides HTTP/REST API integration with:
- EndpointConfig: URL, auth, headers, request validation
- Endpoint: HTTP client with circuit breaker and retry support
- APICalling: Event wrapper for HTTP requests with token estimation

Security:
    API keys resolved from environment variables or passed as SecretStr.
    Raw credentials cleared from config to prevent serialization leaks.
    System env vars (PATH, HOME, etc.) blocked to prevent collision.

Example:
    config = EndpointConfig(
        provider="openai",
        name="gpt-4",
        base_url="https://api.openai.com/v1",
        endpoint="chat/completions",
        api_key="OPENAI_API_KEY",  # env var name
        request_options=ChatRequest,
    )
    endpoint = Endpoint(config=config)
    calling = APICalling(backend=endpoint, payload={...})
    await calling.invoke()
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, TypeVar

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    SecretStr,
    field_serializer,
    field_validator,
    model_validator,
)

from lionagi.service.connections.header_factory import AUTH_TYPES, HeaderFactory
from lionagi.service.resilience import CircuitBreaker, RetryConfig, retry_with_backoff

from .backend import Calling, Normalized, ResourceBackend, ResourceConfig

logger = logging.getLogger(__name__)

# Blocked env vars to prevent collision with system paths/config.
SYSTEM_ENV_VARS = frozenset(
    {
        "HOME",
        "PATH",
        "USER",
        "SHELL",
        "PWD",
        "LANG",
        "TERM",
        "TMPDIR",
        "LOGNAME",
        "HOSTNAME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "PS1",
        "OLDPWD",
        "EDITOR",
        "PAGER",
        "DISPLAY",
        "SSH_AUTH_SOCK",
        "XDG_RUNTIME_DIR",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    }
)


B = TypeVar("B", bound=type[BaseModel])


class EndpointConfig(ResourceConfig):
    """HTTP endpoint configuration with secure credential handling.

    Extends ResourceConfig with HTTP-specific settings: URL construction,
    authentication, headers, and request validation.

    Credential Security:
        - api_key accepts env var name (UPPERCASE_WITH_UNDERSCORES) or raw credential
        - Env var names preserved in api_key for serialization
        - Raw credentials cleared from api_key, stored only in _api_key (SecretStr)
        - _api_key never serialized (PrivateAttr)

    Attributes:
        base_url: API base URL (e.g., "https://api.openai.com/v1").
        endpoint: Path appended to base_url, supports {param} formatting.
        endpoint_params: Expected URL parameter names for validation.
        params: Param values for endpoint formatting.
        method: HTTP method (default "POST").
        content_type: Content-Type header (default "application/json").
        auth_type: Auth header style ("bearer", "x-api-key", "basic", "none").
        default_headers: Headers merged into every request.
        api_key: Env var name (preserved) or raw credential (cleared after resolve).
        api_key_is_env: True if api_key is env var reference (for deserialization).
        openai_compatible: Enable OpenAI response parsing.
        requires_tokens: Enable token estimation for rate limiting.
        client_kwargs: Extra kwargs passed to httpx.AsyncClient.
    """

    base_url: str | None = None
    endpoint: str
    endpoint_params: list[str] | None = None
    method: str = "POST"
    params: dict[str, str] = Field(default_factory=dict)
    content_type: str | None = "application/json"
    auth_type: AUTH_TYPES = "bearer"
    default_headers: dict = Field(default_factory=dict)
    api_key: str | None = Field(None, frozen=True)
    api_key_is_env: bool = Field(False, frozen=True)
    openai_compatible: bool = False
    requires_tokens: bool = False
    client_kwargs: dict = Field(default_factory=dict)
    _api_key: SecretStr | None = PrivateAttr(None)

    @property
    def api_key_env(self) -> str | None:
        """Env var name if api_key_is_env=True, else None."""
        return self.api_key

    @property
    def is_cli(self) -> bool:
        """Manual Override in subclass"""
        return False

    @model_validator(mode="after")
    def _validate_api_key_n_params(self):
        """Resolve api_key and validate endpoint params.

        API key resolution:
            1. If api_key_is_env=True (deserialization): verify env var exists
            2. If matches UPPERCASE_PATTERN and env var exists: mark as env var
            3. Otherwise: treat as raw credential, clear api_key field

        Raises:
            ValueError: If credential empty, system env var used, or invalid params.
        """
        if self.api_key is not None:
            if self.api_key_is_env:
                if not os.getenv(self.api_key):
                    raise ValueError(
                        f"Environment variable '{self.api_key}' not found during deserialization. "
                        f"Model was serialized with env var reference that no longer exists."
                    )
                resolved = os.getenv(self.api_key, None)
                if resolved and resolved.strip():
                    object.__setattr__(self, "_api_key", SecretStr(resolved.strip()))
                return self

            if not self.api_key.strip():
                raise ValueError("api_key cannot be empty or whitespace")

            is_env_var_pattern = bool(re.match(r"^[A-Z][A-Z0-9_]*$", self.api_key))

            if is_env_var_pattern:
                if self.api_key in SYSTEM_ENV_VARS:
                    raise ValueError(
                        f"'{self.api_key}' is a system environment variable and cannot be used as api_key. "
                        f"If this is a raw credential, pass it as SecretStr('{self.api_key}')."
                    )

                resolved = os.getenv(self.api_key, None)
                if resolved is not None:
                    if not resolved.strip():
                        raise ValueError(
                            f"Environment variable '{self.api_key}' is empty or whitespace"
                        )
                    object.__setattr__(self, "api_key_is_env", True)
                    object.__setattr__(self, "_api_key", SecretStr(resolved.strip()))
                else:
                    object.__setattr__(self, "api_key_is_env", False)
                    object.__setattr__(self, "_api_key", SecretStr(self.api_key.strip()))
                    object.__setattr__(self, "api_key", None)
            else:
                object.__setattr__(self, "api_key_is_env", False)
                object.__setattr__(self, "_api_key", SecretStr(self.api_key.strip()))
                object.__setattr__(self, "api_key", None)

        if self.endpoint_params and self.params:
            invalid_params = set(self.params.keys()) - set(self.endpoint_params)
            if invalid_params:
                raise ValueError(
                    f"Invalid params {invalid_params}. Must be subset of endpoint_params: {self.endpoint_params}"
                )
            missing_params = set(self.endpoint_params) - set(self.params.keys())
            if missing_params:
                logger.warning(
                    f"Endpoint expects params {missing_params} but they were not provided. "
                    f"URL formatting may fail."
                )
        return self

    @property
    def full_url(self) -> str:
        """Construct full URL: base_url/endpoint with params formatted."""
        if not self.endpoint_params:
            return f"{self.base_url}/{self.endpoint}"
        return f"{self.base_url}/{self.endpoint.format(**self.params)}"


class Endpoint(ResourceBackend):
    """HTTP API backend with resilience patterns.

    Wraps httpx.AsyncClient with circuit breaker and retry support.
    Handles request validation, header construction, and response normalization.

    Resilience Stack (outer to inner):
        retry_config -> circuit_breaker -> _call_http

    Attributes:
        config: EndpointConfig with URL, auth, and request options.
        circuit_breaker: Optional CircuitBreaker for fail-fast behavior.
        retry_config: Optional RetryConfig for exponential backoff.

    Example:
        endpoint = Endpoint(
            config={"provider": "openai", "name": "gpt-4", ...},
            circuit_breaker=CircuitBreaker(failure_threshold=5),
            retry_config=RetryConfig(max_retries=3),
        )
        response = await endpoint.call(request={"model": "gpt-4", ...})
    """

    circuit_breaker: CircuitBreaker | None = None
    retry_config: RetryConfig | None = None
    config: EndpointConfig

    def __init__(
        self,
        config: dict | EndpointConfig,
        circuit_breaker: CircuitBreaker | None = None,
        retry_config: RetryConfig | None = None,
        **kwargs,
    ):
        """Initialize Endpoint with config and optional resilience patterns.

        Args:
            config: EndpointConfig or dict with endpoint settings.
            circuit_breaker: Optional circuit breaker for fail-fast.
            retry_config: Optional retry configuration.
            **kwargs: Additional config overrides merged into config.

        Raises:
            ValueError: If config invalid or api_key empty.
        """
        secret_api_key = None
        if isinstance(config, dict):
            config_dict = {**config, **kwargs}
            if "api_key" in config_dict and isinstance(config_dict["api_key"], SecretStr):
                secret_api_key = config_dict.pop("api_key")
            _config = EndpointConfig(**config_dict)
        elif isinstance(config, EndpointConfig):
            _config = (
                config.model_copy(deep=True, update=kwargs)
                if kwargs
                else config.model_copy(deep=True)
            )
        else:
            raise ValueError("Config must be a dict or EndpointConfig instance")

        super().__init__(  # type: ignore[call-arg]
            config=_config,
            circuit_breaker=circuit_breaker,
            retry_config=retry_config,
        )

        if secret_api_key is not None:
            raw_value = secret_api_key.get_secret_value()
            if not raw_value.strip():
                raise ValueError("api_key cannot be empty or whitespace")
            object.__setattr__(self.config, "_api_key", SecretStr(raw_value.strip()))

        logger.debug(
            f"Initialized Endpoint: provider={self.config.provider}, "
            f"endpoint={self.config.endpoint}, cb={circuit_breaker is not None}, "
            f"retry={retry_config is not None}"
        )

    def _create_http_client(self):
        """Create httpx.AsyncClient with config timeout and client_kwargs."""
        import httpx

        return httpx.AsyncClient(
            timeout=self.config.timeout,
            **self.config.client_kwargs,
        )

    @property
    def event_type(self) -> type:
        """APICalling event type for this backend."""
        return APICalling

    @property
    def full_url(self) -> str:
        """Full URL from config (base_url/endpoint with params)."""
        return self.config.full_url

    def create_payload(
        self,
        request: dict | BaseModel,
        **kwargs,
    ) -> dict:
        """Build validated payload from request and config defaults.

        Merges: config.kwargs <- request <- kwargs, then validates
        against request_options schema.

        Args:
            request: Request parameters (dict or Pydantic model).
            **kwargs: Additional parameters merged last.

        Returns:
            Validated payload dict filtered to schema fields.

        Raises:
            ValueError: If request_options not defined or validation fails.
        """
        request = request if isinstance(request, dict) else request.model_dump(exclude_none=True)

        payload = self.config.kwargs.copy()
        payload.update(request)
        if kwargs:
            payload.update(kwargs)

        if self.config.request_options is None:
            raise ValueError(
                f"Endpoint {self.config.name} must define request_options schema. "
                "All endpoint backends must use proper request validation."
            )

        valid_fields = set(self.config.request_options.model_fields.keys())
        filtered_payload = {k: v for k, v in payload.items() if k in valid_fields}
        return self.config.validate_payload(filtered_payload)

    def create_headers(self, extra_headers: dict | None = None) -> dict:
        """Build request headers with auth and content type.

        Args:
            extra_headers: Additional headers merged last.

        Returns:
            Headers dict ready for HTTP request.
        """
        headers = HeaderFactory.get_header(
            auth_type=self.config.auth_type,
            content_type=self.config.content_type,
            api_key=self.config._api_key,
            default_headers=self.config.default_headers,
        )
        if extra_headers:
            headers.update(extra_headers)
        return headers

    async def call(
        self,
        request: dict | BaseModel,
        skip_payload_creation: bool = False,
        extra_headers: dict | None = None,
        **kwargs,
    ) -> Normalized:
        """Execute HTTP request with resilience patterns.

        Applies retry -> circuit_breaker -> _call stack.

        Args:
            request: Request parameters or Pydantic model.
            skip_payload_creation: Bypass create_payload validation.
            extra_headers: Additional headers merged with defaults.
            **kwargs: Extra httpx request kwargs.

        Returns:
            Normalized wrapping the API response.
        """
        if skip_payload_creation:
            payload = request if isinstance(request, dict) else request.model_dump()
        else:
            payload = self.create_payload(request, **kwargs)

        headers = self.create_headers(extra_headers)

        from collections.abc import Callable, Coroutine

        base_call = self._call
        inner_call: Callable[..., Coroutine[Any, Any, Any]]

        if self.circuit_breaker:

            async def cb_wrapped_call(p: dict[Any, Any], h: dict[Any, Any], **kw: Any) -> Any:
                return await self.circuit_breaker.execute(base_call, p, h, **kw)  # type: ignore[union-attr]

            inner_call = cb_wrapped_call
        else:
            inner_call = base_call

        if self.retry_config:
            raw_response = await retry_with_backoff(
                inner_call, payload, headers, **kwargs, **self.retry_config.as_kwargs()
            )
        else:
            raw_response = await inner_call(payload, headers, **kwargs)

        return self.normalize_response(raw_response)

    async def _call(self, payload: dict, headers: dict, **kwargs):
        """Execute HTTP request and return JSON response.

        Raises HTTPStatusError for 429 (rate limit) and 5xx (retryable).
        Other non-200 responses raise with error body details.
        """
        import httpx

        async with self._create_http_client() as client:
            response = await client.request(
                method=self.config.method,
                url=self.config.full_url,
                headers=headers,
                json=payload,
                **kwargs,
            )

            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            elif response.status_code != 200:
                try:
                    error_body = response.json()
                    error_message = (
                        f"Request failed with status {response.status_code}: {error_body}"
                    )
                except Exception:
                    error_message = f"Request failed with status {response.status_code}"

                raise httpx.HTTPStatusError(
                    message=error_message,
                    request=response.request,
                    response=response,
                )

            return response.json()

    async def stream(
        self,
        request: dict | BaseModel,
        extra_headers: dict | None = None,
        **kwargs,
    ):
        """Stream responses from endpoint.

        Args:
            request: Request parameters or Pydantic model.
            extra_headers: Additional headers merged with defaults.
            **kwargs: Extra httpx request kwargs.

        Yields:
            Response lines from streaming API.
        """
        payload, headers = self.create_payload(request, extra_headers, **kwargs)

        async for chunk in self._stream(payload=payload, headers=headers, **kwargs):
            yield chunk

    async def _stream(self, payload: dict, headers: dict, **kwargs):
        """Stream HTTP response lines (internal)."""
        import httpx

        payload["stream"] = True

        async with (
            self._create_http_client() as client,
            client.stream(
                method=self.config.method,
                url=self.config.full_url,
                headers=headers,
                json=payload,
                **kwargs,
            ) as response,
        ):
            if response.status_code != 200:
                raise httpx.HTTPStatusError(
                    message=f"Request failed with status {response.status_code}",
                    request=response.request,
                    response=response,
                )

            async for line in response.aiter_lines():
                if line:
                    yield line

    @field_serializer("circuit_breaker")
    def _serialize_circuit_breaker(
        self, circuit_breaker: CircuitBreaker | None
    ) -> dict[str, Any] | None:
        """Serialize CircuitBreaker to dict for transport."""
        if circuit_breaker is None:
            return None
        return circuit_breaker.to_dict()

    @field_serializer("retry_config")
    def _serialize_retry_config(self, retry_config: RetryConfig | None) -> dict[str, Any] | None:
        """Serialize RetryConfig to dict for transport."""
        if retry_config is None:
            return None
        return retry_config.to_dict()

    @field_validator("circuit_breaker", mode="before")
    @classmethod
    def _deserialize_circuit_breaker(cls, v: Any) -> CircuitBreaker | None:
        """Accept CircuitBreaker instance or dict."""
        if v is None:
            return None
        if isinstance(v, CircuitBreaker):
            return v
        if not isinstance(v, dict):
            raise ValueError("circuit_breaker must be a dict or CircuitBreaker instance")
        return CircuitBreaker(**v)

    @field_validator("retry_config", mode="before")
    @classmethod
    def _deserialize_retry_config(cls, v: Any) -> RetryConfig | None:
        """Accept RetryConfig instance or dict."""
        if v is None:
            return None
        if isinstance(v, RetryConfig):
            return v
        if not isinstance(v, dict):
            raise ValueError("retry_config must be a dict or RetryConfig instance")
        return RetryConfig(**v)


class APICalling(Calling):
    """HTTP API calling event for Endpoint backend.

    Wraps HTTP request with event lifecycle, token estimation for rate limiting,
    and extra headers support.

    Attributes:
        backend: Endpoint instance performing the HTTP call.
        extra_headers: Additional headers merged into request.
        payload: Request payload (inherited from Calling).

    Example:
        endpoint = Endpoint(config=config)
        calling = APICalling(
            backend=endpoint,
            payload={"model": "gpt-4", "messages": [...]},
            timeout=30.0,
        )
        await calling.invoke()
        response = calling.response  # Normalized
    """

    backend: Endpoint = Field(exclude=True)
    extra_headers: dict | None = Field(default=None, exclude=True)

    @property
    def required_tokens(self) -> int | None:
        """Estimated tokens for rate limiting (None disables tracking)."""
        if (
            hasattr(self.backend.config, "requires_tokens")
            and not self.backend.config.requires_tokens
        ):
            return None

        if "messages" in self.payload:
            return self._estimate_message_tokens(self.payload["messages"])
        if "input" in self.payload:
            return self._estimate_text_tokens(self.payload["input"])
        return None

    def _estimate_message_tokens(self, messages: list[dict]) -> int:
        """Rough token estimate for chat messages (~4 chars/token)."""
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return total_chars // 4

    def _estimate_text_tokens(self, text: str | list[str]) -> int:
        """Rough token estimate for text/embeddings (~4 chars/token)."""
        inputs = [text] if isinstance(text, str) else text
        total_chars = sum(len(t) for t in inputs)
        return total_chars // 4

    @property
    def request(self) -> dict:
        """Permission request data for rate limiting checks."""
        return {
            "required_tokens": self.required_tokens,
        }

    @property
    def call_args(self) -> dict:
        """Arguments for backend.call(**call_args)."""
        args = {
            "request": self.payload,
            "skip_payload_creation": True,
        }
        if self.extra_headers:
            args["extra_headers"] = self.extra_headers
        return args
