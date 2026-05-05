# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""HTTP endpoint backend with credential resolution, circuit breaker, and retry support."""

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

    api_key accepts an env var name (UPPERCASE pattern) or raw credential; raw
    credentials are cleared from the field and stored only in the private _api_key
    SecretStr to prevent serialization leaks.
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
        return self.api_key

    @property
    def is_cli(self) -> bool:
        """Override in subclass to return True for CLI-backed endpoints."""
        return False

    @model_validator(mode="after")
    def _validate_api_key_n_params(self):
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
        if not self.endpoint_params:
            return f"{self.base_url}/{self.endpoint}"
        return f"{self.base_url}/{self.endpoint.format(**self.params)}"


class Endpoint(ResourceBackend):
    """HTTP API backend wrapping httpx with circuit breaker and retry (retry -> cb -> _call)."""

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
        import httpx

        return httpx.AsyncClient(
            timeout=self.config.timeout,
            **self.config.client_kwargs,
        )

    @property
    def event_type(self) -> type:
        return APICalling

    @property
    def full_url(self) -> str:
        return self.config.full_url

    def create_payload(
        self,
        request: dict | BaseModel,
        **kwargs,
    ) -> dict:
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
        """Execute HTTP request; raises HTTPStatusError for 429 and 5xx (retryable by caller)."""
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
        payload, headers = self.create_payload(request, extra_headers, **kwargs)

        async for chunk in self._stream(payload=payload, headers=headers, **kwargs):
            yield chunk

    async def _stream(self, payload: dict, headers: dict, **kwargs):
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
        if circuit_breaker is None:
            return None
        return circuit_breaker.to_dict()

    @field_serializer("retry_config")
    def _serialize_retry_config(self, retry_config: RetryConfig | None) -> dict[str, Any] | None:
        if retry_config is None:
            return None
        return retry_config.to_dict()

    @field_validator("circuit_breaker", mode="before")
    @classmethod
    def _deserialize_circuit_breaker(cls, v: Any) -> CircuitBreaker | None:
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
        if v is None:
            return None
        if isinstance(v, RetryConfig):
            return v
        if not isinstance(v, dict):
            raise ValueError("retry_config must be a dict or RetryConfig instance")
        return RetryConfig(**v)


class APICalling(Calling):
    """Calling event for Endpoint with token estimation and extra header support."""

    backend: Endpoint = Field(exclude=True)
    extra_headers: dict | None = Field(default=None, exclude=True)

    @property
    def required_tokens(self) -> int | None:
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
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return total_chars // 4

    def _estimate_text_tokens(self, text: str | list[str]) -> int:
        inputs = [text] if isinstance(text, str) else text
        total_chars = sum(len(t) for t in inputs)
        return total_chars // 4

    @property
    def request(self) -> dict:
        return {
            "required_tokens": self.required_tokens,
        }

    @property
    def call_args(self) -> dict:
        args = {
            "request": self.payload,
            "skip_payload_creation": True,
        }
        if self.extra_headers:
            args["extra_headers"] = self.extra_headers
        return args
