# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from typing import Any

from pydantic import Field, PrivateAttr, field_serializer, field_validator

from lionagi._errors import ConfigurationError, ExecutionError
from lionagi.service.processor import Executor
from lionagi.ln.concurrency import sleep
from lionagi.protocols.contracts import Invocable, implements
from lionagi.protocols.types import Element
from lionagi.service.hooks import HookRegistry
from lionagi.service.rate_limiter import RateLimitConfig, TokenBucket

from .backend import Calling, Normalized, ResourceBackend, ResourceConfig
from .utilities.rate_limited_executor import RateLimitedExecutor

__all__ = ("iModel",)


class _ProviderEndpointBackend(ResourceBackend):
    """ResourceBackend adapter for production provider endpoints."""

    _endpoint: Any = PrivateAttr()

    def __init__(self, endpoint: Any, *, name: str | None = None) -> None:
        cfg = endpoint.config
        super().__init__(
            config=ResourceConfig(
                provider=cfg.provider,
                name=name or cfg.name,
                request_options=cfg.request_options,
                timeout=cfg.timeout,
                max_retries=cfg.max_retries,
                kwargs={"endpoint": cfg.endpoint},
            )
        )
        self._endpoint = endpoint

    @property
    def endpoint(self) -> Any:
        return self._endpoint

    @property
    def event_type(self) -> type[Calling]:
        return _ProviderEndpointCalling

    def create_payload(self, request: dict | None = None, **kwargs: Any) -> dict:
        payload = dict(request or {})
        payload.update(kwargs)
        return payload

    async def call(
        self,
        request: dict,
        skip_payload_creation: bool = False,
        **kwargs: Any,
    ) -> Normalized:
        raw = await self._endpoint.call(
            request=request,
            skip_payload_creation=skip_payload_creation,
            **kwargs,
        )
        return self.normalize_response(raw)

    async def stream(self, request: dict, **kwargs: Any):
        async for chunk in self._endpoint.stream(request=request, **kwargs):
            yield self.normalize_chunk(chunk)


class _ProviderEndpointCalling(Calling):
    backend: _ProviderEndpointBackend = Field(exclude=True)

    @property
    def call_args(self) -> dict:
        return {
            "request": self.payload,
            "skip_payload_creation": False,
        }


@implements(Invocable)
class iModel(Element):  # noqa: N801
    """Unified resource interface combining a ResourceBackend with rate limiting and hooks."""

    _EXECUTOR_POLL_TIMEOUT_ITERATIONS = 100
    _EXECUTOR_POLL_SLEEP_INTERVAL = 0.1

    backend: ResourceBackend | None = Field(
        None,
        description="ResourceBackend instance (e.g., Endpoint)",
    )

    rate_limiter: TokenBucket | None = Field(
        None,
        description="Optional TokenBucket rate limiter (simple blocking)",
    )

    executor: Executor | None = Field(
        None,
        description="Optional Executor for event-driven processing with rate limiting",
    )

    hook_registry: HookRegistry | None = Field(
        None,
        description="Optional HookRegistry for invocation lifecycle hooks",
    )

    provider_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific metadata (e.g., Claude Code session_id for context continuation)",
    )

    def __init__(
        self,
        backend: ResourceBackend,
        rate_limiter: TokenBucket | None = None,
        executor: Executor | None = None,
        hook_registry: HookRegistry | None = None,
        queue_capacity: int = 100,
        capacity_refresh_time: float = 60,
        limit_requests: int | None = None,
    ):
        if executor is None and limit_requests:
            request_bucket = TokenBucket(
                RateLimitConfig(
                    capacity=limit_requests,
                    refill_rate=limit_requests / capacity_refresh_time,
                )
            )

            executor = RateLimitedExecutor(
                processor_config={
                    "queue_capacity": queue_capacity,
                    "capacity_refresh_time": capacity_refresh_time,
                    "request_bucket": request_bucket,
                }
            )

        super().__init__(
            backend=backend,
            rate_limiter=rate_limiter,
            executor=executor,
            hook_registry=hook_registry,
        )

    @classmethod
    def from_provider(
        cls,
        spec: str,
        *,
        api_key: str | None = None,
        name: str | None = None,
        yolo: bool = False,
        **kwargs: Any,
    ) -> iModel:
        """Create iModel from a model spec string with automatic effort parsing.

        Supports::

            iModel.from_provider("claude_code/opus-4-7-high")
            # → provider=claude_code, model=claude-opus-4-7, effort=high

            iModel.from_provider("codex/gpt-5.4-mini-xhigh")
            iModel.from_provider("claude")  # alias → claude_code/sonnet
            iModel.from_provider("openai/gpt-4o", api_key="OPENAI_API_KEY")

        Args:
            spec: Model spec. Formats: "provider/model-effort", "provider/model",
                or alias ("claude", "codex", "gemini").
            api_key: API key or env var name.
            name: Resource name for registration (defaults to model).
            yolo: Auto-approve/permissive mode (provider-specific kwargs).
            **kwargs: Additional kwargs passed to endpoint constructor.

        Returns:
            Configured iModel ready for invocation.
        """
        from lionagi.cli._providers import (
            _CLAUDE_PROVIDER_NAMES,
            _CODEX_EFFORT_CLAMP,
            PROVIDER_EFFORT_KWARG,
            PROVIDER_YOLO_KWARGS,
            _clamp_claude_effort,
            parse_model_spec,
        )
        from lionagi.service.connections import match_endpoint

        ms = parse_model_spec(spec)
        if "/" in ms.model:
            provider, model = ms.model.split("/", 1)
        else:
            provider = ms.model
            model = ms.model

        extra = {**kwargs}
        if ms.effort is not None:
            effort = ms.effort
            if provider == "codex":
                effort = _CODEX_EFFORT_CLAMP.get(effort, effort)
            elif provider in _CLAUDE_PROVIDER_NAMES:
                effort = _clamp_claude_effort(effort, model)
            if kwarg := PROVIDER_EFFORT_KWARG.get(provider):
                extra[kwarg] = effort
        if yolo:
            extra.update(PROVIDER_YOLO_KWARGS.get(provider, {}))

        resource_name = name or model

        endpoint_map = {
            "openai": "chat/completions",
            "anthropic": "messages",
            "groq": "chat/completions",
            "openrouter": "chat/completions",
            "nvidia_nim": "chat/completions",
        }
        endpoint_path = endpoint_map.get(provider, "chat/completions")

        endpoint = match_endpoint(
            provider=provider,
            endpoint=endpoint_path,
            name=resource_name,
            model=model,
            api_key=api_key,
            **extra,
        )

        return cls(backend=_ProviderEndpointBackend(endpoint, name=resource_name))

    @property
    def name(self) -> str:
        if self.backend is None:
            raise ConfigurationError("Backend not configured")
        return self.backend.name

    @property
    def version(self) -> str:
        if self.backend is None:
            raise ConfigurationError("Backend not configured")
        return self.backend.version or ""

    @property
    def tags(self) -> set[str]:
        if self.backend is None:
            raise ConfigurationError("Backend not configured")
        return self.backend.tags

    async def create_calling(
        self,
        timeout: float | None = None,
        streaming: bool = False,
        stream_chunk_hook: Any | None = None,
        create_event_exit_hook: bool | None = None,
        create_event_hook_timeout: float = 10.0,
        create_event_hook_params: dict | None = None,
        pre_invoke_exit_hook: bool | None = None,
        pre_invoke_hook_timeout: float = 30.0,
        pre_invoke_hook_params: dict | None = None,
        post_invoke_exit_hook: bool | None = None,
        post_invoke_hook_timeout: float = 30.0,
        post_invoke_hook_params: dict | None = None,
        **arguments: Any,
    ) -> Calling:
        from lionagi.service.hooks import HookEvent, HookEventTypes

        if self.backend is None:
            raise ConfigurationError("Backend not configured")

        calling_type = self.backend.event_type

        if self.hook_registry is not None and self.hook_registry._can_handle(
            ht_=HookEventTypes.PreEventCreate
        ):
            h_ev = HookEvent(
                hook_type=HookEventTypes.PreEventCreate,
                event_like=calling_type,
                registry=self.hook_registry,
                exit=(
                    create_event_exit_hook
                    if create_event_exit_hook is not None
                    else False
                ),
                timeout=create_event_hook_timeout,
                streaming=False,
                params=create_event_hook_params or {},
            )
            await h_ev.invoke()

            if h_ev._should_exit:
                raise h_ev._exit_cause or RuntimeError(
                    "PreEventCreate hook requested exit without a cause"
                )

        payload = self.backend.create_payload(request=arguments)
        if isinstance(payload, tuple):
            payload, _ = payload
        calling: Calling = calling_type(
            backend=self.backend,
            payload=payload,
        )

        if self.hook_registry is not None and self.hook_registry._can_handle(
            ht_=HookEventTypes.PreInvocation
        ):
            calling.create_pre_invoke_hook(
                hook_registry=self.hook_registry,
                exit_hook=(
                    pre_invoke_exit_hook if pre_invoke_exit_hook is not None else False
                ),
                hook_timeout=pre_invoke_hook_timeout,
                hook_params=pre_invoke_hook_params or {},
            )

        if self.hook_registry is not None and self.hook_registry._can_handle(
            ht_=HookEventTypes.PostInvocation
        ):
            calling.create_post_invoke_hook(
                hook_registry=self.hook_registry,
                exit_hook=(
                    post_invoke_exit_hook
                    if post_invoke_exit_hook is not None
                    else False
                ),
                hook_timeout=post_invoke_hook_timeout,
                hook_params=post_invoke_hook_params or {},
            )

        if timeout is not None:
            calling.timeout = timeout
        if streaming:
            calling.streaming = streaming
        if stream_chunk_hook is not None:
            calling._stream_chunk_hook = stream_chunk_hook

        return calling

    async def invoke(
        self,
        calling: Calling | None = None,
        poll_timeout: float | None = None,
        poll_interval: float | None = None,
        **arguments: Any,
    ) -> Calling:
        """Invoke via executor (rate-limited) or direct path; pre-created calling ignores **arguments."""
        if calling is None:
            calling = await self.create_calling(**arguments)

        if self.executor:
            if self.executor.processor is None or self.executor.processor.is_stopped():
                await self.executor.start()

            await self.executor.append(calling)
            await self.executor.forward()

            # Poll for completion (fast backends see ~100-200% overhead, slow backends <10%)
            interval = poll_interval or self._EXECUTOR_POLL_SLEEP_INTERVAL
            timeout_seconds = poll_timeout or (
                self._EXECUTOR_POLL_TIMEOUT_ITERATIONS
                * self._EXECUTOR_POLL_SLEEP_INTERVAL
            )
            max_iterations = int(timeout_seconds / interval)
            ctr = 0
            while calling.execution.status.value in ["pending", "processing"]:
                if ctr > max_iterations:
                    raise TimeoutError(
                        f"Event processing timeout after {timeout_seconds:.1f}s: {calling.id}"
                    )
                await self.executor.forward()
                ctr += 1
                await sleep(interval)

            if calling.execution.status.value == "aborted":
                raise ExecutionError(
                    "Event aborted after 3 permission denials (rate limited)",
                    details={"event_id": str(calling.id)},
                    retryable=False,
                )
            elif calling.execution.status.value == "failed":
                raise calling.execution.error or ExecutionError(
                    "Event failed",
                    details={"event_id": str(calling.id)},
                )

            self._store_claude_code_session_id(calling)
            return calling

        else:
            if self.rate_limiter:
                acquired = await self.rate_limiter.acquire(timeout=30.0)
                if not acquired:
                    raise TimeoutError("Rate limit acquisition timeout (30s)")

            await calling.invoke()
            self._store_claude_code_session_id(calling)
            return calling

    @contextlib.asynccontextmanager
    async def stream(
        self,
        calling: Calling | None = None,
        stream_chunk_hook: Any | None = None,
        **arguments: Any,
    ):
        if calling is None:
            calling = await self.create_calling(
                streaming=True,
                stream_chunk_hook=stream_chunk_hook,
                **arguments,
            )
        elif stream_chunk_hook is not None:
            calling._stream_chunk_hook = stream_chunk_hook

        if self.rate_limiter:
            acquired = await self.rate_limiter.acquire(timeout=30.0)
            if not acquired:
                raise TimeoutError("Rate limit acquisition timeout (30s)")

        async with calling.stream() as chunks:
            yield chunks

        self._store_claude_code_session_id(calling)

    def _store_claude_code_session_id(self, calling: Calling) -> None:
        """Extract and store Claude Code session_id for context continuation."""
        from lionagi.ln.types._sentinel import is_sentinel

        if self.backend is None or self.backend.config.provider != "claude_code":
            return
        if is_sentinel(calling.execution.response):
            return

        if isinstance(self.backend, _ProviderEndpointBackend):
            endpoint_session_id = getattr(
                self.backend.endpoint,
                "provider_session_id",
                None,
            )
            if endpoint_session_id:
                self.provider_metadata["session_id"] = endpoint_session_id
                return

        response = calling.execution.response
        if isinstance(response, Normalized) and response.metadata:
            session_id = response.metadata.get("session_id")
            if session_id:
                self.provider_metadata["session_id"] = session_id

    @field_serializer("backend")
    def _serialize_backend(self, backend: ResourceBackend) -> dict[str, Any] | None:
        if backend is None:
            return None
        backend_dict = backend.model_dump()
        if "metadata" not in backend_dict:
            backend_dict["metadata"] = {}
        backend_dict["metadata"]["lion_class"] = backend.__class__.class_name(full=True)
        return backend_dict

    @field_serializer("rate_limiter")
    def _serialize_rate_limiter(self, v: TokenBucket | None) -> dict[str, Any] | None:
        if v is None:
            return None
        return v.to_dict()

    @field_serializer("executor")
    def _serialize_executor(self, executor: Executor | None) -> dict[str, Any] | None:
        if executor is None:
            return None

        if isinstance(executor, RateLimitedExecutor):
            config = {**executor.processor_config}
            if "request_bucket" in config and config["request_bucket"] is not None:
                bucket = config["request_bucket"]
                if isinstance(bucket, TokenBucket):
                    config["request_bucket"] = bucket.to_dict()
            return config

        return None

    @field_validator("rate_limiter", mode="before")
    @classmethod
    def _deserialize_rate_limiter(cls, v: Any) -> TokenBucket | None:
        if v is None:
            return None

        if isinstance(v, TokenBucket):
            return v

        if not isinstance(v, dict):
            raise ValueError("rate_limiter must be a dict or TokenBucket instance")

        config = RateLimitConfig(**v)
        return TokenBucket(config)

    @field_validator("backend", mode="before")
    @classmethod
    def _deserialize_backend(cls, v: Any) -> ResourceBackend:
        if v is None:
            raise ValueError("backend is required")

        if isinstance(v, ResourceBackend):
            return v

        if not isinstance(v, dict):
            raise ValueError("backend must be a dict or ResourceBackend instance")

        from lionagi.protocols.generic.element import Element

        backend = Element.from_dict(v)

        if not isinstance(backend, ResourceBackend):
            raise ValueError(
                f"Deserialized backend must be ResourceBackend subclass, got: {type(backend).__name__}"
            )
        return backend

    @field_validator("executor", mode="before")
    @classmethod
    def _deserialize_executor(cls, v: Any) -> Executor | None:
        if v is None:
            return None

        if isinstance(v, Executor):
            return v

        if not isinstance(v, dict):
            raise ValueError("executor must be a dict or Executor instance")

        config = {**v}
        if "request_bucket" in config and isinstance(config["request_bucket"], dict):
            config["request_bucket"] = TokenBucket(
                RateLimitConfig(**config["request_bucket"])
            )
        if "token_bucket" in config and isinstance(config["token_bucket"], dict):
            config["token_bucket"] = TokenBucket(
                RateLimitConfig(**config["token_bucket"])
            )

        return RateLimitedExecutor(processor_config=config)

    def _to_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Exclude id/created_at so reconstruction creates a fresh identity."""
        exclude = set(kwargs.pop("exclude", set()))
        exclude.update({"id", "created_at"})
        kwargs["exclude"] = exclude
        return super()._to_dict(**kwargs)

    def __repr__(self) -> str:
        if self.backend is None:
            return "iModel(backend=None)"
        return f"iModel(backend={self.backend.name}, version={self.backend.version})"

    async def __aenter__(self) -> iModel:
        if self.executor is not None and (
            self.executor.processor is None or self.executor.processor.is_stopped()
        ):
            await self.executor.start()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> bool:
        if (
            self.executor is not None
            and self.executor.processor is not None
            and not self.executor.processor.is_stopped()
        ):
            await self.executor.stop()
        return False
