# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import logging
from abc import abstractmethod
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from lionagi.beta.core.base.element import Element
from lionagi.beta.core.base.event import Event, EventStatus
from lionagi._errors import ValidationError
from lionagi.ln.types._sentinel import Unset, UnsetType, is_sentinel, is_unset
from lionagi.ln.types import DataClass, HashableModel, ModelConfig

from lionagi.service.hooks import HookEvent, HookEventTypes, HookRegistry

logger = logging.getLogger(__name__)


_SCHEMA_FIELD_KEYS_CACHE: dict[type[BaseModel], set[str]] = {}


def _get_schema_field_keys(cls: type[BaseModel]) -> set[str]:
    """Return cached field names via model_fields (includes SkipJsonSchema fields)."""
    if cls not in _SCHEMA_FIELD_KEYS_CACHE:
        _SCHEMA_FIELD_KEYS_CACHE[cls] = set(cls.model_fields.keys())
    return _SCHEMA_FIELD_KEYS_CACHE[cls]


class ResourceConfig(HashableModel):
    provider: str = Field(..., min_length=1, max_length=50)
    name: str = Field(..., min_length=1, max_length=100)
    request_options: type[BaseModel] | None = Field(default=None, exclude=True)
    timeout: int = Field(default=300, ge=1, le=3600)
    max_retries: int = Field(default=3, ge=0, le=10)
    version: str | None = None
    tags: list[str] = Field(default_factory=list)
    kwargs: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _validate_kwargs(cls, data: dict[str, Any]) -> dict[str, Any]:
        kwargs = data.pop("kwargs", {})
        field_keys = _get_schema_field_keys(cls)
        for k in list(data.keys()):
            if k not in field_keys:
                kwargs[k] = data.pop(k)
        data["kwargs"] = kwargs
        return data

    @field_validator("request_options", mode="before")
    def _validate_request_options(cls, v):
        if v is None:
            return None
        if isinstance(v, type) and issubclass(v, BaseModel):
            return v
        if isinstance(v, BaseModel):
            return v.__class__
        raise ValueError("request_options must be a Pydantic model type")

    def validate_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.request_options:
            return data
        try:
            self.request_options.model_validate(data)
            return data
        except Exception as e:
            raise ValueError("Invalid payload") from e


@dataclass(slots=True)
class Normalized(DataClass):
    """Normalized backend response: data is the direct output, serialized is the raw dict form."""

    _config: ClassVar[ModelConfig] = ModelConfig(
        sentinel_additions=frozenset({"none", "empty", "pydantic", "dataclass"}),
        use_enum_values=True,
    )
    status: Literal["success", "error"]
    data: Any
    error: str | None = None
    serialized: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class Calling(Event):
    """Base event wrapping a backend.call() with pre/post invocation hooks."""

    backend: ResourceBackend = Field(..., exclude=True, description="Resource backend instance")
    payload: dict[str, Any] = Field(..., description="Request payload/arguments")
    _pre_invoke_hook_event: HookEvent | None = PrivateAttr(None)
    _post_invoke_hook_event: HookEvent | None = PrivateAttr(None)
    _stream_chunk_hook: Callable[..., Any] | None = PrivateAttr(None)

    @property
    def response(self) -> Normalized | UnsetType:
        if is_sentinel(self.execution.response):
            return Unset
        resp = self.execution.response
        if isinstance(resp, Normalized):
            return resp
        return Unset

    @property
    @abstractmethod
    def call_args(self) -> dict:
        """Keyword arguments for backend.call(**self.call_args); must be implemented by subclass."""
        ...

    # Keys in call_args that must not propagate to backend.stream().
    _STREAM_EXCLUDE_KEYS: frozenset[str] = frozenset({"skip_payload_creation"})

    @property
    def stream_args(self) -> dict:
        """call_args minus _STREAM_EXCLUDE_KEYS; override for divergent streaming signatures."""
        args = dict(self.call_args)
        for key in self._STREAM_EXCLUDE_KEYS:
            args.pop(key, None)
        return args

    async def _invoke(self) -> Normalized:
        await self._check_pre_invoke_hook()

        try:
            response = await self.backend.call(**self.call_args)
            return response
        finally:
            await self._check_post_invoke_hook()

    async def _check_pre_invoke_hook(self) -> None:
        if h_ev := self._pre_invoke_hook_event:
            await h_ev.invoke()

            if h_ev.execution.status in (EventStatus.FAILED, EventStatus.CANCELLED):
                raise RuntimeError(
                    f"Pre-invoke hook {h_ev.execution.status.value}: {h_ev.execution.error}"
                )

            if h_ev._should_exit:
                raise h_ev._exit_cause or RuntimeError(
                    "Pre-invocation hook requested exit without a cause"
                )
            logger.debug("hook.pre: %s", h_ev.execution.status)

    async def _check_post_invoke_hook(self) -> None:
        if h_ev := self._post_invoke_hook_event:
            await h_ev.invoke()

            if h_ev.execution.status in (EventStatus.FAILED, EventStatus.CANCELLED):
                logger.warning(
                    f"Post-invoke hook {h_ev.execution.status.value}: {h_ev.execution.error}"
                )

            if h_ev._should_exit:
                raise h_ev._exit_cause or RuntimeError(
                    "Post-invocation hook requested exit without a cause"
                )
            logger.debug("hook.post: %s", h_ev.execution.status)

    async def _stream(self) -> AsyncGenerator[Normalized, Any]:
        await self._check_pre_invoke_hook()

        try:
            async for chunk in self.backend.stream(**self.stream_args):
                normalized = self.backend.normalize_chunk(chunk)
                if self._stream_chunk_hook is not None:
                    result = self._stream_chunk_hook(normalized)
                    if inspect.isawaitable(result):
                        result = await result
                    if isinstance(result, Normalized):
                        normalized = result
                yield normalized
        finally:
            await self._check_post_invoke_hook()

    def create_pre_invoke_hook(
        self,
        hook_registry: HookRegistry,
        exit_hook: bool | None = None,
        hook_timeout: float = 30.0,
        hook_params: dict[str, Any] | None = None,
    ) -> None:
        h_ev = HookEvent(
            hook_type=HookEventTypes.PreInvocation,
            event_like=self,
            registry=hook_registry,
            exit=exit_hook if exit_hook is not None else False,
            timeout=hook_timeout,
            streaming=False,
            params=hook_params or {},
        )
        self._pre_invoke_hook_event = h_ev

    def create_post_invoke_hook(
        self,
        hook_registry: HookRegistry,
        exit_hook: bool | None = None,
        hook_timeout: float = 30.0,
        hook_params: dict[str, Any] | None = None,
    ) -> None:
        h_ev = HookEvent(
            hook_type=HookEventTypes.PostInvocation,
            event_like=self,
            registry=hook_registry,
            exit=exit_hook if exit_hook is not None else False,
            timeout=hook_timeout,
            streaming=False,
            params=hook_params or {},
        )
        self._post_invoke_hook_event = h_ev

    def assert_is_normalized(self) -> None:
        self.assert_completed()
        if is_unset(self.execution.response):
            raise ValidationError("Calling response is not set")
        if not isinstance(self.execution.response, Normalized):
            raise ValidationError("Calling response is not normalized")


class ResourceBackend(Element):
    """Base class for resource backends; subclasses implement event_type and call()."""

    config: ResourceConfig = Field(..., description="Resource configuration")

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def version(self) -> str | None:
        return self.config.version

    @property
    def tags(self) -> set[str]:
        return set(self.config.tags) if self.config.tags else set()

    @property
    def request_options(self) -> type[BaseModel] | None:
        return self.config.request_options

    @property
    @abstractmethod
    def event_type(self) -> type[Calling]:
        """The Calling subclass used to wrap invocations of this backend."""
        ...

    def normalize_response(self, raw_response: Any) -> Normalized:
        return Normalized(
            status="success",
            data=raw_response,
            serialized=raw_response,
        )

    def normalize_chunk(self, raw_response: Any) -> Normalized:
        return Normalized(
            status="success",
            data=raw_response,
            serialized=raw_response,
        )

    @abstractmethod
    async def call(self, *args, **kw) -> Normalized: ...

    async def stream(self, *args, **kw):
        raise NotImplementedError("This backend does not support streaming calls.")
