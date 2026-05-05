# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import Field, PrivateAttr, field_validator
from typing_extensions import TypedDict

from lionagi.beta.core.base.event import Event, EventStatus
from lionagi._errors import ConfigurationError, OperationError
from lionagi.ln.types import Enum
from lionagi.ln.types._sentinel import Undefined
from lionagi.ln import concurrency

SC = TypeVar("SC")
StreamHandlers = dict[str | type, Callable[[SC], Awaitable[None]]]
E = TypeVar("E", bound=Event)


class HookPhase(Enum):
    """Lifecycle phases at which hooks may be registered."""

    PreEventCreate = "pre_event_create"
    PreInvocation = "pre_invocation"
    PostInvocation = "post_invocation"
    ErrorHandling = "error_handling"


class AssociatedEventInfo(TypedDict, total=False):
    """Information about the event associated with the hook."""

    kron_class: str
    """Full qualified name of the event class."""

    event_id: str
    """ID of the event."""

    event_created_at: str
    """Creation timestamp of the event (ISO format string)."""


class HookEvent(Event):
    """Event that delegates _invoke() to a HookRegistry phase handler."""

    registry: HookRegistry = Field(..., exclude=True)
    hook_phase: HookPhase
    exit: bool = Field(False, exclude=True)
    params: dict[str, Any] = Field(default_factory=dict, exclude=True)
    event_like: Event | type[Event] = Field(..., exclude=True)
    _should_exit: bool = PrivateAttr(False)
    _exit_cause: BaseException | None = PrivateAttr(None)

    associated_event_info: AssociatedEventInfo | None = None

    @field_validator("exit", mode="before")
    def _validate_exit(cls, v: Any) -> bool:
        if v is None:
            return False
        return v

    async def _invoke(self) -> Any:
        result = await self.registry.call(
            self.event_like,
            hook_phase=self.hook_phase,
            exit=self.exit,
            **self.params,
        )

        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            inner_tuple, meta = result
            res, se, _ = inner_tuple
        else:
            res, se, _ = result
            meta = {}

        event_info: AssociatedEventInfo = {"kron_class": str(meta.get("kron_class", ""))}
        if "event_id" in meta:
            event_info["event_id"] = str(meta["event_id"])
        if "event_created_at" in meta:
            event_info["event_created_at"] = str(meta["event_created_at"])
        self.associated_event_info = event_info
        self._should_exit = se

        if isinstance(res, tuple) and len(res) == 2:
            self._exit_cause = res[1]
            raise res[1]

        if isinstance(res, Exception):
            self._exit_cause = res
            raise res

        return res


K = TypeVar("K")


def get_handler(
    d_: dict[K, Any], k: K, get: bool = False, /
) -> Callable[..., Awaitable[Any]] | None:
    """Look up a handler, wrapping sync callables; returns passthrough default when get=True and key missing."""
    handler = d_.get(k)
    if handler is None and not get:
        return None

    if handler is not None:
        if not concurrency.is_coro_func(handler):

            async def _wrapper(*args: Any, **kwargs: Any) -> Any:
                await concurrency.sleep(0)
                return handler(*args, **kwargs)

            return _wrapper
        return handler

    async def _default_handler(*args: Any, **_kwargs: Any) -> Any:
        await concurrency.sleep(0)
        return args[0] if args else None

    return _default_handler


def validate_hooks(kw: dict[Any, Any]) -> None:
    if not isinstance(kw, dict):
        raise ValueError("Hooks must be a dictionary of callable functions")

    for k, v in kw.items():
        if not isinstance(k, HookPhase) or k not in HookPhase.allowed():
            raise ValueError(f"Hook key must be one of {HookPhase.allowed()}, got {k}")
        if not callable(v):
            raise ValueError(f"Hook for {k} must be callable, got {type(v)}")


def validate_stream_handlers(kw: dict[Any, Any]) -> None:
    if not isinstance(kw, dict):
        raise ValueError("Stream handlers must be a dictionary of callable functions")

    for k, v in kw.items():
        if not isinstance(k, str | type):
            raise ValueError(f"Stream handler key must be a string or type, got {type(k)}")

        if not callable(v):
            raise ValueError(f"Stream handler for {k} must be callable, got {type(v)}")


class HookRegistry:
    """Registry of phase hooks and stream handlers for the Event lifecycle."""

    _hooks: dict[HookPhase, Callable[..., Any]]
    _stream_handlers: dict[str | type, Callable[..., Any]]

    def __init__(
        self,
        hooks: dict[HookPhase, Callable[..., Any]] | None = None,
        stream_handlers: StreamHandlers[Any] | None = None,
    ):
        _hooks: dict[HookPhase, Callable[..., Any]] = {}
        _stream_handlers: dict[str | type, Callable[..., Any]] = {}

        if hooks is not None:
            validate_hooks(hooks)
            _hooks.update(hooks)

        if stream_handlers is not None:
            validate_stream_handlers(stream_handlers)
            _stream_handlers.update(stream_handlers)

        self._hooks = _hooks
        self._stream_handlers = _stream_handlers

    async def _call(
        self,
        hp_: HookPhase | None,
        ct_: str | type | None,
        ch_: Any,
        ev_: E | type[E],
        /,
        **kw: Any,
    ) -> tuple[Any | Exception, bool]:
        if hp_ is None and ct_ is None:
            raise ConfigurationError("Either hook_type or chunk_type must be provided")
        if hp_ and (self._hooks.get(hp_)):
            validate_hooks({hp_: self._hooks[hp_]})
            h = get_handler(self._hooks, hp_, True)
            if h is not None:
                return await h(ev_, **kw)
            raise OperationError(f"No handler found for hook phase: {hp_}")
        elif not ct_:
            raise ConfigurationError("Hook type is required when chunk_type is not provided")
        else:
            validate_stream_handlers({ct_: self._stream_handlers.get(ct_)})
            h = get_handler(self._stream_handlers, ct_, True)
            if h is not None:
                return await h(ev_, ct_, ch_, **kw)
            raise OperationError(f"No handler found for chunk type: {ct_}")

    async def _call_stream_handler(
        self,
        ct_: str | type,
        ch_: Any,
        ev_: Any,
        /,
        **kw: Any,
    ) -> Any:
        validate_stream_handlers({ct_: self._stream_handlers.get(ct_)})
        handler = get_handler(self._stream_handlers, ct_, True)
        if handler is not None:
            return await handler(ev_, ct_, ch_, **kw)
        raise OperationError(f"No stream handler found for chunk type: {ct_}")

    async def pre_event_create(
        self,
        event_type: type[E],
        /,
        exit: bool = False,  # noqa: A002
        **kw: Any,
    ) -> tuple[Any, bool, EventStatus]:
        try:
            res = await self._call(
                HookPhase.PreEventCreate,
                None,
                None,
                event_type,
                exit=exit,
                **kw,
            )
            return (res, False, EventStatus.COMPLETED)
        except concurrency.get_cancelled_exc_class() as e:
            return ((Undefined, e), True, EventStatus.CANCELLED)
        except Exception as e:
            return (e, exit, EventStatus.CANCELLED)

    async def pre_invocation(
        self,
        event: E,
        /,
        exit: bool = False,  # noqa: A002
        **kw: Any,
    ) -> tuple[Any, bool, EventStatus]:
        try:
            res = await self._call(
                HookPhase.PreInvocation,
                None,
                None,
                event,
                exit=exit,
                **kw,
            )
            return (res, False, EventStatus.COMPLETED)
        except concurrency.get_cancelled_exc_class() as e:
            return ((Undefined, e), True, EventStatus.CANCELLED)
        except Exception as e:
            return (e, exit, EventStatus.CANCELLED)

    async def post_invocation(
        self,
        event: E,
        /,
        exit: bool = False,  # noqa: A002
        **kw: Any,
    ) -> tuple[Any, bool, EventStatus]:
        try:
            res = await self._call(
                HookPhase.PostInvocation,
                None,
                None,
                event,
                exit=exit,
                **kw,
            )
            return (res, False, EventStatus.COMPLETED)
        except concurrency.get_cancelled_exc_class() as e:
            return ((Undefined, e), True, EventStatus.CANCELLED)
        except Exception as e:
            return (e, exit, EventStatus.ABORTED)

    async def handle_streaming_chunk(
        self,
        chunk_type: str | type | None,
        chunk: Any,
        /,
        exit: bool = False,  # noqa: A002
        **kw: Any,
    ) -> tuple[Any, bool, EventStatus | None]:
        if chunk_type is None:
            raise ValueError("chunk_type cannot be None for streaming chunks")
        try:
            res = await self._call_stream_handler(
                chunk_type,
                chunk,
                None,
                exit=exit,
                **kw,
            )
            return (res, False, None)
        except concurrency.get_cancelled_exc_class() as e:
            return ((Undefined, e), True, EventStatus.CANCELLED)
        except Exception as e:
            return (e, exit, EventStatus.ABORTED)

    async def call(
        self,
        event_like: Event | type[Event],
        /,
        *,
        hook_phase: HookPhase | None = None,
        chunk_type: str | type | None = None,
        chunk: Any = None,
        exit: bool = False,  # noqa: A002
        **kw: Any,
    ) -> (
        tuple[tuple[Any, bool, EventStatus], dict[str, Any]] | tuple[Any, bool, EventStatus | None]
    ):
        if hook_phase is None and chunk_type is None:
            raise ValueError("Either method or chunk_type must be provided")

        if hook_phase:
            meta: dict[str, Any] = {"kron_class": event_like.class_name(full=True)}
            match hook_phase:
                case HookPhase.PreEventCreate | HookPhase.PreEventCreate.value:
                    if isinstance(event_like, type):
                        return (
                            await self.pre_event_create(event_like, exit=exit, **kw),
                            meta,
                        )
                    return (
                        await self.pre_event_create(type(event_like), exit=exit, **kw),
                        meta,
                    )
                case HookPhase.PreInvocation | HookPhase.PreInvocation.value:
                    if isinstance(event_like, Event):
                        meta["event_id"] = str(event_like.id)
                        meta["event_created_at"] = event_like.created_at.isoformat()
                        return (
                            await self.pre_invocation(event_like, exit=exit, **kw),
                            meta,
                        )
                    raise TypeError("PreInvocation requires an Event instance, not a type")
                case HookPhase.PostInvocation | HookPhase.PostInvocation.value:
                    if isinstance(event_like, Event):
                        meta["event_id"] = str(event_like.id)
                        meta["event_created_at"] = event_like.created_at.isoformat()
                        return (
                            await self.post_invocation(event_like, exit=exit, **kw),
                            meta,
                        )
                    raise TypeError("PostInvocation requires an Event instance, not a type")
        return await self.handle_streaming_chunk(chunk_type, chunk, exit=exit, **kw)

    def _can_handle(
        self,
        /,
        *,
        hp_: HookPhase | None = None,
        ct_=None,
    ) -> bool:
        if hp_:
            return hp_ in self._hooks
        if ct_:
            return ct_ in self._stream_handlers
        return False
