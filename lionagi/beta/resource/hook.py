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
    """Event lifecycle phases for hook registration.

    Hooks execute at specific points in Event lifecycle:
    - PreEventCreate: Before Event instantiation (receives event type)
    - PreInvocation: After Event created, before invoke() (receives event instance)
    - PostInvocation: After invoke() completes (receives event with result)
    - ErrorHandling: On exception during invocation
    """

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
    """Hook execution event that delegates to HookRegistry.

    Extends Event with hook-specific execution logic.
    Parent Event.invoke() handles lifecycle, this implements _invoke().
    """

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
        """Execute hook via registry (called by parent Event.invoke()).

        Parent Event.invoke() handles status/timing/errors automatically.
        Just execute hook logic and let exceptions propagate naturally.
        """
        result = await self.registry.call(
            self.event_like,
            hook_phase=self.hook_phase,
            exit=self.exit,
            **self.params,
        )

        # Unpack the result - hook_phase returns tuple of (inner_tuple, meta)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            inner_tuple, meta = result
            res, se, _ = inner_tuple
        else:
            # Streaming chunk returns a simpler tuple
            res, se, _ = result
            meta = {}

        # Build associated event info from meta dict
        event_info: AssociatedEventInfo = {"kron_class": str(meta.get("kron_class", ""))}
        if "event_id" in meta:
            event_info["event_id"] = str(meta["event_id"])
        if "event_created_at" in meta:
            event_info["event_created_at"] = str(meta["event_created_at"])
        self.associated_event_info = event_info
        self._should_exit = se

        # Handle error results - raise them so parent Event catches and sets FAILED status
        if isinstance(res, tuple) and len(res) == 2:
            # Tuple (Undefined, exception) from cancelled hook
            self._exit_cause = res[1]
            raise res[1]

        if isinstance(res, Exception):
            # Exception result from failed hook
            self._exit_cause = res
            raise res

        # Success - return result (parent sets COMPLETED status)
        return res


K = TypeVar("K")


def get_handler(
    d_: dict[K, Any], k: K, get: bool = False, /
) -> Callable[..., Awaitable[Any]] | None:
    """Retrieve async handler from dict, wrapping sync functions if needed.

    Args:
        d_: Handler dictionary (HookPhase->handler or chunk_type->handler).
        k: Key to look up.
        get: If True, return default passthrough handler when key missing.

    Returns:
        Async handler function, or None if key missing and get=False.
    """
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
    """Validate hook dict: keys must be HookPhase, values must be callable.

    Raises:
        ValueError: If dict structure or types are invalid.
    """
    if not isinstance(kw, dict):
        raise ValueError("Hooks must be a dictionary of callable functions")

    for k, v in kw.items():
        if not isinstance(k, HookPhase) or k not in HookPhase.allowed():
            raise ValueError(f"Hook key must be one of {HookPhase.allowed()}, got {k}")
        if not callable(v):
            raise ValueError(f"Hook for {k} must be callable, got {type(v)}")


def validate_stream_handlers(kw: dict[Any, Any]) -> None:
    """Validate stream handler dict: keys must be str|type, values callable.

    Raises:
        ValueError: If dict structure or types are invalid.
    """
    if not isinstance(kw, dict):
        raise ValueError("Stream handlers must be a dictionary of callable functions")

    for k, v in kw.items():
        if not isinstance(k, str | type):
            raise ValueError(f"Stream handler key must be a string or type, got {type(k)}")

        if not callable(v):
            raise ValueError(f"Stream handler for {k} must be callable, got {type(v)}")


class HookRegistry:
    """Registry for hook callbacks at Event lifecycle phases.

    Manages two handler types:
    - Phase hooks: Execute at PreEventCreate/PreInvocation/PostInvocation/ErrorHandling
    - Stream handlers: Process chunks during streaming (keyed by type name or class)

    Handler semantics:
    - Return value: Passed through to caller
    - Raise exception: Cancels/aborts operation (status depends on phase)
    - Exit flag: Determines whether exception should halt further processing
    """

    _hooks: dict[HookPhase, Callable[..., Any]]
    _stream_handlers: dict[str | type, Callable[..., Any]]

    def __init__(
        self,
        hooks: dict[HookPhase, Callable[..., Any]] | None = None,
        stream_handlers: StreamHandlers[Any] | None = None,
    ):
        """Initialize registry with optional hooks and stream handlers.

        Args:
            hooks: Mapping of HookPhase to handler callables.
            stream_handlers: Mapping of chunk type (str|type) to handler callables.
        """
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
        """Internal dispatch to hook or stream handler."""
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
        """Internal dispatch to stream handler by chunk type."""
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
        """Execute PreEventCreate hook before Event instantiation.

        Args:
            event_type: Event class being created.
            exit: If True and hook raises, signal caller to halt.
            **kw: Passed to hook handler.

        Returns:
            Tuple of (result|exception, should_exit, status).
        """
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
        """Execute PreInvocation hook before Event.invoke().

        Args:
            event: Event instance about to be invoked.
            exit: If True and hook raises, signal caller to halt.
            **kw: Passed to hook handler.

        Returns:
            Tuple of (result|exception, should_exit, status).
        """
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
        """Execute PostInvocation hook after Event.invoke() completes.

        Args:
            event: Event instance with execution results populated.
            exit: If True and hook raises, signal caller to halt.
            **kw: Passed to hook handler.

        Returns:
            Tuple of (result|exception, should_exit, status). Status is ABORTED on error.
        """
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
        """Process a streaming chunk via registered handler.

        Args:
            chunk_type: Type identifier for handler lookup (str name or class).
            chunk: The chunk data to process.
            exit: If True and handler raises, signal caller to halt.
            **kw: Passed to handler.

        Returns:
            Tuple of (result|exception, should_exit, status|None).

        Raises:
            ValueError: If chunk_type is None.
        """
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
        """Call a hook or stream handler.

        If method is provided, it will call the corresponding hook.
        If chunk_type is provided, it will call the corresponding stream handler.
        If both are provided, method will be used.
        """
        if hook_phase is None and chunk_type is None:
            raise ValueError("Either method or chunk_type must be provided")

        if hook_phase:
            meta: dict[str, Any] = {"kron_class": event_like.class_name(full=True)}
            match hook_phase:
                case HookPhase.PreEventCreate | HookPhase.PreEventCreate.value:
                    # For pre_event_create, event_like should be a type
                    if isinstance(event_like, type):
                        return (
                            await self.pre_event_create(event_like, exit=exit, **kw),
                            meta,
                        )
                    # Fall through to treat as event instance
                    return (
                        await self.pre_event_create(type(event_like), exit=exit, **kw),
                        meta,
                    )
                case HookPhase.PreInvocation | HookPhase.PreInvocation.value:
                    # For pre_invocation, event_like should be an instance
                    if isinstance(event_like, Event):
                        meta["event_id"] = str(event_like.id)
                        meta["event_created_at"] = event_like.created_at.isoformat()
                        return (
                            await self.pre_invocation(event_like, exit=exit, **kw),
                            meta,
                        )
                    raise TypeError("PreInvocation requires an Event instance, not a type")
                case HookPhase.PostInvocation | HookPhase.PostInvocation.value:
                    # For post_invocation, event_like should be an instance
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
        """Check if the registry can handle the given event or chunk type."""
        if hp_:
            return hp_ in self._hooks
        if ct_:
            return ct_ in self._stream_handlers
        return False
