# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TypeVar

from lionagi.libs.concurrency import get_cancelled_exc_class
from lionagi.protocols.types import Event, EventStatus
from lionagi.utils import UNDEFINED

from ._types import HookDict, HookEventTypes, StreamHandlers
from ._utils import validate_hooks, validate_stream_handlers

E = TypeVar("E", bound=Event)


class HookRegistry:
    def __init__(
        self,
        hooks: HookDict = None,
        stream_handlers: StreamHandlers = None,
    ):
        _hooks = {}
        _stream_handlers = {}

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
        ht_: HookEventTypes,
        ct_: str | type,
        ch_: Any,
        ev_: E | type[E],
        /,
        **kw,
    ) -> tuple[Any | Exception, bool]:
        if ht_ is None and ct_ is None:
            raise RuntimeError(
                "Either hook_type or chunk_type must be provided"
            )
        if ht_ and (h := self._hooks.get(ht_)):
            validate_hooks({ht_: h})
            return await h(ev_, **kw)
        elif not ct_:
            raise RuntimeError(
                "Hook type is required when chunk_type is not provided"
            )
        else:
            validate_stream_handlers(
                {ct_: (h := self._stream_handlers.get(ct_))}
            )
            return await h(ev_, ct_, ch_, **kw)

    async def _call_stream_handler(
        self,
        ct_: str | type,
        ch_: Any,
        ev_,
        /,
        **kw,
    ):
        handler = self._stream_handlers.get(ct_)
        validate_stream_handlers({ct_: handler})
        return await handler(ev_, ct_, ch_, **kw)

    async def pre_event_create(
        self, event_type: type[E], /, exit: bool = False, **kw
    ) -> tuple[E | Exception | None, bool, EventStatus]:
        """Hook to be called before an event is created.

        Typically used to modify or validate the event creation parameters.

        The hook function takes an event type and any additional keyword arguments.
        It can:
            - return an instance of the event type
            - return None if no event should be created during handling, event will be
                created in corresponding default manner
            - raise an exception if this event should be cancelled
                (status: cancelled, reason: f"pre-event-create hook aborted this event: {e}")
        """
        try:
            res = await self._call(
                HookEventTypes.PreEventCreate,
                None,
                None,
                event_type,
                **kw,
            )
            return (res, False, EventStatus.COMPLETED)
        except get_cancelled_exc_class() as e:
            return ((UNDEFINED, e), True, EventStatus.CANCELLED)
        except Exception as e:
            return (e, exit, EventStatus.CANCELLED)

    async def pre_invokation(
        self, event: E, /, exit: bool = False, **kw
    ) -> tuple[Any, bool, EventStatus]:
        """Hook to be called when an event is dequeued and right before it is invoked.

        Typically used to check permissions.

        The hook function takes the content of the event as a dictionary.
        It can either raise an exception to abort the event invokation or pass to continue (status: cancelled).
        It cannot modify the event itself, and won't be able to access the event instance.
        """
        try:
            res = await self._call(
                HookEventTypes.PreInvokation,
                None,
                None,
                event,
                **kw,
            )
            return (res, False, EventStatus.COMPLETED)
        except get_cancelled_exc_class() as e:
            return ((UNDEFINED, e), True, EventStatus.CANCELLED)
        except Exception as e:
            return (e, exit, EventStatus.CANCELLED)

    async def post_invokation(
        self, event: E, /, exit: bool = False, **kw
    ) -> tuple[None | Exception, bool, EventStatus, EventStatus]:
        """Hook to be called right after event finished its execution.
        It can either raise an exception to abort the event invokation or pass to continue (status: aborted).
        It cannot modify the event itself, and won't be able to access the event instance.
        """
        try:
            res = await self._call(
                HookEventTypes.PostInvokation,
                None,
                None,
                event,
                **kw,
            )
            return (res, False, EventStatus.COMPLETED)
        except get_cancelled_exc_class() as e:
            return ((UNDEFINED, e), True, EventStatus.CANCELLED)
        except Exception as e:
            return (e, exit, EventStatus.ABORTED)

    async def handle_streaming_chunk(
        self, chunk_type: str | type, chunk: Any, /, exit: bool = False, **kw
    ) -> tuple[Any, bool, EventStatus | None]:
        """Hook to be called to consume streaming chunks.

        Typically used for logging or stream event abortion.

        The handler function signature should be: `async def handler(chunk: Any) -> None`
        It can either raise an exception to mark the event invokation as "failed" or pass to continue (status: aborted).
        """
        try:
            res = await self._call_stream_handler(
                chunk_type,
                chunk,
                None,
                **kw,
            )
            return (res, False, None)
        except get_cancelled_exc_class() as e:
            return ((UNDEFINED, e), True, EventStatus.CANCELLED)
        except Exception as e:
            return (e, exit, EventStatus.ABORTED)

    async def call(
        self,
        event_like: Event | type[Event],
        /,
        *,
        hook_type: HookEventTypes = None,
        chunk_type=None,
        chunk=None,
        exit=False,
        **kw,
    ):
        """Call a hook or stream handler.

        If method is provided, it will call the corresponding hook.
        If chunk_type is provided, it will call the corresponding stream handler.
        If both are provided, method will be used.
        """
        if hook_type is None and chunk_type is None:
            raise ValueError("Either method or chunk_type must be provided")
        if hook_type:
            meta = {}
            meta["event_type"] = event_like.class_name(full=True)
            match hook_type:
                case HookEventTypes.PreEventCreate:
                    return await self.pre_event_create(event_like, **kw), meta
                case HookEventTypes.PreInvokation:
                    meta["event_id"] = str(event_like.id)
                    meta["event_created_at"] = event_like.created_at
                    return await self.pre_invokation(event_like, **kw), meta
                case HookEventTypes.PostInvokation:
                    meta["event_id"] = str(event_like.id)
                    meta["event_created_at"] = event_like.created_at
                    return await self.post_invokation(**kw), meta
        return await self.handle_streaming_chunk(chunk_type, chunk, exit, **kw)

    def _can_handle(
        self,
        /,
        *,
        ht_: HookEventTypes = None,
        ct_=None,
    ) -> bool:
        """Check if the registry can handle the given event or chunk type."""
        if ht_:
            return ht_ in self._hooks
        if ct_:
            return ct_ in self._stream_handlers
        return False
