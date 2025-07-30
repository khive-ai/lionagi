from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal, ParamSpec, TypeVar

from typing_extensions import TypedDict

from lionagi.protocols.types import Event

E = TypeVar("E", bound=Event)
P = ParamSpec("P")
SC = TypeVar("SC")  # streaming chunk type


class EventHooks(TypedDict, total=False):
    pre_event_create: Callable | None
    pre_invokation: Callable | None


HookEventTypes = Literal["pre_event_create", "pre_invokation"]

StreamHandlers = dict[str, Callable[[SC], Awaitable[None]]]


class EndpointEventHook:
    """Event-driven hook system for lionagi endpoint lifecycle management.

    Provides hooks for event creation, invocation, and streaming chunk processing.
    Hook failures are handled gracefully - they can either exit or continue with
    CANCELLED/FAILED status for observability.

    Example:
        ```python
        async def validate_request(event_type, **kwargs):
            if not kwargs.get('api_key'):
                raise ValueError("API key required")

        async def log_chunks(chunk):
            logger.info(f"Received chunk: {chunk}")

        hooks = EventHooks(pre_event_create=validate_request)
        stream_handlers = {"text": log_chunks}
        hook_system = EndpointEventHook(hooks=hooks, stream_handlers=stream_handlers)
        ```
    """

    def __init__(
        self,
        hooks: EventHooks | None = None,
        stream_handlers: StreamHandlers | None = None,
    ):
        self._stream_handlers = {}
        self._hooks = {}
        if hooks:
            self._hooks.update(hooks)
        if stream_handlers:
            self._stream_handlers.update(stream_handlers)

    def can_handle(
        self, method: HookEventTypes = None, chunk_type: str | type = None
    ) -> bool:
        if method is None and chunk_type is None:
            raise ValueError("Either method or chunk_type must be provided")
        if method:
            if method not in EventHooks.__optional_keys__:
                raise ValueError(
                    f"Invalid method: {method}. Must be one of {EventHooks.__optional_keys__}"
                )
            return self._hooks.get(method) is not None
        return self._stream_handlers.get(chunk_type) is not None

    async def call(
        self, method: str, *args, exit: bool, **kwargs
    ) -> tuple[Any | Exception, bool]:
        """Execute a hook with configurable error handling.

        If the hook raises an exception:
        - if exit is True, system will exit after this hook
        - if exit is False, system will proceed to create an event and mark status as "cancelled"
          (for logging purposes) and continue to the next hook or event creation.

        Returns:
            tuple[Any | Exception, bool]: (result_or_exception, should_exit)
        """
        try:
            if (hook := self._hooks.get(method)) is not None:
                result = await hook(*args, **kwargs)
                return (result, False)
        except Exception as e:
            return (e, exit)

    async def pre_event_create(
        self, event_type: type[E], exit: bool = False, **kwargs
    ) -> tuple[E | Exception | None, bool]:
        """Hook to be called before an event is created.

        Typically used to modify or validate the event creation parameters.

        The hook function takes an event type and any additional keyword arguments.
        It can:
            - return an instance of the event type
            - return None if no event should be created during handling, event will be created in corresponding default manner
            - raise an exception if this event should be cancelled (status: cancelled, reason: f"pre-event-create hook aborted this event: {e}")
        """
        return await self.call(
            "pre_event_create", event_type, exit=exit, **kwargs
        )

    async def pre_invokation(
        self, event: E, exit: bool = False
    ) -> tuple[None | Exception, bool]:
        """Hook to be called when an event is dequeued and right before it is invoked.

        Typically used to check permissions.

        The hook function takes the content of the event as a dictionary.
        It can either raise an exception to abort the event invokation or pass to continue (status: cancelled).
        It cannot modify the event itself, and won't be able to access the event instance.
        """
        return await self.call("pre_invokation", event.to_dict(), exit=exit)

    async def handle_streaming_chunk(
        self, chunk_type: str | type, chunk: Any, exit: bool = False
    ) -> tuple[None | Exception, bool]:
        """Hook to be called to consume streaming chunks.

        Typically used for logging or stream event abortion.

        The handler function signature should be: `async def handler(chunk: Any) -> None`

        It can either raise an exception to mark the event invokation as "failed" or pass to continue (status: aborted).
        """
        try:
            if (handler := self._stream_handlers.get(chunk_type)) is not None:
                result = await handler(chunk)
                return (result, False)
        except Exception as e:
            return (e, exit)
