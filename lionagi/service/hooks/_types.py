from typing import Awaitable, Callable, Optional, TypeVar

from typing_extensions import TypedDict

from lionagi.utils import StringEnum

SC = TypeVar("SC")  # streaming chunk type

__all__ = (
    "HookEventTypes",
    "ALLOWED_HOOKS_TYPES",
    "HookDict",
    "StreamHandlers",
    "AssosiatedEventInfo",
)


class HookEventTypes(StringEnum):
    PreEventCreate = "pre_event_create"
    PreInvokation = "pre_invokation"
    PostInvokation = "post_invokation"


ALLOWED_HOOKS_TYPES = HookEventTypes.allowed()


class HookDict(TypedDict):
    pre_event_create: Optional[Callable]
    pre_invokation: Optional[Callable]
    post_invokation: Optional[Callable]


StreamHandlers = dict[str, Callable[[SC], Awaitable[None]]]


class AssosiatedEventInfo(TypedDict, total=False):
    """Information about the event associated with the hook."""

    lion_class: str
    """Full qualified name of the event class."""

    event_id: str
    """ID of the event."""

    event_created_at: float
    """Creation timestamp of the event."""
