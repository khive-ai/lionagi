# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from typing_extensions import TypedDict

from lionagi.utils import Enum

SC = TypeVar("SC")  # streaming chunk type

__all__ = (
    "HookEventTypes",
    "ALLOWED_HOOKS_TYPES",
    "HookDict",
    "StreamHandlers",
    "AssosiatedEventInfo",
)


class HookEventTypes(str, Enum):
    PreEventCreate = "pre_event_create"
    PreInvokation = "pre_invokation"
    PostInvokation = "post_invokation"


ALLOWED_HOOKS_TYPES = HookEventTypes.allowed()


class HookDict(TypedDict):
    pre_event_create: Callable | None
    pre_invokation: Callable | None
    post_invokation: Callable | None


StreamHandlers = dict[str, Callable[[SC], Awaitable[None]]]


class AssosiatedEventInfo(TypedDict, total=False):
    """Information about the event associated with the hook."""

    lion_class: str
    """Full qualified name of the event class."""

    event_id: str
    """ID of the event."""

    event_created_at: float
    """Creation timestamp of the event."""
