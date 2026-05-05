# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from typing_extensions import Self

from lionagi.ln.types._sentinel import MaybeUnset, Unset

from .role import Role, RoledContent

if TYPE_CHECKING:
    from lionagi.beta.resource.backend import Normalized

__all__ = (
    "Assistant",
    "parse_to_assistant_message",
)


@dataclass(slots=True)
class Assistant(RoledContent):
    """Assistant text response."""

    role: ClassVar[Role] = Role.ASSISTANT
    response: MaybeUnset[Any] = Unset

    _buffered_response: Any = Unset

    @classmethod
    def create(cls, response_object: Normalized) -> Self:
        self = cls(response=response_object.data)
        self._buffered_response = response_object
        return self

    @property
    def raw_response(self) -> dict[str, Any] | None:
        if self._is_sentinel(self._buffered_response):
            return None
        buffered = self._buffered_response
        serialized = getattr(buffered, "serialized", None)
        return serialized

    def render(self, *_args, **_kwargs) -> str:
        return str(self.response) if not self.is_sentinel_field("response") else ""


def parse_to_assistant_message(response: Normalized) -> Any:
    """Construct a session Message from a Normalized backend response.

    Returns a Message (session layer) — imported lazily to avoid circular
    imports while session/message is being migrated.
    """
    from lionagi.protocols.messages import Message

    metadata_dict: dict[str, Any] = {"raw_response": response.serialized}
    if response.metadata is not None:
        metadata_dict.update(response.metadata)

    return Message(
        content=Assistant.create(response_object=response),
        metadata=metadata_dict,
    )
