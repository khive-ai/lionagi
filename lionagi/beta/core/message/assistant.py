# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lionagi.protocols.messages.assistant_response import AssistantResponseContent

if TYPE_CHECKING:
    from lionagi.beta.resource.backend import Normalized

__all__ = (
    "Assistant",
    "parse_to_assistant_message",
)

# Production type re-exported under the beta name.
Assistant = AssistantResponseContent


def parse_to_assistant_message(response: Normalized) -> Any:
    """Construct a session Message from a Normalized backend response.

    Returns a Message (session layer) — imported lazily to avoid circular
    imports while session/message is being migrated.
    """
    from lionagi.protocols.messages import Message

    metadata_dict: dict[str, Any] = {"raw_response": response.serialized}
    if response.metadata is not None:
        metadata_dict.update(response.metadata)

    content = AssistantResponseContent.create(response_object=response)
    return Message(
        content=content,
        metadata=metadata_dict,
    )
