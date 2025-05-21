# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, JsonValue

from lion2.core.collections.pile import Pile, Progression
from lion2.core.types import MessageRole, ToolRequest, ToolResponse
from .action_request_response import (
    ActionRequestContent,
    ActionResponseContent,
)
from .assistant_response import AssistantResponseContent
from .instruction import InstructionContent
from .message import (
    Message,
)
from .system import SystemContent

DEFAULT_SYSTEM = "You are a helpful AI assistant. Let's think step by step."


class MessageManager:
    def __init__(
        self,
        messages: list[Message] | None = None,
        progression: Progression | None = None,
    ):
        self.messages: Pile[Message] = Pile(
            collections=messages,
            progression=progression,
            item_type=Message,
            strict_type=True,
        )
        if (
            len(self.messages) > 0
            and self.messages[0].role == MessageRole.SYSTEM
        ):
            self.system = self.messages[0]
        else:
            self.system = None

    async def set_system(self, system: Message) -> None:
        """
        Replace or set the system message. If one existed, remove it.
        """
        if not self.system:
            self.system = system
            await self.messages.insert(0, self.system)
        else:
            old_system = self.system
            self.system = system
            await self.messages.insert(0, self.system)
            await self.messages.exclude(old_system)

    @staticmethod
    def create_message(
        *,
        sender: UUID | None = None,
        recipient: UUID | None = None,
        instruction: JsonValue = None,
        context: JsonValue = None,
        guidance: JsonValue = None,
        request_fields: JsonValue = None,
        plain_content: JsonValue = None,
        response_format: BaseModel | type[BaseModel] = None,
        images: list = None,
        image_detail: Literal["low", "high", "auto"] = None,
        tool_schemas: dict = None,
        system: Any = None,
        system_datetime: bool | str = None,
        model_response: Any = None,
        assistant_name: str = None,
        created_at: str | datetime | int | None = None,
        tool_requests: ToolRequest | list[ToolRequest] = None,
        tool_responses: ToolResponse | list[ToolResponse] = None,
        action_request_id: UUID | None = None,
    ):
        if (
            sum(
                bool(x)
                for x in (
                    instruction,
                    system,
                    model_response,
                    tool_requests,
                    tool_responses,
                )
            )
            > 1
        ):
            raise ValueError("Only one message type can be added at a time.")

        if tool_responses is not None and action_request_id is None:
            raise ValueError(
                "action_request_id must be provided when tool_responses are given."
            )

        message_content = None
        if system:
            message_content = SystemContent(
                system=system, system_datetime=system_datetime
            )
        if instruction:
            message_content = InstructionContent(
                instruction=instruction,
                context=context,
                guidance=guidance,
                request_fields=request_fields,
                plain_content=plain_content,
                response_format=response_format,
                images=images,
                image_detail=image_detail,
                tool_schemas=tool_schemas,
            )
        if model_response:
            message_content = AssistantResponseContent(
                model_response=model_response,
                name=assistant_name,
                created_at=created_at,
            )
        if tool_responses:
            message_content = ActionResponseContent(
                action_request_id=action_request_id,
                responses=tool_responses,
            )
        if tool_requests:
            message_content = ActionRequestContent(
                requests=tool_requests,
            )
        return Message(
            message_content=message_content,
            sender=sender,
            recipient=recipient,
        )

    @staticmethod
    def update_message(
        *,
        message: Message,
        sender: UUID | None = None,
        recipient: UUID | None = None,
        instruction: JsonValue = None,
        context: JsonValue = None,
        guidance: JsonValue = None,
        request_fields: JsonValue = None,
        plain_content: JsonValue = None,
        response_format: BaseModel | type[BaseModel] = None,
        images: list = None,
        image_detail: Literal["low", "high", "auto"] = None,
        tool_schemas: dict = None,
        system: Any = None,
        system_datetime: bool | str = None,
        model_response: Any = None,
        assistant_name: str = None,
        created_at: str | datetime | int | None = None,
        tool_requests: ToolRequest | list[ToolRequest] = None,
        tool_responses: ToolResponse | list[ToolResponse] = None,
        action_request_id: UUID | None = None,
        action_response_id: UUID | None = None,
        append_request_responses: bool = False,
    ):
        if isinstance(message.message_content, SystemContent):
            message.message_content.update(
                system=system, system_datetime=system_datetime
            )
        if isinstance(message.message_content, InstructionContent):
            message.message_content.update(
                instruction=instruction,
                context=context,
                guidance=guidance,
                request_fields=request_fields,
                plain_content=plain_content,
                response_format=response_format,
                images=images,
                image_detail=image_detail,
                tool_schemas=tool_schemas,
            )
        if isinstance(message.message_content, AssistantResponseContent):
            message.message_content.update(
                model_response=model_response,
                name=assistant_name,
                created_at=created_at,
            )
        if isinstance(message.message_content, ActionResponseContent):
            message.message_content.update(
                responses=tool_responses,
                append=append_request_responses,
            )
            if action_request_id:
                message.message_content.action_request_id = action_request_id
        if isinstance(message.message_content, ActionRequestContent):
            message.message_content.update(
                requests=tool_requests,
                append=append_request_responses,
            )
            if action_response_id:
                message.message_content.action_response_id = action_response_id
        if sender:
            message.sender = sender
        if recipient:
            message.recipient = recipient
        return message

    async def add_message(
        self,
        *,
        message: Message | None = None,
        sender: UUID | None = None,
        recipient: UUID | None = None,
        instruction: JsonValue = None,
        context: JsonValue = None,
        guidance: JsonValue = None,
        request_fields: JsonValue = None,
        plain_content: JsonValue = None,
        response_format: BaseModel | type[BaseModel] = None,
        images: list = None,
        image_detail: Literal["low", "high", "auto"] = None,
        tool_schemas: dict = None,
        system: Any = None,
        system_datetime: bool | str = None,
        model_response: Any = None,
        assistant_name: str = None,
        created_at: str | datetime | int | None = None,
        tool_requests: ToolRequest | list[ToolRequest] = None,
        tool_responses: ToolResponse | list[ToolResponse] = None,
        action_request_id: UUID | None = None,
        action_response_id: UUID | None = None,
        append_request_responses: bool = False,
    ):
        params = {
            k: v
            for k, v in locals().items()
            if v is not None
            and k
            not in (
                "self",
                "message",
                "action_request_id",
                "append_request_responses",
            )
        }

        if message is None:
            message = self.create_message(**params)
            await self.messages.add(message)
        else:
            message = self.update_message(
                message=message,
                action_request_id=action_request_id,
                append_request_responses=append_request_responses,
                **params,
            )
            await self.messages.update(message)

        if message.role == MessageRole.SYSTEM:
            self.set_system(message)
        return message

    async def clear_messages(self):
        """Remove all messages except the system message if it exists."""
        await self.messages.clear()
        if self.system:
            await self.messages.insert(0, self.system)

    @property
    def last_response(self) -> Message | None:
        """
        Retrieve the most recent `AssistantResponse`.
        """
        for mid in reversed(self.messages.progression):
            if isinstance(
                self.messages[mid].message_content, AssistantResponseContent
            ):
                return self.messages[mid]
        return None

    @property
    def last_instruction(self) -> Message | None:
        """
        Retrieve the most recent `Instruction`.
        """
        for mid in reversed(self.messages.progression):
            if isinstance(
                self.messages[mid].message_content, InstructionContent
            ):
                return self.messages[mid]
        return None

    @property
    def assistant_responses(self) -> Pile[Message]:
        """All `AssistantResponse` messages in the manager."""
        return Pile(
            collections=[
                self.messages[mid]
                for mid in self.messages.progression
                if isinstance(
                    self.messages[mid].message_content,
                    AssistantResponseContent,
                )
            ],
            item_type=Message,
            strict_type=True,
        )

    @property
    def actions(self) -> Pile[Message]:
        """All action messages in the manager."""
        return Pile(
            collections=[
                self.messages[mid]
                for mid in self.messages.progression
                if self.messages[mid].is_action
            ],
            item_type=Message,
            strict_type=True,
        )

    @property
    def action_requests(self) -> Pile[Message]:
        """All `ActionRequest` messages in the manager."""
        return Pile(
            collections=[
                self.messages[mid]
                for mid in self.messages.progression
                if isinstance(
                    self.messages[mid].message_content, ActionRequestContent
                )
            ],
            item_type=Message,
            strict_type=True,
        )

    @property
    def action_responses(self) -> Pile[Message]:
        """All `ActionResponse` messages in the manager."""
        return Pile(
            collections=[
                self.messages[mid]
                for mid in self.messages.progression
                if isinstance(
                    self.messages[mid].message_content, ActionResponseContent
                )
            ],
            item_type=Message,
            strict_type=True,
        )

    @property
    def instructions(self) -> Pile[Message]:
        """All `Instruction` messages in the manager."""
        return Pile(
            collections=[
                self.messages[mid]
                for mid in self.messages.progression
                if isinstance(
                    self.messages[mid].message_content, InstructionContent
                )
            ],
            item_type=Message,
            strict_type=True,
        )

    def remove_last_instruction_tool_schemas(self) -> None:
        """
        Convenience method to strip 'tool_schemas' from the most recent Instruction.
        """
        if self.last_instruction:
            self.messages[
                self.last_instruction.id
            ].message_content.tool_schemas = None

    def concat_recent_action_responses_to_instruction(
        self, instruction: Message
    ) -> None:
        """
        Example method to merge the content of recent ActionResponses
        into an instruction's context.
        """
        for i in reversed(self.messages.progression):
            if isinstance(
                self.messages[i].message_content, ActionResponseContent
            ):
                instruction.message_content.context.append(
                    self.messages[i].message_content.rendered
                )
            else:
                break

    def to_chat_msgs(self, progression=None) -> list[dict]:
        """
        Convert a subset (or all) of messages into a chat representation array.

        Args:
            progression (Optional[Sequence]): A subset of message IDs or the full progression.

        Returns:
            list[dict]: Each item is a dict with 'role' and 'content'.
        """
        if progression == []:
            return []
        try:
            return [
                self.messages[mid].message_content.chat_msg
                for mid in (progression or self.messages.progression)
            ]
        except Exception as e:
            raise ValueError(
                "One or more messages in the requested progression are invalid."
            ) from e

    def __bool__(self):
        return len(self.messages) > 0

    def __contains__(self, message: Message) -> bool:
        return message in self.messages
