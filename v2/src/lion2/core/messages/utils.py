import lionfuncs as ln

from .action_request_response import (
    ActionRequestContent,
    ActionResponseContent,
)
from .assistant_response import AssistantResponseContent
from .instruction import InstructionContent
from .system import SystemContent

MessageContentType = (
    ActionRequestContent
    | ActionResponseContent
    | AssistantResponseContent
    | SystemContent
    | InstructionContent
)


def create_message_content(
    data: dict | str | MessageContentType, /
) -> MessageContentType:

    if isinstance(data, MessageContentType):
        return data

    data = ln.to_dict(
        data,
        parse_strings=True,
        str_type_for_parsing="json",
        fuzzy_parse_strings=True,
        recursive=True,
    )
    if "tool_requests" in data:
        return ActionRequestContent.model_validate(data)

    elif "tool_response" in data:
        return ActionResponseContent.model_validate(data)

    elif "model_response" in data:
        return AssistantResponseContent.model_validate(data)

    elif "system" in data:
        return SystemContent.model_validate(data)

    elif "instruction" in data:
        return InstructionContent.model_validate(data)
    raise ValueError("Invalid message content data.")
