from pydantic import JsonValue

from lionagi.protocols.types import AssistantResponse, Instruction

from .types import ChatContext, ParseContext


async def communicate(
    branch,
    instruction: JsonValue | Instruction,
    chat_ctx: ChatContext,
    parse_ctx: ParseContext,
    clear_messages: bool = False,
):
    if clear_messages:
        branch.msgs.clear_messages()

    from .chat import chat

    ins, res = await chat(branch, instruction, chat_ctx, True)

    branch.msgs.add_message(instruction=ins)
    branch.msgs.add_message(assistant_response=res)

    if chat_ctx.response_format is not None:
        from .parse import parse

        out, res2 = await parse(branch, res.response, parse_ctx, True)
        if res2 and isinstance(res2, AssistantResponse):
            res.metadata["parse_model_response"] = res2.model_response
        return out

    return res.response
