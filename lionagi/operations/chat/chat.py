# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from pydantic import JsonValue

from lionagi.protocols.types import (
    ActionResponse,
    AssistantResponse,
    Instruction,
    Log,
)
from lionagi.utils import copy, to_list

from ..types import ChatParam

if TYPE_CHECKING:
    from lionagi.session.branch import Branch


async def chat(
    branch: "Branch",
    instruction: JsonValue | Instruction,
    chat_param: ChatParam,
    return_ins_res_message: bool = False,
) -> tuple[Instruction, AssistantResponse]:
    params = chat_param.to_dict(
        exclude={
            "imodel",
            "imodel_kw",
            "include_token_usage_to_model",
            "progression",
        }
    )
    params["sender"] = chat_param.sender or branch.user or "user"
    params["recipient"] = chat_param.recipient or branch.id
    params["instruction"] = instruction

    ins = branch.msgs.create_instruction(**params)

    _use_ins, _use_msgs, _act_res = None, [], []
    progression = chat_param.progression or branch.msgs.progression

    for msg in (branch.msgs.messages[j] for j in progression):
        if isinstance(msg, ActionResponse):
            _act_res.append(msg)

        if isinstance(msg, AssistantResponse):
            _use_msgs.append(
                msg.model_copy(update={"content": msg.content.with_updates()})
            )

        if isinstance(msg, Instruction):
            j = msg.model_copy(update={"content": msg.content.with_updates()})
            j.content.tool_schemas.clear()
            j.content.response_format = None
            j.content._schema_dict = None
            j.content._model_class = None

            if _act_res:
                # Convert ActionResponseContent to dicts for proper rendering
                d_ = []
                for k in to_list(_act_res, flatten=True, unique=True):
                    if hasattr(k.content, "function"):  # ActionResponseContent
                        d_.append(
                            {
                                "function": k.content.function,
                                "arguments": k.content.arguments,
                                "output": k.content.output,
                            }
                        )
                    else:
                        d_.append(k.content)
                j.content.prompt_context.extend(
                    [z for z in d_ if z not in j.content.prompt_context]
                )
                _use_msgs.append(j)
                _act_res = []
            else:
                _use_msgs.append(j)

    if _act_res:
        j = ins.model_copy(update={"content": ins.content.with_updates()})
        # Convert ActionResponseContent to dicts for proper rendering
        d_ = []
        for k in to_list(_act_res, flatten=True, unique=True):
            if hasattr(k.content, "function"):  # ActionResponseContent
                d_.append(
                    {
                        "function": k.content.function,
                        "arguments": k.content.arguments,
                        "output": k.content.output,
                    }
                )
            else:
                d_.append(k.content)
        j.content.prompt_context.extend(
            [z for z in d_ if z not in j.content.prompt_context]
        )
        _use_ins = j

    messages = _use_msgs
    if _use_msgs and len(_use_msgs) > 1:
        _msgs = [_use_msgs[0]]

        for i in _use_msgs[1:]:
            if isinstance(i, AssistantResponse):
                if isinstance(_msgs[-1], AssistantResponse):
                    _msgs[-1].content.assistant_response = (
                        f"{_msgs[-1].content.assistant_response}\n\n{i.content.assistant_response}"
                    )
                else:
                    _msgs.append(i)
            else:
                if isinstance(_msgs[-1], AssistantResponse):
                    _msgs.append(i)
        messages = _msgs

    # All endpoints now assume sequential exchange (system message embedded in first user message)
    if branch.msgs.system:
        messages = [msg for msg in messages if msg.role != "system"]
        first_instruction = None

        if len(messages) == 0:
            first_instruction = copy(ins)
            first_instruction.content.guidance = (
                branch.msgs.system.rendered
                + (first_instruction.content.guidance or "")
            )
            messages.append(first_instruction)
        elif len(messages) >= 1:
            first_instruction = messages[0]
            if not isinstance(first_instruction, Instruction):
                raise ValueError(
                    "First message in progression must be an Instruction or System"
                )
            first_instruction = copy(first_instruction)
            first_instruction.content.guidance = (
                branch.msgs.system.rendered
                + (first_instruction.content.guidance or "")
            )
            messages[0] = first_instruction
            msg_to_append = _use_ins or ins
            if msg_to_append is not None:
                messages.append(msg_to_append)

    else:
        msg_to_append = _use_ins or ins
        if msg_to_append is not None:
            messages.append(msg_to_append)

    kw = (chat_param.imodel_kw or {}).copy()

    # Filter out messages with None chat_msg
    chat_msgs = []
    for msg in messages:
        if msg is not None and hasattr(msg, "chat_msg"):
            chat_msg = msg.chat_msg
            if chat_msg is not None:
                chat_msgs.append(chat_msg)

    kw["messages"] = chat_msgs

    imodel = chat_param.imodel or branch.chat_model
    meth = imodel.stream if "stream" in kw and kw["stream"] else imodel.invoke

    if meth is imodel.invoke:
        # Only set if it's not the Unset sentinel value
        if not chat_param._is_sentinel(chat_param.include_token_usage_to_model):
            kw["include_token_usage_to_model"] = (
                chat_param.include_token_usage_to_model
            )
    api_call = await meth(**kw)

    branch._log_manager.log(api_call)

    if return_ins_res_message:
        # Wrap result in `AssistantResponse` and return
        return ins, AssistantResponse.from_response(
            api_call.response,
            sender=branch.id,
            recipient=branch.user,
        )
    return AssistantResponse.from_response(
        api_call.response,
        sender=branch.id,
        recipient=branch.user,
    ).response
