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

from ..types import ChatContext

if TYPE_CHECKING:
    from lionagi.session.branch import Branch


async def chat(
    branch: "Branch",
    instruction: JsonValue | Instruction,
    chat_ctx: ChatContext,
    return_ins_res_message: bool = False,
) -> tuple[Instruction, AssistantResponse]:
    params = chat_ctx.to_dict(
        exclude={
            "imodel",
            "imodel_kw",
            "include_token_usage_to_model",
            "progression",
        }
    )
    params["sender"] = chat_ctx.sender or branch.user or "user"
    params["recipient"] = chat_ctx.recipient or branch.id
    params["instruction"] = instruction

    ins = branch.msgs.create_instruction(**params)

    _use_ins, _use_msgs, _act_res = None, [], []
    progression = chat_ctx.progression or branch.msgs.progression

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
            j.content.response_schema = None
            j.content.response_format = None

            if _act_res:
                d_ = [
                    k.content
                    for k in to_list(_act_res, flatten=True, unique=True)
                ]
                j.content.context.extend(
                    [z for z in d_ if z not in j.content.context]
                )
                _use_msgs.append(j)
                _act_res = []
            else:
                _use_msgs.append(j)

    if _act_res:
        j = ins.model_copy(update={"content": ins.content.with_updates()})
        d_ = [k.content for k in to_list(_act_res, flatten=True, unique=True)]
        j.content.context.extend([z for z in d_ if z not in j.content.context])
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
            messages.append(_use_ins or ins)

    else:
        messages.append(_use_ins or ins)

    kw = (chat_ctx.imodel_kw or {}).copy()
    kw["messages"] = [i.chat_msg for i in messages]

    imodel = chat_ctx.imodel or branch.chat_model
    meth = imodel.stream if "stream" in kw and kw["stream"] else imodel.invoke

    if meth is imodel.invoke:
        kw["include_token_usage_to_model"] = (
            chat_ctx.include_token_usage_to_model
        )
    api_call = await meth(**kw)

    branch._log_manager.log(Log.create(api_call))

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
