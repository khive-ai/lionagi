# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
from typing import Any, Literal

from pydantic import BaseModel

from lionagi.fields.action import ActionRequestModel, ActionResponseModel
from lionagi.models import FieldModel
from lionagi.protocols.action.manager import ActionConfig
from lionagi.protocols.messages.action_response import ActionResponse
from lionagi.protocols.types import (
    ActionRequest,
    Operative,
    OperativeConfig,
    Progression,
    SenderRecipient,
    ToolRef,
)
from lionagi.service.imodel import iModel
from lionagi.session.branch import Branch


def handle_operative(
    response_format: type[BaseModel] = None,
    operative_config: OperativeConfig | None = None,
    field_models: list[FieldModel] = None,
    llm_reparse: bool = None,
):
    operative = None
    if any(
        i is not None
        for i in (response_format, operative_config, field_models)
    ):
        operative = Operative(
            config=operative_config,
            response_format=response_format,
            field_models=field_models,
        )
        if llm_reparse is not None:
            operative.config.llm_reparse = llm_reparse
        operative.create_response_type()
        if operative.response_type is None:
            raise ValueError(
                "Operative configuration must define a response type or format."
            )
    return operative


async def operate(
    branch: Branch,
    operation_context: dict,
    sender: SenderRecipient = None,
    recipient: SenderRecipient = None,
    parse_model: iModel = None,
    llm_reparse: bool = None,
    handle_parse: Literal[
        "return_value", "return_none", "raise"
    ] = "return_value",
    skip_validation: bool = False,
    response_format: type[BaseModel] = None,
    operative_config: OperativeConfig | None = None,
    field_models: list[FieldModel] = None,
    invoke_actions: bool = True,
    action_strategy: Literal["sequential", "concurrent"] = "concurrent",
    action_config: ActionConfig = None,
    verbose_action: bool = False,
    suppress_action_errors: bool = True,
    tools: ToolRef | list[dict] = None,
    progression: Progression = None,
    chat_model: iModel = None,
    include_token_usage_to_model: bool = False,
    clear_messages: bool = False,
    **kwargs: Any,
):
    if clear_messages:
        await branch.msgs.aclear_messages()

    # 1. handle operative and operation context
    operative = handle_operative(
        response_format=response_format,
        operative_config=operative_config,
        field_models=field_models,
        llm_reparse=llm_reparse,
    )
    context = []
    if (ctx := operation_context.get("context")) is not None:
        context.append(ctx)
    if additional_context := {
        k: v
        for k, v in operation_context.items()
        if k
        not in {"instruction", "guidance", "context", "images", "image_detail"}
    }:
        context.append(additional_context)

    # 2. initial model call
    ins, res = await branch.chat(
        instruction=operation_context.get("instruction"),
        guidance=operation_context.get("guidance"),
        images=operation_context.get("images"),
        image_detail=operation_context.get("image_detail"),
        context=context,
        sender=sender,
        recipient=recipient,
        response_format=operative.request_type,
        progression=progression,
        imodel=chat_model,
        tool_schemas=branch.acts.get_tool_schema(tools=tools),
        return_ins_res_message=True,
        include_token_usage_to_model=include_token_usage_to_model,
        **kwargs,
    )
    branch.msgs.add_message(instruction=ins)
    branch.msgs.add_message(assistant_response=res)

    if skip_validation or operative is None:
        return res.response

    # 3. handle initial validation
    parse_failed = False
    try:
        operative.update_response_model(res.response)
    except Exception as e:
        if not operative.config.llm_reparse:
            if handle_parse == "return_value":
                return res.response
            if handle_parse == "return_none":
                return None
            if handle_parse == "raise":
                raise e
        else:
            parse_failed = True

    # 4. handle reparse if needed
    if parse_failed:
        parse_model = parse_model or branch.parse_model

        async def _reparse():
            _in, _re = await branch.chat(
                instruction="Reformat text into specified model",
                guidance="DO NOT REPHRASE!!! fix the json string, into the required structured response format according to model schema",
                context=[{"text_to_reformat": res.response}],
                response_format=operative.request_type,
                progression=(
                    progression
                    if operative.config.llm_reparse_with_context
                    else []
                ),
                sender=branch.user,
                recipient=branch.id,
                imodel=parse_model or chat_model,
                return_ins_res_message=True,
                include_token_usage_to_model=False,
                **(operative.config.llm_reparse_params or {}),
            )
            with contextlib.suppress(Exception):
                operative.update_response_model(_re.response)
                return False, _in, _re
            return True, _in, _re

        in_, re_ = None, None
        for _ in range(operative.config.max_reparse_attempts):
            parse_failed, in_, re_ = await _reparse()
            if not parse_failed:
                break

        if parse_failed:
            if handle_parse == "return_value":
                return res.response
            if handle_parse == "return_none":
                return None
            if handle_parse == "raise":
                raise ValueError(
                    "Failed to parse the LLM response into the requested format."
                )

        # by here, we should have a valid response model
        branch.msgs.add_message(instruction=in_)
        branch.msgs.add_message(assistant_response=re_)

    # 5. gather action requests if any
    response_model = operative.response_model

    a_req = getattr(response_model, "action_requests", None)
    if a_req is None:
        return response_model

    a_req: list[ActionRequestModel] = (
        [a_req] if not isinstance(a_req, list) else a_req
    )

    a_req_msgs = []
    for r in a_req:
        r_msg = ActionRequest.create(
            function=r.function,
            arguments=r.arguments,
            sender=sender,
            recipient=branch.id,
        )
        branch.msgs.add_message(action_request=r_msg)
        a_req_msgs.append(r_msg)

    # 6. invoke actions if requested
    if invoke_actions:
        action_responses: list[ActionResponse] = await branch.act(
            action_request=response_model.action_requests,
            strategy=action_strategy,
            action_config=action_config,
            verbose_action=verbose_action,
            surpress_errors=suppress_action_errors,
        )
        action_response_models = [
            ActionResponseModel(
                function=r.function,
                arguments=r.arguments,
                output=r.output,
            )
            for r in action_responses
        ]

        operative.update_response_model(
            data={"action_responses": action_response_models}
        )

    return operative.response_model
