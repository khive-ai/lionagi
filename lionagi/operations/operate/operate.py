# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, JsonValue

from lionagi.fields.instruct import Instruct
from lionagi.ln.fuzzy import FuzzyMatchKeysParams
from lionagi.models import FieldModel, ModelParams
from lionagi.protocols.operatives.step import Operative
from lionagi.protocols.types import Instruction, Progression, SenderRecipient
from lionagi.service.imodel import iModel
from lionagi.session.branch import AlcallParams
from lionagi.utils import copy

from ..types import ActionContext, ChatContext, HandleValidation, ParseContext

if TYPE_CHECKING:
    from lionagi.session.branch import Branch, ToolRef


async def operate(
    branch: "Branch",
    *,
    instruct: Instruct = None,
    instruction: Instruction | JsonValue = None,
    guidance: JsonValue = None,
    context: JsonValue = None,
    sender: SenderRecipient = None,
    recipient: SenderRecipient = None,
    progression: Progression = None,
    imodel: iModel = None,  # deprecated, alias of chat_model
    chat_model: iModel = None,
    invoke_actions: bool = True,
    tool_schemas: list[dict] = None,
    images: list = None,
    image_detail: Literal["low", "high", "auto"] = None,
    parse_model: iModel = None,
    skip_validation: bool = False,
    tools: "ToolRef" = None,
    operative: "Operative" = None,
    response_format: type[BaseModel] = None,  # alias of operative.request_type
    actions: bool = False,
    reason: bool = False,
    call_params: AlcallParams = None,
    action_strategy: Literal["sequential", "concurrent"] = "concurrent",
    verbose_action: bool = False,
    field_models: list[FieldModel] = None,
    exclude_fields: list | dict | None = None,
    request_params: ModelParams = None,
    request_param_kwargs: dict = None,
    handle_validation: HandleValidation = "return_value",
    operative_model: type[BaseModel] = None,
    request_model: type[BaseModel] = None,
    include_token_usage_to_model: bool = False,
    clear_messages: bool = False,
    **kwargs,
) -> list | BaseModel | None | dict | str:
    # Handle deprecated parameters
    if operative_model:
        warnings.warn(
            "Parameter 'operative_model' is deprecated. Use 'response_format' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if imodel:
        warnings.warn(
            "Parameter 'imodel' is deprecated. Use 'chat_model' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if (
        (operative_model and response_format)
        or (operative_model and request_model)
        or (response_format and request_model)
    ):
        raise ValueError(
            "Cannot specify both `operative_model` and `response_format` (or `request_model`) "
            "as they are aliases of each other."
        )

    response_format = response_format or operative_model or request_model
    chat_model = chat_model or imodel or branch.chat_model
    parse_model = parse_model or chat_model

    # Convert dict-based instructions to Instruct if needed
    if isinstance(instruct, dict):
        instruct = Instruct(**instruct)

    # Or create a new Instruct if not provided
    instruct = instruct or Instruct(
        instruction=instruction,
        guidance=guidance,
        context=context,
    )

    # If reason or actions are requested, apply them to instruct
    if reason:
        instruct.reason = True
    if actions:
        instruct.actions = True
        if action_strategy:
            instruct.action_strategy = action_strategy

    # Build the Operative - always create it for backwards compatibility
    from lionagi.protocols.operatives.step import Step

    operative = Step.request_operative(
        request_params=request_params,
        reason=instruct.reason,
        actions=instruct.actions or actions,
        exclude_fields=exclude_fields,
        base_type=response_format,
        field_models=field_models,
        **(request_param_kwargs or {}),
    )
    # Use the operative's request_type which is a proper Pydantic model
    # created from field_models if provided
    final_response_format = operative.request_type

    # Build contexts
    chat_ctx = ChatContext(
        guidance=instruct.guidance,
        context=instruct.context,
        sender=sender or branch.user or "user",
        recipient=recipient or branch.id,
        response_format=final_response_format,
        progression=progression,
        tool_schemas=tool_schemas,
        images=images,
        image_detail=image_detail,
        plain_content=None,
        include_token_usage_to_model=include_token_usage_to_model,
        imodel=chat_model,
        imodel_kw=kwargs,
    )

    parse_ctx = None
    if final_response_format and not skip_validation:
        from ..parse.parse import get_default_call

        parse_ctx = ParseContext(
            response_format=final_response_format,
            fuzzy_match_params=FuzzyMatchKeysParams(),
            handle_validation="return_value",
            alcall_params=get_default_call(),
            imodel=parse_model,
            imodel_kw={},
        )

    action_ctx = None
    if invoke_actions and (instruct.actions or actions):
        from ..act.act import _get_default_call_params

        action_ctx = ActionContext(
            action_call_params=call_params or _get_default_call_params(),
            tools=tools,
            strategy=action_strategy
            or instruct.action_strategy
            or "concurrent",
            suppress_errors=True,
            verbose_action=verbose_action,
        )

    return await operate_v1(
        branch,
        instruction=instruct.instruction,
        chat_ctx=chat_ctx,
        parse_ctx=parse_ctx,
        action_ctx=action_ctx,
        handle_validation=handle_validation,
        invoke_actions=invoke_actions,
        skip_validation=skip_validation,
        clear_messages=clear_messages,
    )


async def operate_v1(
    branch: "Branch",
    instruction: JsonValue | Instruction,
    chat_ctx: ChatContext,
    action_ctx: ActionContext | None = None,
    parse_ctx: ParseContext | None = None,
    handle_validation: HandleValidation = "return_value",
    invoke_actions: bool = True,
    skip_validation: bool = False,
    clear_messages: bool = False,
    reason: bool = False,
    field_models: list[FieldModel] | None = None,
) -> BaseModel | dict | str | None:

    # 1. communicate chat context building to avoid changing parameters
    _cctx = copy(chat_ctx)
    _pctx = (
        copy(parse_ctx)
        if parse_ctx
        else ParseContext(
            response_format=chat_ctx.response_format,
            imodel=branch.parse_model,
        )
    )
    _pctx.handle_validation = "return_value"

    if tools := (action_ctx.tools or True) if action_ctx else None:
        _cctx.tool_schemas = branch.acts.get_tool_schema(tools=tools)

    t = type if isinstance(chat_ctx.response_format, type) else dict

    def normalize_field_model(fms):
        if not fms:
            return []
        if not isinstance(fms, list):
            return [fms]
        return fms

    fms = normalize_field_model(field_models)
    operative = None

    if t is type:
        from lionagi.protocols.operatives.step import Step

        operative = Step.request_operative(
            reason=reason,
            actions=bool(action_ctx is not None),
            base_type=chat_ctx.response_format,
            field_models=fms,
        )
        _cctx.response_format = operative.request_type
    elif field_models:
        dict_ = {}
        for fm in fms:
            if fm.name:
                dict_[fm.name] = str(fm.annotated())
        _cctx.response_format = dict_

    from ..communicate.communicate import communicate_v1

    result = await communicate_v1(
        branch,
        instruction,
        _cctx,
        _pctx,
        clear_messages,
        skip_validation=skip_validation,
        request_fields=None,
    )
    if skip_validation:
        return result
    if not isinstance(result, t):
        match handle_validation:
            case "return_value":
                return result
            case "return_none":
                return None
            case "raise":
                raise ValueError(
                    "Failed to parse the LLM response into the requested format."
                )
    if not invoke_actions:
        return result

    requests = (
        getattr(result, "action_requests", None)
        if t is type
        else result.get("action_requests", None)
    )

    action_response_models = None
    if action_ctx and requests is not None:
        from ..act.act import act_v1

        action_response_models = await act_v1(
            branch,
            requests,
            action_ctx,
        )

    if not action_response_models:
        return result

    if t is dict:
        result.update({"action_responses": action_response_models})
        return result

    from lionagi.protocols.operatives.step import Step

    operative.response_model = result
    operative = Step.respond_operative(
        operative=operative,
        additional_data={"action_responses": action_response_models},
    )
    return operative.response_model
