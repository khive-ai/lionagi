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
    return_operative: bool = False,
    actions: bool = False,
    reason: bool = False,
    call_params: AlcallParams = None,
    action_strategy: Literal["sequential", "concurrent"] = "concurrent",
    verbose_action: bool = False,
    field_models: list[FieldModel] = None,
    exclude_fields: list | dict | None = None,
    request_params: ModelParams = None,
    request_param_kwargs: dict = None,
    response_params: ModelParams = None,
    response_param_kwargs: dict = None,
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
    final_response_format = operative.request_type

    # If field_models provided for dict response, build dict format
    if field_models and not isinstance(response_format, type):
        dict_format = {}
        fms = (
            field_models if isinstance(field_models, list) else [field_models]
        )
        for fm in fms:
            if fm.name:
                dict_format[fm.name] = str(fm.annotated())
        final_response_format = dict_format

    # Build contexts
    chat_ctx = ChatContext(
        guidance=instruct.guidance,
        context=instruct.context,
        sender=sender or branch.user or "user",
        recipient=recipient or branch.id,
        response_format=final_response_format,
        progression=progression,
        tool_schemas=tool_schemas or [],
        images=images or [],
        image_detail=image_detail or "auto",
        plain_content="",
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
        operative=operative,
        response_params=response_params,
        response_param_kwargs=response_param_kwargs,
        handle_validation=handle_validation,
        invoke_actions=invoke_actions,
        skip_validation=skip_validation,
        return_operative=return_operative,
        clear_messages=clear_messages,
    )


async def operate_v1(
    branch: "Branch",
    instruction: JsonValue | Instruction,
    chat_ctx: ChatContext,
    parse_ctx: ParseContext | None = None,
    action_ctx: ActionContext | None = None,
    operative: Operative | None = None,
    response_params: ModelParams = None,
    response_param_kwargs: dict = None,
    handle_validation: HandleValidation = "return_value",
    invoke_actions: bool = True,
    skip_validation: bool = False,
    return_operative: bool = False,
    clear_messages: bool = False,
) -> BaseModel | dict | str | None:
    """Execute operation with contexts - clean implementation."""

    if clear_messages:
        branch.msgs.clear_messages()

    # Add tool schemas if action context provided
    if action_ctx and action_ctx.tools:
        # Need to update chat_ctx with tool schemas
        tools = action_ctx.tools if action_ctx.tools is not True else True
        tool_schemas = branch.acts.get_tool_schema(tools=tools)
        # Create modified chat context using copy
        from copy import deepcopy

        _chat_ctx = deepcopy(chat_ctx)
        _chat_ctx.tool_schemas = tool_schemas
    else:
        _chat_ctx = chat_ctx

    # Use communicate for chat + optional parse
    from ..communicate.communicate import communicate_v1

    result = await communicate_v1(
        branch,
        instruction,
        _chat_ctx,
        parse_ctx,
        clear_messages=False,  # Already cleared above if needed
        skip_validation=skip_validation,
        request_fields=None,
    )

    # Populate operative with raw response
    if operative:
        operative.response_str_dict = (
            result if isinstance(result, str) else result
        )

    # If skip_validation, return early
    if skip_validation:
        return operative if (return_operative and operative) else result

    # Validation check
    expected_type = (
        type if isinstance(chat_ctx.response_format, type) else dict
    )
    if not isinstance(result, expected_type):
        match handle_validation:
            case "return_value":
                return result
            case "return_none":
                return None
            case "raise":
                raise ValueError(
                    "Failed to parse the LLM response into the requested format."
                )

    # Update operative with response if we have one
    if operative and isinstance(result, BaseModel):
        operative.response_model = result

    # Return early if not invoking actions
    if not invoke_actions or not action_ctx:
        return operative if (return_operative and operative) else result

    # Check for action requests
    if isinstance(result, BaseModel):
        action_requests = getattr(result, "action_requests", None)
    else:
        action_requests = (
            result.get("action_requests", None)
            if isinstance(result, dict)
            else None
        )

    # No actions needed
    if not action_requests:
        return operative if (return_operative and operative) else result

    # Execute actions
    from ..act.act import act_v1

    action_response_models = await act_v1(
        branch,
        action_requests,
        action_ctx,
    )

    # Handle dict response with actions
    if isinstance(result, dict):
        result["action_responses"] = action_response_models
        return result

    # Handle operative response with actions
    if operative:
        from lionagi.protocols.operatives.step import Step

        operative = Step.respond_operative(
            response_params=response_params,
            operative=operative,
            additional_data={"action_responses": action_response_models},
            **(response_param_kwargs or {}),
        )
        return operative if return_operative else operative.response_model

    # Fallback
    return result
