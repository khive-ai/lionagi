# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import TYPE_CHECKING, Literal, Union

from pydantic import BaseModel, JsonValue

from lionagi.ln import AlcallParams
from lionagi.ln.fuzzy import FuzzyMatchKeysParams
from lionagi.ln.types import Spec
from lionagi.models import FieldModel
from lionagi.protocols.generic import Progression
from lionagi.protocols.messages import Instruction, SenderRecipient

from ..fields import Instruct
from ..types import ActionParam, ChatParam, HandleValidation, ParseParam

if TYPE_CHECKING:
    from lionagi.service.imodel import iModel
    from lionagi.session.branch import Branch, ToolRef

    from .operative import Operative


def prepare_operate_kw(
    branch: "Branch",
    *,
    instruct: Instruct = None,
    instruction: Instruction | JsonValue = None,
    guidance: JsonValue = None,
    context: JsonValue = None,
    sender: SenderRecipient = None,
    recipient: SenderRecipient = None,
    progression: Progression = None,
    imodel: "iModel" = None,  # deprecated
    chat_model: "iModel" = None,
    invoke_actions: bool = True,
    tool_schemas: list[dict] = None,
    images: list = None,
    image_detail: Literal["low", "high", "auto"] = None,
    parse_model: "iModel" = None,
    skip_validation: bool = False,
    handle_validation: HandleValidation = "return_value",
    tools: "ToolRef" = None,
    operative: "Operative" = None,
    response_format: type[BaseModel] = None,
    actions: bool = False,
    reason: bool = False,
    call_params: AlcallParams = None,
    action_strategy: Literal["sequential", "concurrent"] = "concurrent",
    verbose_action: bool = False,
    field_models: list[FieldModel | Spec] = None,
    operative_model: type[BaseModel] = None,  # deprecated
    request_model: type[BaseModel] = None,  # deprecated
    include_token_usage_to_model: bool = False,
    clear_messages: bool = False,
    **kwargs,
) -> dict:
    # Handle deprecated parameters
    if operative_model:
        warnings.warn(
            "Parameter 'operative_model' is deprecated. Use 'response_format'.",
            DeprecationWarning,
            stacklevel=2,
        )
    if request_model:
        warnings.warn(
            "Parameter 'request_model' is deprecated. Use 'response_format'.",
            DeprecationWarning,
            stacklevel=2,
        )
    if imodel:
        warnings.warn(
            "Parameter 'imodel' is deprecated. Use 'chat_model'.",
            DeprecationWarning,
            stacklevel=2,
        )

    if (
        (operative_model and response_format)
        or (operative_model and request_model)
        or (response_format and request_model)
    ):
        raise ValueError(
            "Cannot specify multiple of: operative_model, response_format, request_model"
        )

    response_format = response_format or operative_model or request_model
    chat_model = chat_model or imodel or branch.chat_model
    parse_model = parse_model or chat_model

    # Convert dict-based instructions
    if isinstance(instruct, dict):
        instruct = Instruct(**instruct)

    instruct = instruct or Instruct(
        instruction=instruction,
        guidance=guidance,
        context=context,
    )

    if reason:
        instruct.reason = True
    if actions:
        instruct.actions = True
        if action_strategy:
            instruct.action_strategy = action_strategy

    # Convert field_models to Spec if needed
    fields_dict = None
    if field_models:
        fields_dict = {}
        for fm in field_models:
            # Convert FieldModel to Spec
            if isinstance(fm, FieldModel):
                spec = fm.to_spec()
            elif isinstance(fm, Spec):
                spec = fm
            else:
                raise TypeError(f"Expected FieldModel or Spec, got {type(fm)}")

            if spec.name:
                fields_dict[spec.name] = spec

    # Build Operative if needed
    operative = None
    if instruct.reason or instruct.actions or response_format or fields_dict:
        from .step import Step

        operative = Step.request_operative(
            base_type=response_format,
            reason=instruct.reason,
            actions=instruct.actions or actions,
            fields=fields_dict,
        )

        # Create response model
        operative = Step.respond_operative(operative)

    final_response_format = operative.response_type if operative else response_format

    # Build contexts
    chat_param = ChatParam(
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

    parse_param = None
    if final_response_format and not skip_validation:
        from ..parse.parse import get_default_call

        parse_param = ParseParam(
            response_format=final_response_format,
            fuzzy_match_params=FuzzyMatchKeysParams(),
            handle_validation=handle_validation,
            alcall_params=get_default_call(),
            imodel=parse_model,
            imodel_kw={},
        )

    action_param = None
    if invoke_actions and (instruct.actions or actions):
        from ..act.act import _get_default_call_params

        action_param = ActionParam(
            action_call_params=call_params or _get_default_call_params(),
            tools=tools,
            strategy=action_strategy or instruct.action_strategy or "concurrent",
            suppress_errors=True,
            verbose_action=verbose_action,
        )

    return {
        "instruction": instruct.instruction,
        "chat_param": chat_param,
        "parse_param": parse_param,
        "action_param": action_param,
        "handle_validation": handle_validation,
        "invoke_actions": invoke_actions,
        "skip_validation": skip_validation,
        "clear_messages": clear_messages,
        "operative": operative,
    }


async def operate(
    branch: "Branch",
    instruction: JsonValue | Instruction,
    chat_param: ChatParam,
    action_param: ActionParam | None = None,
    parse_param: ParseParam | None = None,
    handle_validation: HandleValidation = "return_value",
    invoke_actions: bool = True,
    skip_validation: bool = False,
    clear_messages: bool = False,
    reason: bool = False,
    field_models: list[FieldModel | Spec] | None = None,
    operative: Union["Operative", None] = None,
) -> BaseModel | dict | str | None:
    """Execute operation with optional action handling.

    Args:
        branch: Branch instance
        instruction: Instruction or JSON value
        chat_param: Chat parameters
        action_param: Action parameters
        parse_param: Parse parameters
        handle_validation: Validation handling strategy
        invoke_actions: Whether to invoke actions
        skip_validation: Whether to skip validation
        clear_messages: Whether to clear messages
        reason: Whether to include reasoning
        field_models: List of FieldModel or Spec objects
        operative: Operative instance

    Returns:
        Result of operation
    """
    _cctx = chat_param
    _pctx = (
        parse_param.with_updates(handle_validation="return_value")
        if parse_param
        else ParseParam(
            response_format=chat_param.response_format,
            imodel=branch.parse_model,
            handle_validation="return_value",
        )
    )

    # Update tool schemas
    if tools := (action_param.tools or True) if action_param else None:
        tool_schemas = branch.acts.get_tool_schema(tools=tools)
        _cctx = _cctx.with_updates(tool_schemas=tool_schemas)

    # Extract model class
    model_class = None
    if chat_param.response_format is not None:
        if isinstance(chat_param.response_format, type) and issubclass(
            chat_param.response_format, BaseModel
        ):
            model_class = chat_param.response_format
        elif isinstance(chat_param.response_format, BaseModel):
            model_class = type(chat_param.response_format)

    # Convert field_models to fields dict
    fields_dict = None
    if field_models:
        fields_dict = {}
        for fm in field_models:
            if isinstance(fm, FieldModel):
                spec = fm.to_spec()
            elif isinstance(fm, Spec):
                spec = fm
            else:
                raise TypeError(f"Expected FieldModel or Spec, got {type(fm)}")

            if spec.name:
                fields_dict[spec.name] = spec

    # Create operative if needed
    if not operative and (model_class or action_param or fields_dict):
        from .step import Step

        operative = Step.request_operative(
            base_type=model_class,
            reason=reason,
            actions=bool(action_param),
            fields=fields_dict,
        )
        operative = Step.respond_operative(operative)

        # Update contexts
        response_fmt = operative.response_type or model_class
        if response_fmt:
            _cctx = _cctx.with_updates(response_format=response_fmt)
            _pctx = _pctx.with_updates(response_format=response_fmt)

    from ..communicate.communicate import communicate

    result = await communicate(
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

    if model_class and not isinstance(result, model_class):
        match handle_validation:
            case "return_value":
                return result
            case "return_none":
                return None
            case "raise":
                raise ValueError("Failed to parse LLM response.")

    if not invoke_actions:
        return result

    # Handle actions
    requests = (
        getattr(result, "action_requests", None)
        if model_class
        else result.get("action_requests", None)
    )

    action_response_models = None
    if action_param and requests is not None:
        from ..act.act import act

        action_response_models = await act(branch, requests, action_param)

    if not action_response_models:
        return result

    # Filter None values
    action_response_models = [r for r in action_response_models if r is not None]

    if not action_response_models:
        return result

    if not model_class:  # Dict response
        result.update({"action_responses": action_response_models})
        return result

    # If we have model_class, we must have operative (created at line 268)
    # First set the response_model to the existing result
    operative.response_model = result
    # Then update it with action_responses
    operative.update_response_model(data={"action_responses": action_response_models})
    return operative.response_model
