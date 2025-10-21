# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, JsonValue

from lionagi.ln import AlcallParams
from lionagi.ln.fuzzy import FuzzyMatchKeysParams
from lionagi.ln.types import Spec
from lionagi.models import FieldModel, ModelParams
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
    imodel: "iModel" = None,  # deprecated, alias of chat_model
    chat_model: "iModel" = None,
    invoke_actions: bool = True,
    tool_schemas: list[dict] = None,
    images: list = None,
    image_detail: Literal["low", "high", "auto"] = None,
    parse_model: "iModel" = None,
    skip_validation: bool = False,
    tools: "ToolRef" = None,
    operative: "Operative" = None,
    response_format: type[BaseModel] = None,  # alias of operative.request_type
    actions: bool = False,
    reason: bool = False,
    call_params: AlcallParams = None,
    action_strategy: Literal["sequential", "concurrent"] = "concurrent",
    verbose_action: bool = False,
    field_models: list[FieldModel | Spec] = None,
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

    # Build response format from base model + field_models/reason/actions
    final_response_format = response_format

    # Build operative if we need to add reason/actions fields
    operative = None
    if instruct.reason or instruct.actions or actions:
        from .step import Step

        operative = Step.request_operative(
            base_type=response_format,  # Use response_format as base
            reason=instruct.reason,
            actions=instruct.actions or actions,
        )

        # Create response model with action_responses field
        operative = Step.respond_operative(operative)
        final_response_format = operative.response_type

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
            handle_validation="return_value",
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
            strategy=action_strategy
            or instruct.action_strategy
            or "concurrent",
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
        "operative": operative,  # Pass the operative if created
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
    operative: Optional["Operative"] = None,
) -> BaseModel | dict | str | None:

    # 1. communicate chat context building to avoid changing parameters
    # Start with base chat param
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

    # Update tool schemas if needed
    if tools := (action_param.tools or True) if action_param else None:
        tool_schemas = branch.acts.get_tool_schema(tools=tools)
        _cctx = _cctx.with_updates(tool_schemas=tool_schemas)

    # Extract model class from response_format (can be class, instance, or dict)
    model_class = None
    if chat_param.response_format is not None:
        if isinstance(chat_param.response_format, type) and issubclass(
            chat_param.response_format, BaseModel
        ):
            model_class = chat_param.response_format
        elif isinstance(chat_param.response_format, BaseModel):
            model_class = type(chat_param.response_format)

    def normalize_field_model(fms):
        if not fms:
            return []
        if not isinstance(fms, list):
            return [fms]
        return fms

    fms = normalize_field_model(field_models)

    # Use operative passed from prepare_operate_kw if available
    if operative:
        # Operative was already created with proper fields
        response_fmt = (
            operative.response_type or operative.request_type or model_class
        )
        if response_fmt:
            _cctx = _cctx.with_updates(response_format=response_fmt)
            _pctx = _pctx.with_updates(response_format=response_fmt)
    elif model_class or action_param:
        # Create operative if we have a model class OR if action_param is provided
        # (action_param means we need action fields even without explicit model)
        from .step import Step

        # Convert field_models to fields dict if provided
        fields_dict = None
        if fms:
            from lionagi.ln.types import Spec
            from lionagi.models import FieldModel

            fields_dict = {}
            for fm in fms:
                # Convert FieldModel to Spec
                if isinstance(fm, FieldModel):
                    spec = fm.to_spec()
                elif isinstance(fm, Spec):
                    spec = fm
                else:
                    raise TypeError(
                        f"Expected FieldModel or Spec, got {type(fm)}"
                    )

                # Get the field name from the spec
                field_name = (
                    spec.name if hasattr(spec, "name") else spec.get("name")
                )
                if field_name:
                    fields_dict[field_name] = spec

        operative = Step.request_operative(
            base_type=model_class,  # Use model_class as base if available
            reason=reason,
            actions=bool(
                action_param is not None
            ),  # Add action fields if action_param exists
            fields=fields_dict,
        )

        # Create response model with action_responses field
        operative = Step.respond_operative(operative)

        # Update contexts with new response format
        response_fmt = (
            operative.response_type or operative.request_type or model_class
        )
        if response_fmt:
            _cctx = _cctx.with_updates(response_format=response_fmt)
            _pctx = _pctx.with_updates(response_format=response_fmt)
    elif field_models:
        dict_ = {}
        for fm in fms:
            if fm.name:
                dict_[fm.name] = str(fm.annotated())
        # Update contexts with dict format
        _cctx = _cctx.with_updates(response_format=dict_)
        _pctx = _pctx.with_updates(response_format=dict_)

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
                raise ValueError(
                    "Failed to parse the LLM response into the requested format."
                )
    if not invoke_actions:
        return result

    requests = (
        getattr(result, "action_requests", None)
        if model_class
        else result.get("action_requests", None)
    )

    action_response_models = None
    if action_param and requests is not None:
        from ..act.act import act

        action_response_models = await act(
            branch,
            requests,
            action_param,
        )

    if not action_response_models:
        return result

    # Filter out None values from action responses
    action_response_models = [
        r for r in action_response_models if r is not None
    ]

    if not action_response_models:  # All were None
        return result

    if not model_class:  # Dict response
        result.update({"action_responses": action_response_models})
        return result

    # If we have an operative, update its response model with action responses
    if operative:
        operative.response_model = result

        # If operative needs response type created, create it
        if not operative._response_model_cls:
            from .step import Step

            operative = Step.respond_operative(operative=operative)

        # Update the response model with action responses data
        if operative._response_model_cls:
            # Convert result to dict if it's a BaseModel
            if isinstance(result, BaseModel):
                result_dict = result.model_dump()
            else:
                result_dict = result
            result_dict["action_responses"] = action_response_models
            operative.response_model = (
                operative._response_model_cls.model_validate(result_dict)
            )

        return operative.response_model
    else:
        # No operative, just add action_responses to the result
        if isinstance(result, BaseModel):
            # Try to add action_responses field dynamically if possible
            setattr(result, "action_responses", action_response_models)
        return result
