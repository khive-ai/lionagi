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
from lionagi.protocols.messages import (
    Instruction,
    JsonFormatter,
    LndlFormatter,
    SenderRecipient,
)

from ..fields import Instruct
from ..types import (
    ActionParam,
    ChatParam,
    HandleValidation,
    Middle,
    ParseParam,
    RunParam,
)

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
    lndl: bool = False,
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
    stream_persist: bool = False,
    persist_dir: str | None = None,
    middle: Middle | None = None,
    **kwargs,
) -> dict:
    # Handle deprecated parameters
    if operative_model:
        warnings.warn(
            "Parameter 'operative_model' is deprecated and will be removed in v0.21.0. "
            "Use 'response_format'.",
            DeprecationWarning,
            stacklevel=2,
        )
    if request_model:
        warnings.warn(
            "Parameter 'request_model' is deprecated and will be removed in v0.21.0. "
            "Use 'response_format'.",
            DeprecationWarning,
            stacklevel=2,
        )
    if imodel:
        warnings.warn(
            "Parameter 'imodel' is deprecated and will be removed in v0.21.0. Use 'chat_model'.",
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
    _has_actions = instruct.actions or actions
    _actions_for_request = _has_actions and not lndl  # LNDL: no action fields in prompt
    _actions_for_response = _has_actions  # but still need them in response model
    _need_operative = instruct.reason or _has_actions or fields_dict or response_format
    if _need_operative and (
        response_format or _has_actions or fields_dict or instruct.reason
    ):
        from .step import Step

        operative = Step.request_operative(
            base_type=response_format,
            reason=instruct.reason,
            actions=_has_actions,  # always True when actions requested (for response model)
            fields=fields_dict,
        )
        operative = Step.respond_operative(operative)

    # LNDL: prompt uses the clean user model (no action fields)
    # JSON: prompt uses the request model (with action_required/action_requests)
    if lndl:
        final_response_format = response_format
    else:
        final_response_format = operative.request_type if operative else response_format
    # Choose ChatParam vs RunParam. RunParam is required when the middle
    # streams via run() (CLI endpoints, explicit stream_persist, or when
    # caller passes a middle that needs persist_dir). Defaulting to
    # RunParam for CLI endpoints keeps the call sites free of path plumbing.
    is_cli = bool(getattr(chat_model, "is_cli", False))
    use_run_param = is_cli or stream_persist or persist_dir is not None

    param_cls = RunParam if use_run_param else ChatParam
    param_kw = dict(
        guidance=instruct.guidance,
        context=instruct.context,
        sender=sender or branch.user or "user",
        recipient=recipient or branch.id,
        response_format=final_response_format,
        formatter=LndlFormatter if lndl else JsonFormatter,
        progression=progression,
        tool_schemas=tool_schemas,
        images=images,
        image_detail=image_detail,
        plain_content=None,
        include_token_usage_to_model=include_token_usage_to_model,
        imodel=chat_model,
        imodel_kw=kwargs,
    )
    if use_run_param:
        param_kw["stream_persist"] = stream_persist
        if persist_dir is not None:
            param_kw["persist_dir"] = persist_dir
    chat_param = param_cls(**param_kw)

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
            formatter=chat_param.formatter,
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
        "middle": middle,
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
    middle: Middle | None = None,
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
            formatter=chat_param.formatter,
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
    _is_lndl = chat_param.formatter is LndlFormatter
    _actions_here = bool(action_param) and not _is_lndl
    if not operative and (model_class or _actions_here or fields_dict):
        from .step import Step

        operative = Step.request_operative(
            base_type=model_class,
            reason=reason,
            actions=_actions_here,
            fields=fields_dict,
        )
        operative = Step.respond_operative(operative)

        # Update contexts — use request_type (excludes action_responses)
        request_fmt = operative.request_type or model_class
        if request_fmt:
            _cctx = _cctx.with_updates(response_format=request_fmt)
            _pctx = _pctx.with_updates(response_format=request_fmt)

    if middle is None:
        if isinstance(_cctx, RunParam) or getattr(branch.chat_model, "is_cli", False):
            from ..run.run import run_and_collect

            middle = run_and_collect
        else:
            from ..communicate.communicate import communicate

            middle = communicate

    result = await middle(
        branch,
        instruction,
        _cctx,
        _pctx,
        clear_messages,
        skip_validation=skip_validation,
    )

    if skip_validation:
        return result

    # For LNDL with ActionCalls, defer model validation until after execution
    _is_lndl = chat_param.formatter is LndlFormatter
    _has_action_calls = False
    if _is_lndl and isinstance(result, dict):
        from lionagi.lndl.types import ActionCall as _AC

        _has_action_calls = any(isinstance(v, _AC) for v in result.values())

    if model_class and not isinstance(result, model_class) and not _has_action_calls:
        match handle_validation:
            case "return_value":
                return result
            case "return_none":
                return None
            case "raise":
                expected_name = getattr(model_class, "__name__", repr(model_class))
                received_snippet = repr(result)[:200]
                raise ValueError(
                    f"Failed to parse LLM response into '{expected_name}'. "
                    f"Received (truncated): {received_snippet}. "
                    f"Hint: verify the model supports structured JSON output "
                    f"(e.g. response_format / function-calling) for this provider."
                )

    if not invoke_actions:
        return result

    # LNDL path: parse <lact> tags from raw text, or execute ActionCalls from parsed dict
    _is_lndl = chat_param.formatter is LndlFormatter

    # Case: LNDL parse already returned dict with ActionCall values
    if _is_lndl and isinstance(result, dict) and action_param:
        from lionagi.lndl.types import ActionCall

        from ..act.act import act
        from ..fields import ActionRequestModel

        action_calls = {k: v for k, v in result.items() if isinstance(v, ActionCall)}
        if action_calls:
            action_requests = [
                ActionRequestModel(function=ac.function, arguments=ac.arguments)
                for ac in action_calls.values()
            ]
            action_responses = await act(branch, action_requests, action_param)
            action_responses = [r for r in (action_responses or []) if r is not None]

            # Find the spec names from OUT{} via the assistant's raw response
            # The dict keys from parse are aliases — need to map back to spec names
            # Get the raw response to re-parse OUT{}
            raw_response = None
            for m in branch.messages:
                if hasattr(m.content, "assistant_response"):
                    raw_response = m.content.assistant_response

            if raw_response:
                from lionagi.lndl import Lexer, Parser

                lexer = Lexer(raw_response)
                tokens = lexer.tokenize()
                parser = Parser(tokens, source_text=raw_response)
                program = parser.parse()

                # Build alias → spec_name mapping from OUT{}
                alias_to_spec = {}
                if program.out_block:
                    for spec_name, refs in program.out_block.fields.items():
                        if isinstance(refs, list):
                            for alias in refs:
                                alias_to_spec[alias] = spec_name

                output = {}
                # Non-ActionCall values (lvars already resolved)
                for k, v in result.items():
                    if not isinstance(v, ActionCall):
                        spec = alias_to_spec.get(k, k)
                        output[spec] = v

                # ActionCall results mapped to spec names
                for (alias, ac), resp in zip(action_calls.items(), action_responses):
                    spec = alias_to_spec.get(alias, alias)
                    output[spec] = resp.output if hasattr(resp, "output") else resp

                # Assemble into operative response model (has action fields)
                output["action_requests"] = action_requests
                output["action_responses"] = action_responses

                # Only validate into operative response model if user provided
                # a real response_format (base_type). Otherwise the operative
                # response_type only has action fields and would drop the
                # LNDL-declared specs (q1, q2, etc).
                if operative and operative.base_type is not None:
                    try:
                        return operative.response_type.model_validate(output)
                    except Exception:
                        pass

                return output

    if _is_lndl and isinstance(result, str) and action_param:
        from lionagi.lndl import Lexer, Parser
        from lionagi.lndl._parse_function_call import parse_function_call
        from lionagi.lndl.types import ActionCall

        from ..act.act import act
        from ..fields import ActionRequestModel

        lexer = Lexer(result)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=result)
        program = parser.parse()

        out_aliases = set()
        if program.out_block:
            for v in program.out_block.fields.values():
                if isinstance(v, list):
                    out_aliases.update(v)

        # Build ordered list of (alias, lact_node, action_request)
        active_lacts = []
        action_requests = []
        for lact in program.lacts:
            if lact.alias not in out_aliases:
                continue
            try:
                parsed = parse_function_call(lact.call)
                req = ActionRequestModel(
                    function=parsed["operation"],
                    arguments=parsed["arguments"],
                )
                active_lacts.append(lact)
                action_requests.append(req)
            except ValueError:
                pass

        action_responses = None
        if action_requests:
            action_responses = await act(branch, action_requests, action_param)
            action_responses = [r for r in (action_responses or []) if r is not None]

        # Resolve: alias → result, keyed by alias
        alias_to_result = {}
        if action_responses:
            for lact_node, resp in zip(active_lacts, action_responses):
                alias_to_result[lact_node.alias] = (
                    resp.output if hasattr(resp, "output") else resp
                )

        # Build output using spec names from OUT{}, not aliases
        output = {}

        # Map OUT{spec: [aliases]} → resolve each alias to its value
        if program.out_block:
            for spec_name, refs in program.out_block.fields.items():
                if isinstance(refs, list):
                    for alias in refs:
                        if alias in alias_to_result:
                            output[spec_name] = alias_to_result[alias]
                        else:
                            # Check lvars
                            for lvar in program.lvars:
                                if lvar.alias == alias and lvar.alias in out_aliases:
                                    output[spec_name] = lvar.content
                                    break
                else:
                    output[spec_name] = refs

        if action_requests:
            output["action_requests"] = action_requests
        if action_responses:
            output["action_responses"] = action_responses

        # Only validate into operative response model if user provided
        # a real response_format (base_type). Otherwise operative.response_type
        # only has action fields and would drop LNDL-declared specs.
        if operative and operative.base_type is not None:
            try:
                return operative.response_type.model_validate(output)
            except Exception:
                pass

        return output

    # JSON path: look for action_requests on the parsed result
    if model_class:
        requests = getattr(result, "action_requests", None)
    elif isinstance(result, dict):
        requests = result.get("action_requests")
    else:
        requests = None

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

    if not model_class:
        # Dict response: merge action_responses in. Raw-text results stay
        # untouched (text has no structured slot for action_responses).
        if isinstance(result, dict):
            result["action_responses"] = action_response_models
        return result

    # If we have model_class, we must have operative (created at line 268)
    # First set the response_model to the existing result
    operative.response_model = result
    # Then update it with action_responses
    operative.update_response_model(data={"action_responses": action_response_models})
    return operative.response_model
