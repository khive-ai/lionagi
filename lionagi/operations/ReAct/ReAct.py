# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

from lionagi.fields.instruct import Instruct
from lionagi.libs.schema.as_readable import as_readable
from lionagi.libs.validate.common_field_validators import (
    validate_model_to_type,
)
from lionagi.ln.fuzzy import FuzzyMatchKeysParams
from lionagi.models import FieldModel, OperableModel
from lionagi.service.imodel import iModel

from ..types import (
    ActionContext,
    ChatContext,
    HandleValidation,
    InterpretContext,
    ParseContext,
)
from .utils import Analysis, ReActAnalysis

if TYPE_CHECKING:
    from lionagi.session.branch import Branch

B = TypeVar("B", bound=type[BaseModel])
logger = logging.getLogger(__name__)


async def ReAct(
    branch: "Branch",
    instruct: Instruct | dict[str, Any],
    interpret: bool = False,
    interpret_domain: str | None = None,
    interpret_style: str | None = None,
    interpret_sample: str | None = None,
    interpret_model: iModel | None = None,
    interpret_kwargs: dict | None = None,
    tools: Any = None,
    tool_schemas: Any = None,
    response_format: type[BaseModel] | BaseModel = None,
    intermediate_response_options: list[BaseModel] | BaseModel = None,
    intermediate_listable: bool = False,
    reasoning_effort: Literal["low", "medium", "high"] = None,
    extension_allowed: bool = True,
    max_extensions: int | None = 3,
    response_kwargs: dict | None = None,
    display_as: Literal["json", "yaml"] = "yaml",
    return_analysis: bool = False,
    analysis_model: iModel | None = None,
    verbose_analysis: bool = False,
    verbose_length: int = None,
    include_token_usage_to_model: bool = True,
    continue_after_failed_response: bool = False,
    **kwargs,
):
    """ReAct reasoning loop with legacy API - wrapper around ReAct_v1."""

    # Handle legacy verbose parameter
    if "verbose" in kwargs:
        verbose_analysis = kwargs.pop("verbose")

    # Convert Instruct to dict if needed
    instruct_dict = (
        instruct.to_dict()
        if isinstance(instruct, Instruct)
        else dict(instruct)
    )

    # Build InterpretContext if interpretation requested
    intp_ctx = None
    if interpret:
        intp_ctx = InterpretContext(
            domain=interpret_domain or "general",
            style=interpret_style or "concise",
            sample_writing=interpret_sample or "",
            imodel=interpret_model or analysis_model or branch.chat_model,
            imodel_kw=interpret_kwargs or {},
        )

    # Build ChatContext
    chat_ctx = ChatContext(
        guidance=instruct_dict.get("guidance"),
        context=instruct_dict.get("context"),
        sender=branch.user or "user",
        recipient=branch.id,
        response_format=None,  # Will be set in operate calls
        progression=None,
        tool_schemas=tool_schemas or [],
        images=[],
        image_detail="auto",
        plain_content="",
        include_token_usage_to_model=include_token_usage_to_model,
        imodel=analysis_model or branch.chat_model,
        imodel_kw=kwargs,
    )

    # Build ActionContext
    action_ctx = None
    if tools is not None or tool_schemas is not None:
        from ..act.act import _get_default_call_params

        action_ctx = ActionContext(
            action_call_params=_get_default_call_params(),
            tools=tools or True,
            strategy="concurrent",
            suppress_errors=True,
            verbose_action=False,
        )

    # Build ParseContext
    from ..parse.parse import get_default_call

    parse_ctx = ParseContext(
        response_format=ReActAnalysis,  # Initial format
        fuzzy_match_params=FuzzyMatchKeysParams(),
        handle_validation="return_value",
        alcall_params=get_default_call(),
        imodel=analysis_model or branch.chat_model,
        imodel_kw={},
    )

    # Response context for final answer
    resp_ctx = response_kwargs or {}
    if response_format:
        resp_ctx["response_format"] = response_format

    return await ReAct_v1(
        branch,
        instruction=instruct_dict.get("instruction", str(instruct)),
        chat_ctx=chat_ctx,
        action_ctx=action_ctx,
        parse_ctx=parse_ctx,
        intp_ctx=intp_ctx,
        resp_ctx=resp_ctx,
        reasoning_effort=reasoning_effort,
        reason=True,  # ReAct always uses reasoning
        field_models=None,
        handle_validation="return_value",
        invoke_actions=True,  # ReAct always invokes actions
        clear_messages=False,
        intermediate_response_options=intermediate_response_options,
        intermediate_listable=intermediate_listable,
        intermediate_nullable=False,
        max_extensions=max_extensions,
        extension_allowed=extension_allowed,
        verbose_analysis=verbose_analysis,
        display_as=display_as,
        verbose_length=verbose_length,
        continue_after_failed_response=continue_after_failed_response,
        return_analysis=return_analysis,
    )


async def ReAct_v1(
    branch: "Branch",
    instruction: str,
    chat_ctx: ChatContext,
    action_ctx: ActionContext | None = None,
    parse_ctx: ParseContext | None = None,
    intp_ctx: InterpretContext | None = None,
    resp_ctx: dict | None = None,
    reasoning_effort: Literal["low", "medium", "high"] | None = None,
    reason: bool = False,
    field_models: list[FieldModel] | None = None,
    handle_validation: HandleValidation = "raise",
    invoke_actions: bool = True,
    clear_messages=False,
    intermediate_response_options: B | list[B] = None,
    intermediate_listable: bool = False,
    intermediate_nullable: bool = False,
    max_extensions: int | None = 0,
    extension_allowed: bool = True,
    verbose_analysis: bool = False,
    display_as: Literal["yaml", "json"] = "yaml",
    verbose_length: int = None,
    continue_after_failed_response: bool = False,
    return_analysis: bool = False,
):
    """
    Context-based ReAct implementation - collects all outputs from ReActStream.

    Args:
        return_analysis: If True, returns list of all intermediate analyses.
                        If False, returns only the final result.
    """
    outs = []

    if verbose_analysis:
        async for i in ReActStream(
            branch=branch,
            instruction=instruction,
            chat_ctx=chat_ctx,
            action_ctx=action_ctx,
            parse_ctx=parse_ctx,
            intp_ctx=intp_ctx,
            resp_ctx=resp_ctx,
            reasoning_effort=reasoning_effort,
            reason=reason,
            field_models=field_models,
            handle_validation=handle_validation,
            invoke_actions=invoke_actions,
            clear_messages=clear_messages,
            intermediate_response_options=intermediate_response_options,
            intermediate_listable=intermediate_listable,
            intermediate_nullable=intermediate_nullable,
            max_extensions=max_extensions,
            extension_allowed=extension_allowed,
            verbose_analysis=verbose_analysis,
            display_as=display_as,
            verbose_length=verbose_length,
            continue_after_failed_response=continue_after_failed_response,
        ):
            analysis, str_ = i
            # str_ is already formatted markdown - just print it
            as_readable(
                str_,
                md=True,
                display_str=True,
            )
            outs.append(analysis)
    else:
        async for i in ReActStream(
            branch=branch,
            instruction=instruction,
            chat_ctx=chat_ctx,
            action_ctx=action_ctx,
            parse_ctx=parse_ctx,
            intp_ctx=intp_ctx,
            resp_ctx=resp_ctx,
            reasoning_effort=reasoning_effort,
            reason=reason,
            field_models=field_models,
            handle_validation=handle_validation,
            invoke_actions=invoke_actions,
            clear_messages=clear_messages,
            intermediate_response_options=intermediate_response_options,
            intermediate_listable=intermediate_listable,
            intermediate_nullable=intermediate_nullable,
            max_extensions=max_extensions,
            extension_allowed=extension_allowed,
            display_as=display_as,
            verbose_length=verbose_length,
            continue_after_failed_response=continue_after_failed_response,
        ):
            outs.append(i)

    if return_analysis:
        return outs

    # Extract answer from the final Analysis object
    final_result = outs[-1]
    if hasattr(final_result, "answer"):
        return final_result.answer
    return final_result


async def handle_instruction_interpretation(
    branch: "Branch",
    instruction: str,
    chat_ctx: ChatContext,
    intp_ctx: InterpretContext | None,
):
    """Handle instruction interpretation if requested."""
    if not intp_ctx:
        return instruction

    from ..interpret.interpret import interpret_v1

    return await interpret_v1(branch, instruction, intp_ctx)


def handle_field_models(
    field_models: list[FieldModel] | None,
    intermediate_response_options: B | list[B] = None,
    intermediate_listable: bool = False,
    intermediate_nullable: bool = False,
):
    """Build field models including intermediate response options."""
    fms = [] if not field_models else field_models

    if intermediate_response_options:

        def create_intermediate_response_field_model():
            _iro = intermediate_response_options
            iro = [_iro] if not isinstance(_iro, list) else _iro
            opm = OperableModel()

            for i in iro:
                type_ = validate_model_to_type(None, i)
                opm.add_field(
                    str(type_.__name__).lower(),
                    annotation=type_ | None,
                    # Remove lambda validator to avoid Pydantic serialization errors
                )

            m_ = opm.new_model(name="IntermediateResponseOptions")
            irfm = FieldModel(
                name="intermediate_response_options",
                base_type=m_,
                description="Intermediate deliverable outputs. fill as needed ",
                # Remove lambda validator to avoid Pydantic serialization errors
            )

            if intermediate_listable:
                irfm = irfm.as_listable()

            if intermediate_nullable:
                irfm = irfm.as_nullable()

            return irfm

        fms = [fms] if not isinstance(fms, list) else fms
        fms += [create_intermediate_response_field_model()]

    return fms


async def ReActStream(
    branch: "Branch",
    instruction: str,
    chat_ctx: ChatContext,
    action_ctx: ActionContext | None = None,
    parse_ctx: ParseContext | None = None,
    intp_ctx: InterpretContext | None = None,
    resp_ctx: dict | None = None,
    reasoning_effort: Literal["low", "medium", "high"] | None = None,
    reason: bool = False,
    field_models: list[FieldModel] | None = None,
    handle_validation: HandleValidation = "raise",
    invoke_actions: bool = True,
    clear_messages=False,
    intermediate_response_options: B | list[B] = None,
    intermediate_listable: bool = False,
    intermediate_nullable: bool = False,
    max_extensions: int | None = 0,
    extension_allowed: bool = True,
    verbose_analysis: bool = False,
    display_as: Literal["yaml", "json"] = "yaml",
    verbose_length: int = None,
    continue_after_failed_response: bool = False,
) -> AsyncGenerator:
    """Core ReAct streaming implementation with context-based architecture."""

    # Validate and clamp max_extensions
    if max_extensions and max_extensions > 100:
        logger.warning(
            "max_extensions should not exceed 100; defaulting to 100."
        )
        max_extensions = 100

    def verbose_yield(title, s_):
        if verbose_analysis:
            str_ = title + "\n"
            str_ += as_readable(
                s_,
                md=True,
                format_curly=True if display_as == "yaml" else False,
                max_chars=verbose_length,
            )
            return s_, str_
        else:
            return s_

    # Step 1: Interpret instruction if requested
    ins_str = await handle_instruction_interpretation(
        branch, instruction=instruction, chat_ctx=chat_ctx, intp_ctx=intp_ctx
    )
    # Print interpreted instruction if verbose (don't yield it - not an analysis object)
    if verbose_analysis and intp_ctx:
        str_ = "\n### Interpreted instruction:\n"
        str_ += as_readable(
            ins_str,
            md=True,
            format_curly=True if display_as == "yaml" else False,
            max_chars=verbose_length,
        )

    # Step 2: Handle field models
    fms = handle_field_models(
        field_models,
        intermediate_response_options,
        intermediate_listable,
        intermediate_nullable,
    )

    # Step 3: Initial ReAct analysis
    from ..operate.operate import operate_v1

    # Build context for initial analysis
    initial_chat_ctx = chat_ctx.with_updates(response_format=ReActAnalysis)

    initial_parse_ctx = (
        parse_ctx.with_updates(response_format=ReActAnalysis)
        if parse_ctx
        else None
    )

    # Add proper extension prompt for initial analysis
    initial_instruction = ins_str
    if extension_allowed and max_extensions:
        initial_instruction += "\n\n" + ReActAnalysis.FIRST_EXT_PROMPT.format(
            extensions=max_extensions
        )

    analysis = await operate_v1(
        branch,
        instruction=initial_instruction,
        chat_ctx=initial_chat_ctx,
        action_ctx=action_ctx,
        parse_ctx=initial_parse_ctx,
        handle_validation=handle_validation,
        invoke_actions=invoke_actions,
        skip_validation=False,
        clear_messages=clear_messages,
        reason=reason,
        field_models=fms,
    )

    out = verbose_yield("\n### ReAct Round No.1 Analysis:\n", analysis)
    yield out

    # Step 4: Extension loop
    extensions = max_extensions or 0
    round_count = 1

    def _need_extension(analysis):
        if hasattr(analysis, "extension_needed"):
            return analysis.extension_needed
        if isinstance(analysis, dict):
            return analysis.get("extension_needed", False)
        return False

    def _extension_allowed(exts):
        return extension_allowed and exts > 0

    def prepare_analysis_kwargs(exts):
        new_instruction = None
        if exts == max_extensions:
            new_instruction = ReActAnalysis.FIRST_EXT_PROMPT.format(
                extensions=exts
            )
        else:
            new_instruction = ReActAnalysis.CONTINUE_EXT_PROMPT.format(
                extensions=exts
            )

        # Use with_updates to create new context instances
        updates = {"response_format": ReActAnalysis}

        if reasoning_effort:
            guide = {
                "low": "Quick concise reasoning.\n",
                "medium": "Reasonably balanced reasoning.\n",
                "high": "Thorough, try as hard as you can in reasoning.\n",
            }.get(reasoning_effort, "")

            updates["guidance"] = (guide or "") + (chat_ctx.guidance or "")
            updates["imodel_kw"] = {
                **(chat_ctx.imodel_kw or {}),
                "reasoning_effort": reasoning_effort,
            }

        _cctx = chat_ctx.with_updates(**updates)

        # Import default call params if needed
        from ..act.act import _get_default_call_params

        _actx = (
            action_ctx.with_updates(
                strategy=getattr(analysis, "action_strategy", "concurrent")
            )
            if action_ctx
            else ActionContext(
                action_call_params=_get_default_call_params(),
                tools=True,
                strategy=getattr(analysis, "action_strategy", "concurrent"),
                suppress_errors=True,
                verbose_action=False,
            )
        )

        return {
            "instruction": new_instruction,
            "chat_ctx": _cctx,
            "action_ctx": _actx,
            "reason": reason,
            "field_models": fms,
        }

    while _extension_allowed(extensions) and _need_extension(analysis):
        kwargs = prepare_analysis_kwargs(extensions)

        # Build parse context for extension
        ext_parse_ctx = (
            parse_ctx.with_updates(
                response_format=kwargs["chat_ctx"].response_format
            )
            if parse_ctx
            else None
        )

        analysis = await operate_v1(
            branch,
            instruction=kwargs["instruction"],
            chat_ctx=kwargs["chat_ctx"],
            action_ctx=kwargs.get("action_ctx"),
            parse_ctx=ext_parse_ctx,
            handle_validation=handle_validation,
            invoke_actions=invoke_actions,
            skip_validation=False,
            clear_messages=False,  # Keep messages to maintain context
            reason=kwargs.get("reason", True),
            field_models=kwargs.get("field_models"),
        )
        round_count += 1

        if isinstance(analysis, dict) and all(
            i is None for i in analysis.values()
        ):
            if not continue_after_failed_response:
                raise ValueError(
                    "All values in the response are None. "
                    "This might be due to a failed response. "
                    "Set `continue_after_failed_response=True` to ignore this error."
                )

        out = verbose_yield(
            f"\n### ReAct Round No.{round_count} Analysis:\n", analysis
        )
        yield out

        if extensions:
            extensions -= 1

    # Step 5: Final answer
    answer_prompt = ReActAnalysis.ANSWER_PROMPT.format(instruction=ins_str)

    final_response_format = (
        resp_ctx.get("response_format") if resp_ctx else None
    )
    if not final_response_format:
        final_response_format = Analysis

    # Build contexts for final answer
    resp_ctx_updates = {"response_format": final_response_format}
    if resp_ctx:
        # Merge resp_ctx into updates (filter allowed keys)
        for k, v in resp_ctx.items():
            if k in chat_ctx.allowed() and k != "response_format":
                resp_ctx_updates[k] = v

    final_chat_ctx = chat_ctx.with_updates(**resp_ctx_updates)

    final_parse_ctx = (
        parse_ctx.with_updates(response_format=final_response_format)
        if parse_ctx
        else None
    )

    try:
        out = await operate_v1(
            branch,
            instruction=answer_prompt,
            chat_ctx=final_chat_ctx,
            action_ctx=None,  # No actions in final answer
            parse_ctx=final_parse_ctx,
            handle_validation=handle_validation,
            invoke_actions=False,
            skip_validation=False,
            clear_messages=False,
            reason=False,  # No reasoning wrapper in final answer
            field_models=None,
        )

        if isinstance(out, dict) and all(i is None for i in out.values()):
            if not continue_after_failed_response:
                raise ValueError(
                    "All values in the response are None. "
                    "This might be due to a failed response. "
                    "Set `continue_after_failed_response=True` to ignore this error."
                )
    except Exception:
        out = branch.msgs.last_response.response

    # Don't extract .answer - return the full Analysis object
    _o = verbose_yield("\n### ReAct Final Answer:\n", out)
    yield _o
