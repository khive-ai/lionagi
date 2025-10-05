# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import AsyncGenerator
from copy import copy as shallow_copy
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
    instruct: Instruct | dict[str, Any] = None,
    # Modern API: pass contexts directly
    chat_ctx: ChatContext = None,
    action_ctx: ActionContext | None = None,
    parse_ctx: ParseContext | None = None,
    intp_ctx: InterpretContext | None = None,
    resp_ctx: dict | None = None,
    # Legacy API: individual parameters (backward compatible)
    instruction: str = None,
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
    # ReAct-specific parameters
    reason: bool = True,
    field_models: list[FieldModel] | None = None,
    handle_validation: HandleValidation = "return_value",
    invoke_actions: bool = True,
    clear_messages: bool = False,
    intermediate_nullable: bool = False,
    **kwargs,
):
    """
    ReAct reasoning loop with advanced multi-step reasoning and tool integration.

    Two usage patterns:

    1. Modern (recommended):
        chat_ctx = ChatContext(...)
        action_ctx = ActionContext(...)
        result = await ReAct(branch, instruct, chat_ctx=chat_ctx, action_ctx=action_ctx, ...)

    2. Legacy (backward compatible):
        result = await ReAct(branch, instruct, tools=[...], max_extensions=5, ...)

    Args:
        branch: Branch instance for execution
        instruct: Instruct object or dict with instruction/guidance/context
        chat_ctx: ChatContext object (modern API)
        action_ctx: ActionContext object (modern API)
        parse_ctx: ParseContext object (modern API)
        intp_ctx: InterpretContext object (modern API)
        resp_ctx: Response context dict (modern API)
        instruction: Raw instruction string (legacy, overrides instruct)
        interpret: Enable instruction interpretation (legacy)
        interpret_domain: Interpretation domain (legacy)
        interpret_style: Interpretation style (legacy)
        interpret_sample: Sample writing style (legacy)
        interpret_model: Model for interpretation (legacy)
        interpret_kwargs: Interpretation kwargs (legacy)
        tools: Tools to use (legacy)
        tool_schemas: Tool schemas (legacy)
        response_format: Final response format (legacy)
        intermediate_response_options: Intermediate response options (legacy)
        intermediate_listable: Make intermediate responses listable (legacy)
        reasoning_effort: Reasoning effort level (legacy)
        extension_allowed: Allow extensions (legacy)
        max_extensions: Maximum extension rounds (legacy)
        response_kwargs: Response kwargs (legacy)
        display_as: Display format (legacy)
        return_analysis: Return all analyses (legacy)
        analysis_model: Model for analysis (legacy)
        verbose_analysis: Verbose output (legacy)
        verbose_length: Verbose output length (legacy)
        include_token_usage_to_model: Include token usage (legacy)
        continue_after_failed_response: Continue on failures (legacy)
        reason: Enable reasoning (default: True)
        field_models: Additional field models
        handle_validation: Validation handling strategy
        invoke_actions: Invoke action requests
        clear_messages: Clear message history
        intermediate_nullable: Make intermediate responses nullable
        **kwargs: Additional model parameters

    Returns:
        Final result or list of analyses if return_analysis=True
    """
    # Build contexts from whichever input was provided
    if chat_ctx is None:
        # Convert Instruct to dict if needed
        if instruction is not None:
            # Explicit instruction overrides instruct
            instruct_dict = {"instruction": instruction}
        elif isinstance(instruct, Instruct):
            instruct_dict = instruct.to_dict()
        elif isinstance(instruct, dict):
            instruct_dict = dict(instruct)
        else:
            instruct_dict = {}

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

        # Extract instruction for execution
        instruction = instruct_dict.get(
            "instruction", str(instruct) if instruct else ""
        )

    if intp_ctx is None and interpret:
        intp_ctx = InterpretContext(
            domain=interpret_domain or "general",
            style=interpret_style or "concise",
            sample_writing=interpret_sample or "",
            imodel=interpret_model or analysis_model or branch.chat_model,
            imodel_kw=interpret_kwargs or {},
        )

    if action_ctx is None and (tools is not None or tool_schemas is not None):
        from ..act.act import ActionExecutor

        action_ctx = ActionContext(
            action_call_params=ActionExecutor.DEFAULT_ALCALL_PARAMS,
            tools=tools or True,
            strategy="concurrent",
            suppress_errors=True,
            verbose_action=False,
        )

    if parse_ctx is None:
        from ..parse.parse import ParseExecutor

        parse_ctx = ParseContext(
            response_format=ReActAnalysis,  # Initial format
            fuzzy_match_params=FuzzyMatchKeysParams(),
            handle_validation="return_value",
            alcall_params=ParseExecutor.DEFAULT_ALCALL_PARAMS,
            imodel=analysis_model or branch.chat_model,
            imodel_kw={},
        )

    if resp_ctx is None:
        resp_ctx = response_kwargs or {}
        if response_format:
            resp_ctx["response_format"] = response_format

    # Use provided instruction or extract from chat_ctx
    if instruction is None:
        # Instruction should come from earlier processing or be in kwargs
        instruction = kwargs.get("instruction", "")

    # Execute ReAct stream
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
            str_ += "\n---------\n"
            as_readable(str_, md=True, display_str=True)
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
            verbose_analysis=verbose_analysis,
            display_as=display_as,
            verbose_length=verbose_length,
            continue_after_failed_response=continue_after_failed_response,
        ):
            outs.append(i)

    if return_analysis:
        return outs
    return outs[-1]


async def handle_instruction_interpretation(
    branch: "Branch",
    instruction: str,
    chat_ctx: ChatContext,
    intp_ctx: InterpretContext | None,
):
    """Handle instruction interpretation if requested."""
    if not intp_ctx:
        return instruction

    from ..interpret.interpret import interpret

    return await interpret(branch, instruction, intp_ctx=intp_ctx)


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
    # Only yield interpreted instruction if verbose or if interpretation was actually done
    if verbose_analysis or intp_ctx:
        out = verbose_yield("\n### Interpreted instruction:\n", ins_str)
        yield out

    # Step 2: Handle field models
    fms = handle_field_models(
        field_models,
        intermediate_response_options,
        intermediate_listable,
        intermediate_nullable,
    )

    # Step 3: Initial ReAct analysis
    analysis = await branch.operate(
        instruction=ins_str
        + f"\nIf needed, you can do up to {max_extensions or 0 if extension_allowed else 0} expansions.",
        response_format=ReActAnalysis,
        tools=action_ctx.tools if action_ctx else True,
        actions=True,
        reason=reason,
        field_models=fms,
        handle_validation=handle_validation,
        invoke_actions=invoke_actions,
        skip_validation=False,
        clear_messages=clear_messages,
        chat_model=chat_ctx.imodel,
        **chat_ctx.imodel_kw,
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

        # Shallow copy contexts to avoid mutation (deep copy fails with unpicklable iModel)
        _cctx = shallow_copy(chat_ctx)
        _cctx.response_format = ReActAnalysis

        _actx = (
            shallow_copy(action_ctx)
            if action_ctx
            else ActionContext(
                action_call_params=None,
                tools=True,
                strategy="concurrent",
                suppress_errors=True,
                verbose_action=False,
            )
        )
        _actx.strategy = getattr(analysis, "action_strategy", "concurrent")

        if reasoning_effort:
            guide = {
                "low": "Quick concise reasoning.\n",
                "medium": "Reasonably balanced reasoning.\n",
                "high": "Thorough, try as hard as you can in reasoning.\n",
            }.get(reasoning_effort, "")

            _cctx.guidance = (guide or "") + (_cctx.guidance or "")
            if _cctx.imodel_kw is None:
                _cctx.imodel_kw = {}
            _cctx.imodel_kw["reasoning_effort"] = reasoning_effort

        return {
            "instruction": new_instruction,
            "chat_ctx": _cctx,
            "action_ctx": _actx,
            "reason": reason,
            "field_models": fms,
        }

    while _extension_allowed(extensions) and _need_extension(analysis):
        kwargs = prepare_analysis_kwargs(extensions)
        # Convert contexts back to legacy parameters
        analysis = await branch.operate(
            instruction=kwargs["instruction"],
            response_format=kwargs["chat_ctx"].response_format,
            tools=(
                kwargs["action_ctx"].tools
                if kwargs.get("action_ctx")
                else True
            ),
            actions=True,
            reason=kwargs.get("reason", True),
            field_models=kwargs.get("field_models"),
            handle_validation=handle_validation,
            invoke_actions=invoke_actions,
            skip_validation=False,
            chat_model=kwargs["chat_ctx"].imodel,
            **kwargs["chat_ctx"].imodel_kw,
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

    try:
        out = await branch.operate(
            instruction=answer_prompt,
            response_format=final_response_format,
            chat_model=chat_ctx.imodel,
            **(resp_ctx or {}),
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
