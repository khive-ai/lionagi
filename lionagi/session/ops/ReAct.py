import logging
from copy import deepcopy
from typing import Literal, TypeVar

from pydantic import BaseModel, JsonValue

from lionagi.fields.analysis import Analysis
from lionagi.libs.schema.as_readable import as_readable
from lionagi.libs.validate.common_field_validators import (
    validate_model_to_type,
)
from lionagi.models import FieldModel, OperableModel
from lionagi.protocols.types import Instruction

from .types import (
    ActionContext,
    ChatContext,
    HandleValidation,
    InterpretContext,
    ParseContext,
)

B = TypeVar("B", bound=type[BaseModel])


logger = logging.getLogger(__name__)


async def ReAct(
    branch,
    instruction: JsonValue | Instruction,
    chat_ctx: ChatContext,
    action_ctx: ActionContext | None = None,
    parse_ctx: ParseContext | None = None,
    intp_ctx: InterpretContext | bool = None,
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
    Convenience wrapper around ReActStream that collects all outputs.

    Args:
        return_analysis: If True, returns list of all intermediate analyses.
                        If False, returns only the final result.
        (other args same as ReActStream)

    Returns:
        List of all outputs if return_analysis=True, otherwise final output only.
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
    branch, instruction, chat_ctx, intp_ctx
):
    ins_str = (
        instruction.instruction
        if isinstance(instruction, Instruction)
        else str(instruction)
    )
    if not intp_ctx:
        return ins_str

    _ictx = intp_ctx or InterpretContext(
        domain="general",
        style="concise",
        sample_writing="",
        imodel=chat_ctx.imodel,
        imodel_kw=chat_ctx.imodel_kw,
    )
    from .interpret import interpret

    return await interpret(branch, ins_str, _ictx)


def handle_field_models(
    field_models: list[FieldModel] | None,
    intermediate_response_options: B | list[B] = None,
    intermediate_listable: bool = False,
    intermediate_nullable: bool = False,
):
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
                    validator=lambda cls, x: None if x == {} else x,
                )
            m_ = opm.new_model(name="IntermediateResponseOptions")
            irfm = FieldModel(
                name="intermediate_response_options",
                base_type=m_,
                description="Intermediate deliverable outputs. fill as needed ",
                validator=lambda cls, x: None if not x else x,
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
    branch,
    instruction: JsonValue | Instruction,
    chat_ctx: ChatContext,
    action_ctx: ActionContext | None = None,
    parse_ctx: ParseContext | None = None,
    intp_ctx: InterpretContext | bool = None,
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
):

    # Validate and clamp max_extensions if needed
    if max_extensions and max_extensions > 100:
        logging.warning(
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

    # --------- round 1 analysis ---------
    ins_str = await handle_instruction_interpretation(
        branch, instruction=instruction, chat_ctx=chat_ctx, intp_ctx=intp_ctx
    )
    out = verbose_yield("\n### Interpreted instruction:\n", ins_str)
    yield out

    fms = handle_field_models(
        field_models,
        intermediate_response_options,
        intermediate_listable,
        intermediate_nullable,
    )

    from .operate import operate

    analysis = await operate(
        branch,
        instruction=ins_str
        + f"\nIf needed, you can do up to {max_extensions or 0 if extension_allowed else 0} expansions.",
        chat_ctx=chat_ctx,
        action_ctx=action_ctx,
        parse_ctx=parse_ctx,
        reason=reason,
        field_models=fms,
        handle_validation=handle_validation,
        invoke_actions=invoke_actions,
        clear_messages=clear_messages,
    )

    out = verbose_yield("\n### ReAct Round No.1 Analysis:\n", analysis)
    yield out

    # --------- possibly loop through expansions if extension_needed ---------
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

    from lionagi.fields.analysis import ReActAnalysis

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

        # Deep copy contexts to avoid mutating originals (dataclasses, not Pydantic)
        _cctx = deepcopy(chat_ctx)
        _cctx.response_format = ReActAnalysis

        _actx = deepcopy(action_ctx) if action_ctx else ActionContext()
        _actx.tools = _actx.tools or True
        _actx.strategy = analysis.action_strategy or "concurrent"

        if reasoning_effort:
            guide = None
            if reasoning_effort == "low":
                guide = "Quick concise reasoning.\n"
            if reasoning_effort == "medium":
                guide = "Reasonably balanced reasoning.\n"
            if reasoning_effort == "high":
                guide = "Thorough, try as hard as you can in reasoning.\n"

            _cctx.guidance = (guide or "") + (_cctx.guidance or "")
            if _cctx.imodel_kw is None:
                _cctx.imodel_kw = {}
            _cctx.imodel_kw["reasoning_effort"] = reasoning_effort

        return {
            "instruction": new_instruction,
            "reason": reason,
            "field_models": fms,
            "chat_ctx": _cctx,
            "action_ctx": _actx,
        }

    while _extension_allowed(extensions) and _need_extension(analysis):
        analysis = await branch.operate(**prepare_analysis_kwargs(extensions))
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

    # Step 3: Produce final answer by calling branch.instruct with an answer prompt
    answer_prompt = ReActAnalysis.ANSWER_PROMPT.format(instruction=ins_str)

    # Get response format from resp_ctx or use default Analysis
    final_response_format = (
        resp_ctx.get("response_format") if resp_ctx else None
    )
    if not final_response_format:
        final_response_format = Analysis

    try:
        from .operate import operate

        out = await operate(
            branch,
            instruction=answer_prompt,
            chat_ctx=ChatContext(
                **{
                    **chat_ctx.to_dict(),
                    "response_format": final_response_format,
                }
            ),
            response_format=final_response_format,
            **(resp_ctx or {}),
        )
        if isinstance(analysis, dict) and all(
            i is None for i in analysis.values()
        ):
            if not continue_after_failed_response:
                raise ValueError(
                    "All values in the response are None. "
                    "This might be due to a failed response. "
                    "Set `continue_after_failed_response=True` to ignore this error."
                )
    except Exception:
        out = branch.msgs.last_response.response

    if isinstance(out, Analysis):
        out = out.answer

    _o = verbose_yield("\n### ReAct Final Answer:\n", out)
    yield _o
