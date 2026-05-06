# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""LLM-assisted reparse: use a model to reformat malformed text.

Called as a fallback when direct parse (regex/fuzzy) fails.
For JSON: asks the model to extract structured data, then fuzzy-validates.
For LNDL: sends the parse error back, asks the model to fix LNDL syntax,
then runs the LNDL parser again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from lionagi.ln.fuzzy import HandleUnmatched, fuzzy_validate_mapping
from lionagi.ln.types._sentinel import MaybeUnset, Unset, is_sentinel
from lionagi.protocols.messages.instruction import InstructionContent as Instruction
from lionagi.protocols.messages.rendering import CustomParser, CustomRenderer

from .utils import ReturnAs

if TYPE_CHECKING:
    from typing import Any

    from lionagi.service.imodel_v2 import iModel
    from lionagi.session.session import Branch, Session

__all__ = ("_llm_reparse",)

JSON_REPARSE_PROMPT = "Reformat text into specified model or structure, using the provided schema format as a guide"

LNDL_REPARSE_PROMPT = (
    "Your previous LNDL output had a syntax error. Fix it and respond with ONLY valid LNDL.\n\n"
    "Parse error: {error}\n\n"
    "Original output:\n{original_text}\n\n"
    "Rules:\n"
    "- OUT{{}} spec names must exactly match the spec names from the task\n"
    "- OUT{{}} arrays must contain ONLY declared <lvar>/<lact> aliases\n"
    "- Every alias in OUT{{}} must have a <lvar> or <lact> declaration\n"
    "Respond with corrected LNDL only — no explanation."
)


async def _llm_reparse(
    session: Session,
    branch: Branch,
    text: str,
    imodel: iModel | str,
    tool_schemas: MaybeUnset[list[str]] = Unset,
    request_model: MaybeUnset[type[BaseModel]] = Unset,
    structure_format: MaybeUnset[Literal["json", "custom", "lndl"]] = Unset,
    custom_renderer: MaybeUnset[CustomRenderer] = Unset,
    custom_parser: CustomParser | None = None,
    fill_mapping: dict[str, Any] | None = None,
    operable: Any = None,
    parse_error: str | None = None,
    **imodel_kwargs: Any,
) -> dict[str, Any]:
    """Ask LLM to reformat text into structured output.

    For LNDL format: sends the parse error back and asks the model to fix
    the LNDL syntax, then re-parses with the LNDL parser.
    For JSON format: asks the model to extract structured data.
    """
    if (
        not is_sentinel(structure_format, additions={"none", "empty"})
        and structure_format == "lndl"
    ):
        return await _lndl_reparse(
            session=session,
            branch=branch,
            text=text,
            imodel=imodel,
            request_model=request_model,
            tool_schemas=tool_schemas,
            operable=operable,
            parse_error=parse_error,
            **imodel_kwargs,
        )

    instruction = Instruction.create(
        primary=JSON_REPARSE_PROMPT,
        context=[{"text_to_format": text}],
        request_model=request_model,
        tool_schemas=tool_schemas,
        structure_format=structure_format,
        custom_renderer=custom_renderer,
    )

    from .generate import _generate

    res = await _generate(
        session=session,
        branch=branch,
        instruction=instruction,
        imodel=imodel,
        return_as=ReturnAs.TEXT,
        **imodel_kwargs,
    )

    if is_sentinel(request_model, additions={"none", "empty"}):
        raise ValueError("request_model is required for LLM reparse")
    target_keys = list(request_model.model_fields.keys())

    if custom_parser is not None:
        return custom_parser(res, target_keys)

    # The LLM may have wrapped the JSON in prose or markdown fences.
    # Extract the first JSON object before fuzzy-matching keys.
    from lionagi.ln.fuzzy import extract_json

    extracted = extract_json(res, return_one_if_single=False)
    if not extracted or not isinstance(extracted[0], dict):
        raise ValueError(
            f"LLM reparse failed to produce a JSON object. Got: {str(res)[:200]}"
        )

    return fuzzy_validate_mapping(
        extracted[0],
        target_keys,
        handle_unmatched=HandleUnmatched.FORCE,
        fill_mapping=fill_mapping,
    )


async def _lndl_reparse(
    session: Session,
    branch: Branch,
    text: str,
    imodel: iModel | str,
    request_model: MaybeUnset[type[BaseModel]] = Unset,
    tool_schemas: MaybeUnset[list[str]] = Unset,
    operable: Any = None,
    parse_error: str | None = None,
    **imodel_kwargs: Any,
) -> dict[str, Any]:
    """LNDL-specific reparse: send error back, ask LLM to fix, re-parse."""
    error_msg = parse_error or "LNDL parsing failed"
    prompt = LNDL_REPARSE_PROMPT.format(error=error_msg, original_text=text[:2000])

    instruction = Instruction.create(
        primary=prompt,
        request_model=request_model,
        tool_schemas=tool_schemas,
        structure_format="lndl",
    )

    from .generate import _generate

    fixed_text = await _generate(
        session=session,
        branch=branch,
        instruction=instruction,
        imodel=imodel,
        return_as=ReturnAs.TEXT,
        **imodel_kwargs,
    )

    from lionagi.lndl import parse_lndl_fuzzy
    from lionagi.lndl.types import ActionCall, has_action_calls

    if operable is None:
        raise ValueError("operable is required for LNDL reparse")

    output = parse_lndl_fuzzy(fixed_text, operable)

    result = {}
    for name, value in output.fields.items():
        if isinstance(value, ActionCall) or (
            isinstance(value, BaseModel) and has_action_calls(value)
        ):
            result[name] = value
        elif isinstance(value, BaseModel):
            result[name] = value.model_dump()
        else:
            result[name] = value

    if output.actions:
        result["_lndl_actions"] = {
            action_name: {"function": ac.function, "arguments": ac.arguments}
            for action_name, ac in output.actions.items()
        }

    return result
