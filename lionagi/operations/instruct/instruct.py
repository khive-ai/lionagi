# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
"""branch.instruct() — CLI-endpoint operate.

Uses branch.run() (CLI streaming) to execute, then branch.parse()
for structured extraction. Two-phase: stream → parse.

    result = await branch.instruct(
        instruction="Analyze auth middleware",
        field_models=[AGENT_REQUEST_FIELDS],
        reason=True,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, JsonValue

from lionagi.ln.types import Spec
from lionagi.models import FieldModel
from lionagi.protocols.messages.assistant_response import AssistantResponse

from ..fields import Instruct

if TYPE_CHECKING:
    from lionagi.service.imodel import iModel
    from lionagi.session.branch import Branch


async def instruct(
    branch: Branch,
    instruction: str | Instruct | None = None,
    *,
    guidance: JsonValue = None,
    context: JsonValue = None,
    chat_model: iModel | None = None,
    field_models: list[FieldModel | Spec] | None = None,
    response_format: type[BaseModel] | None = None,
    reason: bool = False,
    skip_validation: bool = False,
    handle_validation: Literal["raise", "return_value", "return_none"] = "return_value",
    max_retries: int = 3,
    images: list | None = None,
    image_detail: str = "auto",
    **kwargs,
) -> BaseModel | dict | str | None:
    """Run instruction via CLI endpoint, extract structured output.

    1. Resolve response type from field_models/response_format
    2. Stream via branch.run() with response_format passed to endpoint
    3. Collect all AssistantResponse texts
    4. Parse via branch.parse() if structured output requested
    """
    if isinstance(instruction, Instruct):
        inst = instruction
    elif isinstance(instruction, dict):
        inst = Instruct(**instruction)
    else:
        inst = Instruct(
            instruction=instruction,
            guidance=guidance,
            context=context,
        )

    if guidance and not inst.guidance:
        inst.guidance = guidance
    if context and not inst.context:
        inst.context = context
    if reason:
        inst.reason = True

    request_type = _resolve_response_type(
        field_models, response_format, inst.reason or False
    )

    # Stream via run — pass response_format to CLI endpoint
    run_kwargs = dict(kwargs)
    if request_type:
        run_kwargs["response_format"] = request_type

    all_texts: list[str] = []
    async for msg in branch.run(
        instruction=inst.instruction,
        chat_model=chat_model,
        guidance=inst.guidance,
        context=inst.context,
        images=images,
        image_detail=image_detail,
        **run_kwargs,
    ):
        if isinstance(msg, AssistantResponse):
            text = msg.response or ""
            if text:
                all_texts.append(text)

    if not all_texts:
        return None

    full_text = "\n\n".join(all_texts)

    if not request_type or skip_validation:
        return full_text

    return await branch.parse(
        full_text,
        response_format=request_type,
        handle_validation=handle_validation,
        max_retries=max_retries,
    )


def _resolve_response_type(
    field_models: list[FieldModel | Spec] | None,
    response_format: type[BaseModel] | None,
    reason: bool,
) -> type[BaseModel] | None:
    """Build the response type from field_models or response_format."""
    if response_format:
        return response_format

    if not field_models:
        return None

    from ..operate.step import Step

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

    operative = Step.request_operative(
        base_type=None,
        reason=reason,
        actions=False,
        fields=fields_dict,
    )
    operative = Step.respond_operative(operative)
    return operative.response_type if operative else None
