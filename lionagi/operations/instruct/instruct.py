# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
"""branch.instruct() — universal structured-output operation.

Routes automatically based on endpoint type:
  - CLI endpoints → stream via branch.run(), then branch.parse()
  - API endpoints → branch.operate() with field_models
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, JsonValue

from lionagi.ln.types import Spec
from lionagi.models import FieldModel
from lionagi.protocols.messages.assistant_response import AssistantResponse

from ..fields import Instruct
from ..types import RunParam

if TYPE_CHECKING:
    from lionagi.service.imodel import iModel
    from lionagi.session.branch import Branch

HandleValidation = Literal["raise", "return_value", "return_none"]


def prepare_instruct_kw(
    branch: Branch,
    *,
    instruct: Instruct | None = None,
    instruction: str | None = None,
    guidance: JsonValue = None,
    context: JsonValue = None,
    reason: bool = False,
    chat_model: iModel | None = None,
    field_models: list[FieldModel | Spec] | None = None,
    response_format: type[BaseModel] | None = None,
    skip_validation: bool = False,
    handle_validation: HandleValidation = "return_value",
    max_retries: int = 3,
    images: list | None = None,
    image_detail: str = "auto",
    stream_persist: bool = False,
    persist_dir: str | None = None,
    sender=None,
    recipient=None,
    **kwargs,
) -> dict:
    inst = Instruct.handle(instruct, instruction, guidance, context, reason)
    model = chat_model or branch.chat_model

    request_type = _resolve_response_type(
        field_models, response_format, inst.reason or False
    )

    run_param_kw = dict(
        guidance=inst.guidance,
        context=inst.context,
        sender=sender or branch.user or "user",
        recipient=recipient or branch.id,
        images=images,
        image_detail=image_detail,
        stream_persist=stream_persist,
    )
    if chat_model is not None:
        run_param_kw["imodel"] = chat_model
    if persist_dir is not None:
        run_param_kw["persist_dir"] = persist_dir
    if request_type:
        run_param_kw["response_format"] = request_type
    if kwargs:
        run_param_kw["imodel_kw"] = kwargs

    return {
        "instruction": inst.instruction,
        "run_param": RunParam(**run_param_kw),
        "request_type": request_type,
        "skip_validation": skip_validation,
        "handle_validation": handle_validation,
        "max_retries": max_retries,
        "is_cli": model.is_cli,
        "inst": inst,
    }


async def instruct(
    branch: Branch,
    instruction: str,
    run_param: RunParam,
    *,
    request_type: type[BaseModel] | None = None,
    skip_validation: bool = False,
    handle_validation: HandleValidation = "return_value",
    max_retries: int = 3,
    is_cli: bool = True,
    inst: Instruct | None = None,
) -> BaseModel | dict | str | None:
    if not is_cli:
        return await _instruct_api(
            branch, inst or Instruct(instruction=instruction), run_param
        )

    return await _instruct_cli(
        branch,
        instruction,
        run_param,
        request_type=request_type,
        skip_validation=skip_validation,
        handle_validation=handle_validation,
        max_retries=max_retries,
    )


async def _instruct_cli(
    branch: Branch,
    instruction: str,
    run_param: RunParam,
    *,
    request_type: type[BaseModel] | None = None,
    skip_validation: bool = False,
    handle_validation: HandleValidation = "return_value",
    max_retries: int = 3,
) -> BaseModel | dict | str | None:
    """CLI path: stream via run() → collect text → parse()."""
    from ..run.run import run as _run

    all_texts: list[str] = []
    async for msg in _run(branch, instruction, run_param):
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


async def _instruct_api(
    branch: Branch,
    inst: Instruct,
    run_param: RunParam,
) -> BaseModel | dict | str | None:
    """API path: delegate to branch.operate() for structured output."""
    return await branch.operate(
        instruct=inst,
        chat_model=run_param.imodel,
        response_format=run_param.response_format,
        images=run_param.images,
        image_detail=run_param.image_detail,
    )


def _resolve_response_type(
    field_models: list[FieldModel | Spec] | None,
    response_format: type[BaseModel] | None,
    reason: bool,
) -> type[BaseModel] | None:
    """Build the response type from field_models or response_format."""
    if response_format:
        return response_format

    if not field_models and not reason:
        return None

    from ..operate.step import Step

    fields_dict = {}
    for fm in field_models or []:
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
