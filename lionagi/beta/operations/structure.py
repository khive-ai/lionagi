# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Structure operation: generate -> parse -> validate pipeline.

Handler signature: structure(params, ctx) -> validated dict or model instance

Three-stage pipeline:
  1. generate: LLM call (TEXT or MESSAGE depending on persist)
  2. parse: extract JSON from text (direct + LLM reparse fallback)
  3. validate: enforce operable specs + cast to composed structure

When persist=True, the assistant message is stored in the branch
before parsing. The text is extracted from message.content.response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from lionagi.beta.rules import Validator
from lionagi.ln.types._sentinel import MaybeUnset, Unset, is_unset
from lionagi.ln.types import ModelConfig, Params

from .generate import GenerateParams
from .parse import ParseParams
from .utils import ReturnAs

if TYPE_CHECKING:
    from lionagi.beta.core.message.common import CustomParser
    from lionagi.beta.resource.imodel import iModel
    from lionagi.ln.types import Operable
    from lionagi.beta.session.context import RequestContext

__all__ = ("StructureParams", "structure")


@dataclass(frozen=True, slots=True)
class StructureParams(Params):
    """Parameters for generate -> parse -> validate pipeline; structure auto-composed from operable if None."""

    _config = ModelConfig(sentinel_additions=frozenset({"none", "empty"}))

    generate_params: GenerateParams
    operable: Operable

    validator: Validator = field(default_factory=Validator)
    structure: type[BaseModel] | None = None
    capabilities: set[str] | None = None
    auto_fix: bool = True
    strict: bool = True

    persist: bool = False

    parse_imodel: MaybeUnset[iModel | str] = Unset
    parse_imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    custom_parser: CustomParser | None = None
    similarity_threshold: float = 0.85
    max_retries: int = 3
    fill_mapping: dict[str, Any] | None = None
    fill_value: Any = Unset


async def structure(params: StructureParams, ctx: RequestContext) -> Any:
    structure_type = params.structure
    if structure_type is None:
        structure_type = params.operable.compose_structure()

    if params.persist:
        text = await _generate_and_persist(params.generate_params, ctx)
    else:
        gen_params = params.generate_params.with_updates(
            copy_containers="deep", return_as=ReturnAs.TEXT
        )
        text = await ctx.conduct("generate", gen_params)

    gen = params.generate_params
    parse_imodel = params.parse_imodel if not is_unset(params.parse_imodel) else gen.imodel
    parse_params = ParseParams(
        text=text,
        imodel=parse_imodel if not is_unset(parse_imodel) else None,
        imodel_kwargs=params.parse_imodel_kwargs or {},
        custom_parser=params.custom_parser,
        similarity_threshold=params.similarity_threshold,
        max_retries=params.max_retries,
        fill_mapping=params.fill_mapping,
        fill_value=params.fill_value,
        request_model=gen.request_model,
        tool_schemas=gen.tool_schemas,
        structure_format=gen.structure_format,
        custom_renderer=gen.custom_renderer,
        operable=params.operable,
    )
    parsed = await ctx.conduct("parse", parse_params)

    return await params.validator.validate(
        parsed,
        params.operable,
        capabilities=params.capabilities,
        auto_fix=params.auto_fix,
        strict=params.strict,
        structure=structure_type,
    )


async def _generate_and_persist(
    generate_params: GenerateParams,
    ctx: RequestContext,
) -> str:
    """Generate as MESSAGE and persist before returning text; ensures the assistant message exists in branch history before parse reads it."""
    gen_params = generate_params.with_updates(copy_containers="deep", return_as=ReturnAs.MESSAGE)
    message = await ctx.conduct("generate", gen_params)

    session = await ctx.get_session()
    branch = await ctx.get_branch()
    session.add_message(message, branches=branch)
    return message.content.response
