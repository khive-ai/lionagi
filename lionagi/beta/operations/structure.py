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
    """Parameters for structure operation (generate -> parse -> validate).

    Attributes:
        generate_params: LLM generation config.
        validator: Rule-based validator for operable spec enforcement.
        operable: Spec definition for field validation (required).
        structure: Pre-composed Pydantic model to cast into.
            If None, auto-composed from operable via compose_structure().
        persist: Persist assistant message to branch before parsing.
        capabilities: Allowed field subset (None = all operable fields).
        auto_fix: Auto-coerce validation issues (e.g., wrap scalar -> list).
        strict: Raise on validation failure vs. skip.
    """

    _config = ModelConfig(sentinel_additions=frozenset({"none", "empty"}))

    # Generate stage
    generate_params: GenerateParams
    operable: Operable

    # Validate stage
    validator: Validator = field(default_factory=Validator)
    structure: type[BaseModel] | None = None
    capabilities: set[str] | None = None
    auto_fix: bool = True
    strict: bool = True

    # Message persistence
    persist: bool = False

    # Parse stage overrides
    parse_imodel: MaybeUnset[iModel | str] = Unset
    parse_imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    custom_parser: CustomParser | None = None
    similarity_threshold: float = 0.85
    max_retries: int = 3
    fill_mapping: dict[str, Any] | None = None
    fill_value: Any = Unset


async def structure(params: StructureParams, ctx: RequestContext) -> Any:
    """Structure operation handler: generate -> parse -> validate.

    When persist=True, generates as MESSAGE, persists to branch,
    then extracts text from message.content.response for parsing.
    """
    # Resolve structure type: explicit or auto-composed from operable
    structure_type = params.structure
    if structure_type is None:
        structure_type = params.operable.compose_structure()

    # Stage 1: Generate
    if params.persist:
        text = await _generate_and_persist(params.generate_params, ctx)
    else:
        gen_params = params.generate_params.with_updates(
            copy_containers="deep", return_as=ReturnAs.TEXT
        )
        text = await ctx.conduct("generate", gen_params)

    # Stage 2: Parse (inherit schema config from generate params)
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

    # Stage 3: Validate against operable specs + structure type
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
    """Generate as MESSAGE, persist to branch, return text.

    The assistant message is added to both session messages and
    the branch progression for conversation continuity.
    """
    gen_params = generate_params.with_updates(copy_containers="deep", return_as=ReturnAs.MESSAGE)
    message = await ctx.conduct("generate", gen_params)

    # Persist to branch
    session = await ctx.get_session()
    branch = await ctx.get_branch()
    session.add_message(message, branches=branch)

    # Extract text from assistant content
    return message.content.response
