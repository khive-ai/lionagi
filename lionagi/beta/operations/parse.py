# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Parse operation: extract structured JSON from raw LLM text.

Handler signature: parse(params, ctx) → dict[str, Any]

Two-stage pipeline:
  1. _direct_parse: regex/fuzzy extraction (fast, no LLM call)
  2. _llm_reparse: LLM-assisted fallback (up to max_retries)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from lionagi.beta.core.message.common import CustomParser, CustomRenderer, StructureFormat
from lionagi._errors import ConfigurationError, ExecutionError, LionError, ValidationError
from lionagi.ln.types._sentinel import MaybeUnset, Unset, is_sentinel
from lionagi.ln.types import ModelConfig, Params
from lionagi.ln.fuzzy import HandleUnmatched, extract_json, fuzzy_validate_mapping

if TYPE_CHECKING:
    from typing import Any

    from lionagi.beta.resource.imodel import iModel
    from lionagi.beta.session.context import RequestContext

    Branch = Any  # Branch not yet fully migrated
    Session = Any  # Session not yet migrated to beta

__all__ = ("ParseParams", "parse")


@dataclass(frozen=True, slots=True)
class ParseParams(Params):
    """Parse operation parameters; max_retries=0 disables LLM reparse fallback."""

    _config = ModelConfig(sentinel_additions=frozenset({"none", "empty"}))

    text: str
    target_keys: MaybeUnset[list[str]] = Unset
    imodel: iModel | str | None = None
    imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    custom_parser: CustomParser | None = None
    custom_renderer: MaybeUnset[CustomRenderer] = Unset
    structure_format: StructureFormat = StructureFormat.JSON
    tool_schemas: MaybeUnset[list[str]] = Unset
    request_model: MaybeUnset[type[BaseModel]] = Unset
    similarity_threshold: float = 0.85
    handle_unmatched: HandleUnmatched = HandleUnmatched.FORCE
    max_retries: int = 3
    fill_mapping: dict[str, Any] | None = None
    fill_value: Any = Unset
    operable: Any = None
    scratchpad: Any = None


async def parse(params: ParseParams, ctx: RequestContext) -> dict[str, Any]:
    target_keys = params.target_keys

    if params.is_sentinel_field("target_keys"):
        if params.is_sentinel_field("request_model"):
            raise ValidationError(
                "Either 'target_keys' or 'request_model' must be provided for parse"
            )
        target_keys = list(params.request_model.model_fields.keys())

    session = await ctx.get_session()
    branch = session.get_branch(ctx.branch)

    data = params.to_dict(exclude={"target_keys", "imodel_kwargs", "scratchpad"})

    return await _parse(
        session=session,
        branch=branch,
        target_keys=target_keys,
        scratchpad=branch.scratchpad if params.scratchpad is None else params.scratchpad,
        **data,
        **params.imodel_kwargs,
    )


async def _parse(
    session: Session,
    branch: Branch | str,
    text: str,
    target_keys: list[str],
    structure_format: StructureFormat = StructureFormat.JSON,
    custom_parser: CustomParser | None = None,
    similarity_threshold: float = 0.85,
    handle_unmatched: HandleUnmatched = HandleUnmatched.FORCE,
    fill_mapping: dict[str, Any] | None = None,
    fill_value: Any = Unset,
    max_retries: MaybeUnset[int] = Unset,
    imodel: iModel | str | None = None,
    tool_schemas: MaybeUnset[list[str]] = Unset,
    request_model: MaybeUnset[type[BaseModel]] = Unset,
    custom_renderer: MaybeUnset[CustomRenderer] = Unset,
    operable: Any = None,
    scratchpad: Any = None,
    **imodel_kwargs: Any,
) -> dict[str, Any]:
    _sentinel_check = {"none", "empty"}
    if is_sentinel(target_keys, _sentinel_check):
        raise ValidationError("No target_keys provided for parse operation")
    if is_sentinel(text, _sentinel_check):
        raise ValidationError("No text provided for parse operation")

    direct_error: Exception | None = None
    try:
        return _direct_parse(
            text=text,
            target_keys=target_keys,
            structure_format=structure_format,
            custom_parser=custom_parser,
            similarity_threshold=similarity_threshold,
            handle_unmatched=handle_unmatched,
            fill_mapping=fill_mapping,
            fill_value=fill_value,
            operable=operable,
            scratchpad=scratchpad,
        )
    except LionError as e:
        if e.retryable is False:
            raise
        direct_error = e
    except Exception as e:
        from lionagi.beta.lndl.errors import MissingOutBlockError

        if isinstance(e, MissingOutBlockError):
            # Propagate — operate catches this to trigger LNDL continuation.
            raise
        direct_error = e

    if is_sentinel(max_retries, _sentinel_check) or max_retries < 1:
        raise ExecutionError(
            "Direct parse failed and max_retries not enabled, no reparse attempted",
            retryable=False,
            cause=direct_error,
        )

    from .llm_reparse import _llm_reparse

    parse_error_msg = str(direct_error) if direct_error else None

    for _ in range(max_retries):
        try:
            return await _llm_reparse(
                session=session,
                branch=branch,
                text=text,
                imodel=imodel,
                tool_schemas=tool_schemas,
                request_model=request_model,
                structure_format=structure_format,
                custom_renderer=custom_renderer,
                custom_parser=custom_parser,
                operable=operable,
                parse_error=parse_error_msg,
                **imodel_kwargs,
            )
        except LionError as e:
            if e.retryable is False:
                raise

    raise ExecutionError(
        "All parse attempts (direct and LLM reparse) failed",
        retryable=False,
    )


def _direct_parse(
    text: str,
    target_keys: list[str],
    structure_format: StructureFormat = StructureFormat.JSON,
    custom_parser: CustomParser | None = None,
    similarity_threshold: float = 0.85,
    handle_unmatched: HandleUnmatched = HandleUnmatched.FORCE,
    fill_mapping: dict[str, Any] | None = None,
    fill_value: Any = Unset,
    operable: Any = None,
    scratchpad: Any = None,
) -> dict[str, Any]:
    """Extract structured data from text without LLM assistance.

    Routes to LNDL parser, custom_parser, or built-in JSON extraction.
    When scratchpad is provided and format is LNDL, prior-round lvars
    are merged into resolution so OUT{} can reference earlier declarations.
    """
    _sentinel_check = {"none", "empty"}
    if is_sentinel(target_keys, _sentinel_check):
        raise ValidationError("No target_keys provided for direct_parse operation")

    match structure_format:
        case StructureFormat.LNDL:
            if operable is None:
                raise ConfigurationError(
                    "structure_format='lndl' requires operable to be provided",
                    retryable=False,
                )
            try:
                from lionagi.beta.lndl import parse_lndl_fuzzy
                from lionagi.beta.lndl.errors import MissingOutBlockError

                output = parse_lndl_fuzzy(
                    text, operable, threshold=similarity_threshold, scratchpad=scratchpad
                )

                result = {}
                for name, value in output.fields.items():
                    from lionagi.beta.lndl.types import ActionCall, has_action_calls

                    if isinstance(value, ActionCall) or (
                        isinstance(value, BaseModel) and has_action_calls(value)
                    ):
                        result[name] = value
                    elif isinstance(value, BaseModel):
                        result[name] = value.model_dump()
                    else:
                        result[name] = value

                return result
            except MissingOutBlockError:
                raise
            except Exception as e:
                raise ExecutionError(
                    "LNDL parser failed to extract data from text",
                    retryable=True,
                    cause=e,
                ) from e

        case StructureFormat.CUSTOM:
            if not callable(custom_parser):
                raise ConfigurationError(
                    "structure_format='custom' requires a custom_parser to be provided",
                    retryable=False,
                )
            try:
                return custom_parser(text, target_keys)
            except Exception as e:
                raise ExecutionError(
                    "Custom parser failed to extract data from text",
                    retryable=True,
                    cause=e,
                ) from e

        case StructureFormat.JSON:
            pass

        case _:
            raise ValidationError(
                f"Unsupported structure_format '{structure_format}' in direct_parse",
                retryable=False,
            )

    extracted = Unset
    try:
        extracted = extract_json(text, fuzzy_parse=True, return_one_if_single=False)
    except Exception as e:
        raise ExecutionError(
            "Failed to extract JSON from text during parse",
            retryable=True,
            cause=e,
        ) from e

    if is_sentinel(extracted, _sentinel_check):
        raise ExecutionError(
            "No JSON object could be extracted from text during parse",
            retryable=True,
        )

    try:
        return fuzzy_validate_mapping(
            extracted[0],
            target_keys,
            similarity_threshold=similarity_threshold,
            handle_unmatched=handle_unmatched,
            fill_mapping=fill_mapping,
            fill_value=fill_value,
        )
    except Exception as e:
        raise ExecutionError(
            "Failed to validate extracted JSON during parse",
            retryable=True,
            cause=e,
        ) from e
