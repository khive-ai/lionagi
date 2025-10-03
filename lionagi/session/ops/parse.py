# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import contextlib
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from lionagi.ln import fuzzy_validate_mapping, get_cancelled_exc_class
from lionagi.ln.fuzzy import FuzzyMatchKeysParams, fuzzy_validate_pydantic
from lionagi.protocols.types import AssistantResponse
from lionagi.session.branch import AlcallParams

from .types import ParseContext

if TYPE_CHECKING:
    from lionagi.session.branch import Branch


_CALL = None  # type: ignore


async def parse(
    branch: "Branch",
    text: str,
    parse_ctx: ParseContext,
    return_res_message: bool = False,
) -> Any | tuple[Any, AssistantResponse | None]:

    with contextlib.suppress(Exception):
        result = _validate_dict_or_model(
            text, parse_ctx.response_format, parse_ctx.fuzzy_match_params
        )
        return result if not return_res_message else (result, None)

    async def _inner_parse(i):
        _, res = await branch.chat(
            instruction="reformat text into specified model or structure",
            guidance="follow the required response format, using the model schema as a guide",
            context=[{"text_to_format": text}],
            request_fields=(
                parse_ctx.response_format
                if isinstance(parse_ctx.response_format, dict)
                else None
            ),
            response_format=(
                parse_ctx.response_format
                if isinstance(parse_ctx.response_format, BaseModel)
                else None
            ),
            imodel=parse_ctx.imodel or branch.parse_model,
            sender=branch.user,
            recipient=branch.id,
            return_ins_res_message=True,
        )

        res.metadata["is_parsed"] = True
        res.metadata["original_text"] = text

        return (
            _validate_dict_or_model(
                res.response,
                parse_ctx.response_format,
                parse_ctx.fuzzy_match_params,
            ),
            res,
        )

    _call = parse_ctx.alcall_params or get_default_call()
    if isinstance(parse_ctx.alcall_params, dict):
        _call = AlcallParams(**parse_ctx.alcall_params)

    try:
        result = await _call([0], _inner_parse)
    except get_cancelled_exc_class():
        raise
    except Exception as e:
        match parse_ctx.handle_validation:
            case "raise":
                raise ValueError(f"Failed to parse response: {e}") from e
            case "return_none":
                return (None, None) if return_res_message else None
            case "return_value":
                return (text, None) if return_res_message else text
    return (*result[0],) if return_res_message else result[0][0]


def _validate_dict_or_model(
    text: str,
    response_format: type[BaseModel] | dict,
    fuzzy_match_params: FuzzyMatchKeysParams | dict = None,
):
    try:
        if isinstance(fuzzy_match_params, dict):
            fuzzy_match_params = FuzzyMatchKeysParams(**fuzzy_match_params)

        if isinstance(response_format, type) and issubclass(
            response_format, BaseModel
        ):
            return fuzzy_validate_pydantic(
                text,
                response_format,
                fuzzy_match_params=fuzzy_match_params,
            )

        if isinstance(response_format, dict):
            if fuzzy_match_params is None:
                return fuzzy_validate_mapping(text, response_format)
            elif isinstance(fuzzy_match_params, FuzzyMatchKeysParams):
                return fuzzy_validate_mapping(
                    text, response_format, **fuzzy_match_params.to_dict()
                )

    except Exception as e:
        raise ValueError(f"Failed to parse text: {e}") from e


def get_default_call() -> AlcallParams:
    global _CALL
    if _CALL is None:
        _CALL = AlcallParams(
            retry_initial_delay=1,
            retry_backoff=1.85,
            retry_attempts=3,
            max_concurrent=1,
            throttle_period=1,
        )
    return _CALL
