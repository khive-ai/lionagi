# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import contextlib
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from lionagi.ln import fuzzy_validate_mapping, get_cancelled_exc_class
from lionagi.ln.fuzzy import FuzzyMatchKeysParams, fuzzy_validate_pydantic
from lionagi.session.branch import AlcallParams

if TYPE_CHECKING:
    from lionagi.service.imodel import iModel
    from lionagi.session.branch import Branch

HandleValidation = Literal["raise", "return_value", "return_none"]

_CALL = None  # type: ignore


async def parse(
    branch: "Branch",
    text: str,
    response_format: type[BaseModel] | dict,
    fuzzy_match_params: FuzzyMatchKeysParams | dict = None,
    handle_validation: HandleValidation = "raise",
    alcall_params: AlcallParams | dict | None = None,
    parse_model: "iModel" = None,
    return_res_message: bool = False,
):

    with contextlib.suppress(Exception):
        result = _validate_dict_or_model(
            text, response_format, fuzzy_match_params
        )
        return result if not return_res_message else (result, None)

    async def _inner_parse(i):
        _, res = await branch.chat(
            instruction="reformat text into specified model or structure",
            guidance="follow the required response format, using the model schema as a guide",
            context=[{"text_to_format": text}],
            request_fields=(
                response_format if isinstance(response_format, dict) else None
            ),
            response_format=(
                response_format
                if isinstance(response_format, BaseModel)
                else None
            ),
            imodel=parse_model or branch.parse_model,
            sender=branch.user,
            recipient=branch.id,
            return_ins_res_message=True,
        )

        res.metadata["is_parsed"] = True
        res.metadata["original_text"] = text

        return (
            _validate_dict_or_model(
                res.response, response_format, fuzzy_match_params
            ),
            res,
        )

    _call = alcall_params or get_default_call()
    if isinstance(alcall_params, dict):
        _call = AlcallParams(**alcall_params)

    try:
        result = await _call([0], _inner_parse)
    except get_cancelled_exc_class():
        raise
    except Exception as e:
        match handle_validation:
            case "raise":
                raise ValueError(f"Failed to parse response: {e}") from e
            case "return_none":
                return (None, None) if return_res_message else None
            case "return_value":
                return (text, None) if return_res_message else text
    return result[0] if return_res_message else result[0][0]


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
