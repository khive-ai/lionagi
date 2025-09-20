# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from lionagi.ln import get_cancelled_exc_class
from lionagi.ln.fuzzy._fuzzy_match import FuzzyMatchKeysParams
from lionagi.ln.fuzzy._fuzzy_validate import fuzzy_validate_pydantic

if TYPE_CHECKING:
    from lionagi.session.branch import Branch


async def parse(
    branch: "Branch",
    text: str,
    handle_validation: Literal[
        "raise", "return_value", "return_none"
    ] = "return_value",
    max_retries: int = 3,
    request_type: type[BaseModel] = None,
    operative=None,
    similarity_algo="jaro_winkler",
    similarity_threshold: float = 0.85,
    fuzzy_match: bool = True,
    handle_unmatched: Literal[
        "ignore", "raise", "remove", "fill", "force"
    ] = "force",
    fill_value: Any = None,
    fill_mapping: dict[str, Any] | None = None,
    strict: bool = False,
    suppress_conversion_errors: bool = False,
    response_format=None,
):
    import logging

    # Extract parameters from operative if provided
    if operative is not None:
        max_retries = operative.max_retries
        response_format = operative.request_type or response_format
        request_type = request_type or operative.request_type

    if not request_type and not response_format:
        raise ValueError(
            "Either request_type or response_format must be provided"
        )

    request_type = request_type or response_format

    # Build fuzzy match parameters
    fuzzy_match_params = (
        FuzzyMatchKeysParams(
            similarity_algo=similarity_algo,
            similarity_threshold=similarity_threshold,
            handle_unmatched=handle_unmatched,
            fill_value=fill_value,
            fill_mapping=fill_mapping,
            strict=strict,
        )
        if fuzzy_match
        else None
    )

    async def _attempt_parse(input_text: str, suppress_errors: bool = False):
        """Helper function to attempt parsing with consistent logic."""
        try:
            # Use fuzzy_validate_pydantic for consistent behavior
            response_model = fuzzy_validate_pydantic(
                input_text,
                request_type,
                fuzzy_parse=True,
                fuzzy_match=fuzzy_match,
                fuzzy_match_params=fuzzy_match_params,
            )

            # If operative is provided, update it with the parsed data
            if operative is not None:
                # Update operative's response model with validated data
                operative.response_model = response_model
                operative._should_retry = False
                return response_model

            return response_model

        except Exception as e:
            if not suppress_errors:
                logging.debug(f"Parse attempt failed: {e}")
            # If operative is provided, update its state for retry logic
            if operative is not None:
                operative.response_str_dict = input_text
                operative._should_retry = True
            raise

    # First attempt: try to parse the text directly
    try:
        response_model = await _attempt_parse(text)
        if isinstance(response_model, BaseModel):
            return response_model

    except Exception as e:
        initial_error = e
        logging.debug(
            f"Initial parsing failed for text '{text[:100]}...': {e}"
        )

        # Handle case with no retries
        if max_retries <= 0:
            match handle_validation:
                case "raise":
                    raise ValueError(f"Failed to parse response: {e}") from e
                case "return_none":
                    return None
                case _:
                    return text

    # Retry loop with parse model
    last_error = initial_error
    for attempt in range(1, max_retries + 1):
        try:
            logging.debug(f"Retry {attempt}: Using parse model to reformat")

            # Use the branch's parse model to reformat the text
            _, res = await branch.chat(
                instruction="reformat text into specified model",
                guidance="follow the required response format, using the model schema as a guide",
                context=[{"text_to_format": text}],
                response_format=request_type,
                sender=branch.user,
                recipient=branch.id,
                imodel=branch.parse_model,
                return_ins_res_message=True,
            )

            # Attempt to parse the reformatted response
            response_model = await _attempt_parse(
                res.response, suppress_errors=suppress_conversion_errors
            )
            if isinstance(response_model, BaseModel):
                return response_model

        except get_cancelled_exc_class():
            raise
        except Exception as e:
            last_error = e
            continue

    # All retries exhausted - handle according to specified behavior
    match handle_validation:
        case "raise":
            error_msg = "Failed to parse response into request format"
            if last_error:
                error_msg += f": {last_error}"
            raise ValueError(error_msg) from last_error
        case "return_none":
            return None
        case _:
            return text
