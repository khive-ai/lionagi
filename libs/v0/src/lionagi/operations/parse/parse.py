# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from lionagi.libs.validate.fuzzy_validate_mapping import fuzzy_validate_mapping
from lionagi.protocols.types import Operative
from lionagi.utils import breakdown_pydantic_annotation

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
    operative: Operative = None,
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
    if operative is not None:
        max_retries = operative.max_retries
        response_format = operative.request_type or response_format
        request_type = request_type or operative.request_type

    if not request_type and not response_format:
        raise ValueError(
            "Either request_type or response_format must be provided"
        )

    request_type = request_type or response_format

    # First attempt: try to parse the text directly
    import logging

    initial_error = None
    parsed_data = None  # Initialize to avoid scoping issues

    try:
        # Try fuzzy validation first
        parsed_data = fuzzy_validate_mapping(
            text,
            breakdown_pydantic_annotation(request_type),
            similarity_algo=similarity_algo,
            similarity_threshold=similarity_threshold,
            fuzzy_match=fuzzy_match,
            handle_unmatched=handle_unmatched,
            fill_value=fill_value,
            fill_mapping=fill_mapping,
            strict=strict,
            suppress_conversion_errors=False,  # Don't suppress on first attempt
        )

        logging.debug(f"Parsed data from fuzzy validation: {parsed_data}")

        # Validate with pydantic
        if operative is not None:
            response_model = operative.update_response_model(parsed_data)
        else:
            response_model = request_type.model_validate(parsed_data)

        # If successful, return immediately
        if isinstance(response_model, BaseModel):
            return response_model

    except Exception as e:
        initial_error = e
        # Log the initial parsing error for debugging
        logging.debug(
            f"Initial parsing failed for text '{text[:100]}...': {e}"
        )
        logging.debug(
            f"Parsed data was: {locals().get('parsed_data', 'not set')}"
        )

        # Only continue if we have retries left
        if max_retries <= 0:
            if handle_validation == "raise":
                raise ValueError(f"Failed to parse response: {e}") from e
            elif handle_validation == "return_none":
                return None
            else:  # return_value
                return text

    # If direct parsing failed, try using the parse model
    num_try = 0
    last_error = initial_error

    # Check if the parsed_data exists but just failed validation
    # This might mean we have the right structure but wrong values
    if parsed_data is not None and isinstance(parsed_data, dict):
        logging.debug(
            f"Have parsed_data dict, checking if it's close to valid..."
        )
        # If we got a dict with the right keys, maybe we just need to clean it up
        expected_fields = set(request_type.model_fields.keys())
        parsed_fields = set(parsed_data.keys())
        if expected_fields == parsed_fields and all(
            parsed_data.get(k) is not None for k in expected_fields
        ):
            # We have the right structure with non-None values, don't retry with parse model
            logging.debug(
                "Structure matches with valid values, returning original error"
            )
            if handle_validation == "raise":
                raise ValueError(
                    f"Failed to parse response: {initial_error}"
                ) from initial_error
            elif handle_validation == "return_none":
                return None
            else:
                return text

    while num_try < max_retries:
        num_try += 1

        try:
            logging.debug(f"Retry {num_try}: Using parse model to reformat")
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

            # Try to parse the reformatted response
            parsed_data = fuzzy_validate_mapping(
                res.response,
                breakdown_pydantic_annotation(request_type),
                similarity_algo=similarity_algo,
                similarity_threshold=similarity_threshold,
                fuzzy_match=fuzzy_match,
                handle_unmatched=handle_unmatched,
                fill_value=fill_value,
                fill_mapping=fill_mapping,
                strict=strict,
                suppress_conversion_errors=suppress_conversion_errors,
            )

            if operative is not None:
                response_model = operative.update_response_model(parsed_data)
            else:
                response_model = request_type.model_validate(parsed_data)

            # If successful, return
            if isinstance(response_model, BaseModel):
                return response_model

        except InterruptedError as e:
            raise e
        except Exception as e:
            last_error = e
            # Continue to next retry
            continue

    # All retries exhausted
    match handle_validation:
        case "return_value":
            return text
        case "return_none":
            return None
        case "raise":
            error_msg = "Failed to parse response into request format"
            if last_error:
                error_msg += f": {last_error}"
            raise ValueError(error_msg) from last_error
