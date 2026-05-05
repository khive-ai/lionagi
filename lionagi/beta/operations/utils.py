# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for agent operations.

ReturnAs: Enum controlling how Calling results are unwrapped.
handle_return: Dispatch Calling → desired output form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lionagi._errors import ValidationError
from lionagi.ln.types import Enum
from lionagi.ln.types._sentinel import MaybeUnset, Unset, is_sentinel

if TYPE_CHECKING:
    from collections.abc import Callable

    from lionagi.beta.resource.backend import Calling


class ReturnAs(Enum):
    """How to unwrap a Calling result.

    TEXT      - response.data (typically the LLM text)
    RAW       - response.raw_response (provider-specific dict)
    RESPONSE  - the Normalized object
    MESSAGE   - wrapped as a Message with Assistant content
    CALLING   - the raw Calling event (no unwrap, no validation)
    CUSTOM    - apply caller-supplied return_parser
    """

    TEXT = "text"
    RAW = "raw"
    RESPONSE = "response"
    MESSAGE = "message"
    CALLING = "calling"
    CUSTOM = "custom"


def handle_return(
    calling: Calling,
    return_as: ReturnAs,
    /,
    *,
    return_parser: MaybeUnset[Callable] = Unset,
):
    """Unwrap a Calling into the form requested by return_as.

    CALLING and CUSTOM bypass normalization checks.
    All other modes call calling.assert_is_normalized() first.
    """
    if return_as == ReturnAs.CALLING:
        return calling

    if return_as == ReturnAs.CUSTOM:
        if is_sentinel(return_parser, additions={"none", "empty"}) or not callable(
            return_parser
        ):
            raise ValidationError(
                "return_parser must be provided as a callable when return_as is 'custom'"
            )
        return return_parser(calling)

    calling.assert_is_normalized()
    response = calling.response

    match return_as:
        case ReturnAs.TEXT:
            return response.data
        case ReturnAs.RAW:
            return response.serialized
        case ReturnAs.RESPONSE:
            return response
        case ReturnAs.MESSAGE:
            from lionagi.protocols.messages.assistant_response import (
                parse_to_assistant_message,
            )

            return parse_to_assistant_message(response)
        case _:
            raise ValidationError(f"Unsupported return_as: {return_as.value}")
