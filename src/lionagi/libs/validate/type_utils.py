# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
from datetime import datetime
from uuid import UUID
from typing import Union # For Python < 3.10 compatibility for str | UUID

__all__ = ("convert_to_datetime", "validate_uuid")

def convert_to_datetime(v_input) -> datetime: # Added type hint for v_input and return
    if isinstance(v_input, datetime):
        return v_input
    if isinstance(v_input, str):
        # Attempt to remove 'Z' if it's UTC, as fromisoformat might not handle it directly in all Python versions
        # for timezone-aware datetimes without further parsing.
        # However, Python 3.7+ fromisoformat handles 'Z' correctly.
        # For simplicity and broad compatibility if issues arise, one might strip 'Z'.
        # For now, assume direct fromisoformat is sufficient.
        with contextlib.suppress(ValueError):
            return datetime.fromisoformat(v_input.replace('Z', '+00:00') if v_input.endswith('Z') else v_input)


    error_msg = "Input value should be a `datetime.datetime` object or `isoformat` string"
    raise ValueError(error_msg)

def validate_uuid(v_input: Union[str, UUID]) -> UUID: # Added type hint for v_input
    if isinstance(v_input, UUID):
        return v_input
    if isinstance(v_input, str):
        try:
            return UUID(v_input)
        except ValueError as e: # Catch specific ValueError for clarity
            error_msg = "Input string is not a valid UUID representation"
            raise ValueError(error_msg) from e
    
    # If not str or UUID, it's an invalid type for this validator's typical use.
    error_msg = "Input value should be a `uuid.UUID` object or a valid UUID string"
    raise TypeError(error_msg) # Changed to TypeError for incorrect input type