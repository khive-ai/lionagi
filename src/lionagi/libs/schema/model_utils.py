# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel
from typing import Any, Dict 

__all__ = ("validate_model_to_dict",)

def validate_model_to_dict(v_input: Any) -> Dict[str, Any]:
    """
    Serialize a Pydantic model to a dictionary, or return if already a dict.
    Returns an empty dict for None input.
    """

    if isinstance(v_input, BaseModel):
        return v_input.model_dump()
    if v_input is None:
        return {}
    if isinstance(v_input, dict):
        return v_input
    
    error_msg = (
        "Input value for serialization to dict should be a `pydantic.BaseModel` "
        "object, a `dict`, or `None`."
    )
    raise ValueError(error_msg)