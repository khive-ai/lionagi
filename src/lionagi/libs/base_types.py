# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import copy # Renamed _copy to copy for clarity
from abc import ABC
from typing import Any, Literal, TypedDict
from pydantic import BaseModel

__all__ = ("UndefinedType", "UNDEFINED", "KeysDict", "Params", "DataClass")

class UndefinedType:
    def __init__(self) -> None:
        self.undefined = True

    def __bool__(self) -> Literal[False]:
        return False

    def __deepcopy__(self, memo):
        # Ensure UNDEFINED is universal by returning the singleton
        memo[id(self)] = self 
        return self

    def __repr__(self) -> Literal["UNDEFINED"]:
        return "UNDEFINED"

    __slots__ = ("undefined",) # Added comma for single-element tuple

UNDEFINED = UndefinedType()

class KeysDict(TypedDict, total=False):
    """TypedDict for keys dictionary."""
    key: Any

class Params(BaseModel):
    """Base class for parameter models, often used with callable instances."""
    def keys(self):
        return self.__class__.model_fields.keys()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "This __call__ method is intended to be implemented by subclasses."
        )

class DataClass(ABC):
    """Abstract base class, can be used as a marker for custom data classes."""
    pass