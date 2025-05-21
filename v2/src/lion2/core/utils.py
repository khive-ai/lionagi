from __future__ import annotations

import copy as _copy
from inspect import isclass
from typing import Any, TypeVar, get_args, get_origin

from pydantic import BaseModel

T = TypeVar("T")

__all__ = (
    "copy",
    "breakdown_pydantic_annotation",
)


def copy(obj: T, /, *, deep: bool = True, num: int = 1) -> T | list[T]:
    if num < 1:
        raise ValueError("Number of copies must be at least 1")

    copy_func = _copy.deepcopy if deep else _copy.copy
    return [copy_func(obj) for _ in range(num)] if num > 1 else copy_func(obj)


def breakdown_pydantic_annotation(
    model: type[BaseModel],
    max_depth: int | None = None,
    current_depth: int = 0,
) -> dict[str, Any]:
    if not _is_pydantic_model(model):
        raise TypeError("Input must be a Pydantic model")

    if max_depth is not None and current_depth >= max_depth:
        raise RecursionError("Maximum recursion depth reached")

    out: dict[str, Any] = {}
    for k, v in model.__annotations__.items():
        origin = get_origin(v)
        if _is_pydantic_model(v):
            out[k] = breakdown_pydantic_annotation(
                v, max_depth, current_depth + 1
            )
        elif origin is list:
            args = get_args(v)
            if args and _is_pydantic_model(args[0]):
                out[k] = [
                    breakdown_pydantic_annotation(
                        args[0], max_depth, current_depth + 1
                    )
                ]
            else:
                out[k] = [args[0] if args else Any]
        elif v is list:  # Handle plain 'list' annotation as List[Any]
            out[k] = [Any]
        else:
            out[k] = v

    return out


def _is_pydantic_model(x: Any) -> bool:
    try:
        return isclass(x) and issubclass(x, BaseModel)
    except TypeError:
        return False


def validate_model_to_type(cls, value):
    if value is None:
        return BaseModel
    if isinstance(value, type) and issubclass(value, BaseModel):
        return value
    if isinstance(value, BaseModel):
        return value.__class__
    raise ValueError("Base must be a BaseModel subclass or instance.")