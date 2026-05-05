import re
from inspect import isclass
from typing import Any, get_args, get_origin

from pydantic import BaseModel

__all__ = (
    "_clean_result",
    "_clean_type_repr",
    "breakdown_pydantic_annotation",
    "is_pydantic_model",
)

_MODULE_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*\.)+([a-zA-Z_][a-zA-Z0-9_]*)")
_CLASS_PATTERN = re.compile(r"<class '([^']+)'>")


def _clean_type_repr(t: Any) -> str:
    s = str(t) if not isinstance(t, str) else t
    if match := _CLASS_PATTERN.match(s):
        type_name = match.group(1)
        return type_name.rsplit(".", 1)[-1]
    return _MODULE_PATTERN.sub(r"\2", s)


def _clean_result(result: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in result.items():
        if isinstance(v, dict):
            out[k] = _clean_result(v)
        elif isinstance(v, list) and v:
            if isinstance(v[0], dict):
                out[k] = [_clean_result(v[0])]
            else:
                out[k] = [_clean_type_repr(v[0])]
        else:
            out[k] = _clean_type_repr(v)
    return out


def breakdown_pydantic_annotation(
    model: type[BaseModel],
    max_depth: int | None = None,
    current_depth: int = 0,
    clean_types: bool = False,
) -> dict[str, Any]:
    result = _breakdown_pydantic_annotation(
        model=model,
        max_depth=max_depth,
        current_depth=current_depth,
    )
    if clean_types:
        return _clean_result(result)
    return result


def _breakdown_pydantic_annotation(
    model: type[BaseModel],
    max_depth: int | None = None,
    current_depth: int = 0,
) -> dict[str, Any]:
    if not is_pydantic_model(model):
        raise TypeError("Input must be a Pydantic model")

    if max_depth is not None and current_depth >= max_depth:
        raise RecursionError("Maximum recursion depth reached")

    out: dict[str, Any] = {}
    for k, v in model.__annotations__.items():
        origin = get_origin(v)
        if is_pydantic_model(v):
            out[k] = _breakdown_pydantic_annotation(v, max_depth, current_depth + 1)
        elif origin is list:
            args = get_args(v)
            if args and is_pydantic_model(args[0]):
                out[k] = [
                    _breakdown_pydantic_annotation(
                        args[0], max_depth, current_depth + 1
                    )
                ]
            else:
                out[k] = [args[0] if args else Any]
        else:
            out[k] = v

    return out


def is_pydantic_model(x: Any) -> bool:
    try:
        return isclass(x) and issubclass(x, BaseModel)
    except TypeError:
        return False


_is_pydantic_model = is_pydantic_model
