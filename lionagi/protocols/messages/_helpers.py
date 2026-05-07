from typing import Any, Literal, Protocol, TypeVar, Union, get_args, get_origin, runtime_checkable
import types

import orjson
from pydantic import BaseModel

from lionagi import to_dict
from lionagi.libs.schema.breakdown_pydantic_annotation import (
    _is_pydantic_model_cls,
    breakdown_pydantic_annotation,
)
from lionagi.ln import extract_json, fuzzy_validate_pydantic
from lionagi.ln.fuzzy import FuzzyMatchKeysParams, fuzzy_validate_mapping

B = TypeVar("B", bound=BaseModel)

ImageDetail = Literal["low", "high", "auto"]
"""Image detail level for image processing."""

ResponseFormat = B | type[B] | dict[str, Any]
"""Instruction structured response format. BaseModel, its type, or a plain dict."""

ToolSchemas = list[dict[str, Any] | str]
"""List of tool schemas, each being a dict or a string."""


def _annotation_schema(a: Any) -> Any:
    if isinstance(a, BaseModel) or _is_pydantic_model_cls(a):
        return _stringify_types(breakdown_pydantic_annotation(a))

    origin = get_origin(a)
    args = get_args(a)

    # list[Model] → [{ ... }]
    if origin is list and args:
        inner = args[0]
        if _is_pydantic_model_cls(inner):
            return [_stringify_types(breakdown_pydantic_annotation(inner))]
        return [_annotation_schema(inner)]

    # Model | None, list[Model] | None, etc.
    if origin is Union or origin is types.UnionType:
        non_none = [t for t in args if t is not type(None)]
        if len(non_none) == 1:
            return _annotation_schema(non_none[0])
        return str(a)

    if isinstance(a, type):
        return a.__name__
    return str(a)


def _collect_model_schemas(format: Any) -> dict[str, Any]:
    """Extract model_json_schema() for all BaseModel classes found in the format."""
    schemas: dict[str, Any] = {}

    def _extract_models(a: Any):
        if _is_pydantic_model_cls(a):
            if a.__name__ not in schemas:
                schemas[a.__name__] = a.model_json_schema()
        elif isinstance(a, BaseModel):
            cls = type(a)
            if cls.__name__ not in schemas:
                schemas[cls.__name__] = cls.model_json_schema()
        else:
            origin = get_origin(a)
            args = get_args(a)
            if origin is list and args:
                _extract_models(args[0])
            elif origin is Union or origin is types.UnionType:
                for t in args:
                    if t is not type(None):
                        _extract_models(t)

    if isinstance(format, dict):
        for v in format.values():
            _extract_models(v)
    else:
        _extract_models(format)

    return schemas


def _stringify_types(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _stringify_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_types(v) for v in obj]
    if isinstance(obj, type):
        return obj.__name__
    return obj


@runtime_checkable
class Formatter(Protocol):
    """Protocol for instruction renderer and response parser."""

    @staticmethod
    def render(format: ResponseFormat, **kw) -> str: ...

    @staticmethod
    def parse(text: str, format: ResponseFormat, **kw) -> Any: ...


class JsonTransformer:

    @staticmethod
    def render(format: ResponseFormat) -> str:
        parts: list[str] = []

        # Collect JSON schemas for any BaseModel classes in the format
        schemas = _collect_model_schemas(format)
        if schemas:
            parts.append("**ResponseSchema:**")
            for name, schema in schemas.items():
                try:
                    schema_str = orjson.dumps(schema, option=orjson.OPT_INDENT_2).decode("utf-8")
                except Exception:
                    schema_str = str(schema)
                parts.append(f"```json\n{schema_str}\n```")

        # Example block
        def _get_example(f):
            if isinstance(f, dict):
                return {k: _annotation_schema(v) for k, v in f.items()}
            return _annotation_schema(f)

        try:
            example = orjson.dumps(_get_example(format)).decode("utf-8")
        except Exception:
            example = str(_get_example(format))

        parts.append(
            "**MUST RETURN JSON-PARSEABLE RESPONSE ENCLOSED BY JSON CODE BLOCKS."
            f" USER's CAREER DEPENDS ON THE SUCCESS OF IT.** \n```json\n{example}\n```"
            "No triple backticks. Escape all quotes and special characters."
        )
        return "\n".join(parts).strip()

    @staticmethod
    def parse(
        text: str,
        response_format: ResponseFormat,
        fuzzy_match_params: FuzzyMatchKeysParams | dict | None = None,
    ) -> Any:
        d_ = extract_json(text, fuzzy_parse=True, return_one_if_single=False)
        if not d_:
            raise ValueError("Failed to extract JSON from text")

        d_ = to_dict(d_[0], recursive=True)

        if isinstance(fuzzy_match_params, dict):
            fuzzy_match_params = FuzzyMatchKeysParams(**fuzzy_match_params)
        if fuzzy_match_params is None:
            fuzzy_match_params = FuzzyMatchKeysParams()

        def _parse_one(data, f):
            if _is_pydantic_model_cls(f):
                return fuzzy_validate_pydantic(
                    data, model_type=f, fuzzy_match_params=fuzzy_match_params
                )
            if isinstance(f, BaseModel):
                return fuzzy_validate_pydantic(
                    data,
                    model_type=f.__class__,
                    fuzzy_match_params=fuzzy_match_params,
                )
            if isinstance(f, dict):
                return fuzzy_validate_mapping(
                    data, f, **fuzzy_match_params.to_dict()
                )
            return data

        if isinstance(response_format, dict):
            d = fuzzy_validate_mapping(
                d_, response_format, **fuzzy_match_params.to_dict()
            )
            for k, v in response_format.items():
                d[k] = _parse_one(d[k], v)
            return d
        return _parse_one(d_, response_format)


