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


# ---------------------------------------------------------------------------
# Type annotation helpers
# ---------------------------------------------------------------------------

def _annotation_schema(a: Any) -> Any:
    if isinstance(a, BaseModel) or _is_pydantic_model_cls(a):
        return _stringify_types(breakdown_pydantic_annotation(a))

    origin = get_origin(a)
    args = get_args(a)

    if origin is list and args:
        inner = args[0]
        if _is_pydantic_model_cls(inner):
            return [_stringify_types(breakdown_pydantic_annotation(inner))]
        return [_annotation_schema(inner)]

    if origin is Union or origin is types.UnionType:
        non_none = [t for t in args if t is not type(None)]
        if len(non_none) == 1:
            return _annotation_schema(non_none[0])
        return str(a)

    if isinstance(a, type):
        return a.__name__
    if a is Any:
        return "any"
    return str(a)


def _stringify_types(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _stringify_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_types(v) for v in obj]
    if obj is Any:
        return "any"
    if isinstance(obj, type):
        return obj.__name__
    origin = get_origin(obj)
    if origin is not None:
        return _annotation_schema(obj)
    return obj


# ---------------------------------------------------------------------------
# TypeScript-style schema rendering
# ---------------------------------------------------------------------------

def _typescript_type(json_type: str) -> str:
    return {"string": "string", "integer": "int", "number": "float",
            "boolean": "bool", "array": "array", "object": "object",
            "null": "null"}.get(json_type, json_type)


def _extract_ts_type(spec: dict, required: bool) -> tuple[str, bool]:
    if "enum" in spec:
        parts = [f'"{v}"' if isinstance(v, str) else ("null" if v is None else str(v))
                 for v in spec["enum"]]
        return " | ".join(parts), not required or None in spec["enum"]

    if "anyOf" in spec:
        parts, has_null = [], False
        for opt in spec["anyOf"]:
            if opt.get("type") == "null":
                has_null = True; continue
            if "type" in opt:
                if opt["type"] == "array" and "items" in opt:
                    it = opt["items"]
                    parts.append(f"{it.get('$ref', '').split('/')[-1] or _typescript_type(it.get('type', 'any'))}[]")
                else:
                    parts.append(_typescript_type(opt["type"]))
            elif "$ref" in opt:
                parts.append(opt["$ref"].split("/")[-1])
        if has_null:
            parts.append("null")
        return " | ".join(parts) or "any", not required or has_null

    if spec.get("type") == "array":
        items = spec.get("items", {})
        it = items.get("$ref", "").split("/")[-1] or _typescript_type(items.get("type", "any"))
        return f"{it}[]", not required

    if "$ref" in spec:
        return spec["$ref"].split("/")[-1], not required

    if "type" in spec:
        return _typescript_type(spec["type"]), not required

    return "any", not required


def _ts_field_line(name: str, spec: dict, required: bool, indent: int = 0) -> str:
    prefix = "  " * indent
    ts_type, optional = _extract_ts_type(spec, required)
    default = ""
    if "default" in spec:
        d = spec["default"]
        if isinstance(d, str):
            default = f' = "{d}"'
        elif d is None:
            default = " = null"
        elif isinstance(d, bool):
            default = f" = {'true' if d else 'false'}"
        else:
            default = f" = {d}"
    opt = "?" if optional else ""
    line = f"{prefix}{name}{opt}: {ts_type}{default}"
    if spec.get("description"):
        line += f" - {spec['description']}"
    return line


def _typescript_schema(schema: dict, indent: int = 0) -> str:
    if "properties" not in schema:
        return ""
    required = set(schema.get("required", []))
    lines = [_ts_field_line(n, s, n in required, indent)
             for n, s in schema["properties"].items()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schema / tool display helpers
# ---------------------------------------------------------------------------

def _collect_model_schemas(format: Any) -> dict[str, Any]:
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


def _referenced_schemas_display(format: Any) -> str | None:
    """Render only $defs / referenced sub-models as TypeScript interfaces."""
    schema = None
    if _is_pydantic_model_cls(format):
        schema = format.model_json_schema()
    elif isinstance(format, BaseModel):
        schema = type(format).model_json_schema()
    elif isinstance(format, dict):
        schemas = _collect_model_schemas(format)
        if not schemas:
            return None
        parts = []
        for name, s in schemas.items():
            desc = s.get("description", "")
            header = f"interface {name}"
            if desc:
                header += f"  // {desc.splitlines()[0]}"
            parts.append(f"{header} {{\n{_typescript_schema(s, 1)}\n}}")
        return "\n\n".join(parts)
    else:
        return None

    if schema is None:
        return None

    defs = schema.get("$defs", {})
    if not defs:
        return None

    parts = []
    for def_name, def_schema in defs.items():
        desc = def_schema.get("description", "")
        header = f"interface {def_name}"
        if desc:
            header += f"  // {desc.splitlines()[0]}"
        parts.append(f"{header} {{\n{_typescript_schema(def_schema, 1)}\n}}")
    return "\n\n".join(parts)


def _response_format_display(format: Any) -> str:
    """Render the ResponseFormat section: field defs + JSON example."""
    schema = None
    if _is_pydantic_model_cls(format):
        schema = format.model_json_schema()
    elif isinstance(format, BaseModel):
        schema = type(format).model_json_schema()

    parts = []

    # Inline field definitions (TypeScript-style) for the top-level response
    if schema and "properties" in schema:
        parts.append(_typescript_schema(schema, 0))
    elif isinstance(format, dict):
        # For dict formats, build inline field defs from the dict
        lines = []
        for k, v in format.items():
            ts = _annotation_schema(v)
            if isinstance(ts, (dict, list)):
                ts = str(ts)
            lines.append(f"{k}: {ts}")
        parts.append("\n".join(lines))

    # JSON example
    def _get_example(f):
        if isinstance(f, dict):
            return {k: _annotation_schema(v) for k, v in f.items()}
        return _annotation_schema(f)

    try:
        example = orjson.dumps(_get_example(format)).decode("utf-8")
    except Exception:
        example = str(_get_example(format))
    parts.append(f"\nExample:\n```json\n{example}\n```")

    parts.append(
        "**MUST RETURN EXACTLY ONE JSON-PARSEABLE RESPONSE ENCLOSED "
        "BY JSON CODE BLOCKS. USER's CAREER DEPENDS ON THE SUCCESS OF IT."
        "NO TRIPLE BACKTICKS. ESCAPE ALL QUOTES AND SPECIAL CHARACTERS.**"
    )

    return "\n".join(parts)


def _tool_schemas_display(tool_schemas: list[dict[str, Any]]) -> str | None:
    parts = []
    for item in tool_schemas:
        tools = item.get("tools", [item] if "function" in item else [])
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
            elif "name" in tool:
                func = tool
            else:
                continue
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            header = f"function {name}("
            if params.get("properties"):
                args = []
                required = set(params.get("required", []))
                for pname, pspec in params["properties"].items():
                    ptype = _typescript_type(pspec.get("type", "any"))
                    opt = "" if pname in required else "?"
                    arg = f"{pname}{opt}: {ptype}"
                    if pspec.get("description"):
                        arg += f"  // {pspec['description']}"
                    args.append(arg)
                header += ", ".join(args)
            header += ")"
            if desc:
                header += f"  // {desc}"
            parts.append(header)
    return "\n".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Formatter protocol + JsonTransformer
# ---------------------------------------------------------------------------

@runtime_checkable
class Formatter(Protocol):
    """Protocol for instruction renderer and response parser."""

    @staticmethod
    def render_schema(format: ResponseFormat) -> str | None: ...

    @staticmethod
    def render_format(format: ResponseFormat) -> str: ...

    @staticmethod
    def render_tools(tool_schemas: list[dict[str, Any]]) -> str | None: ...

    @staticmethod
    def parse(text: str, format: ResponseFormat, **kw) -> Any: ...


class JsonTransformer:

    @staticmethod
    def render_schema(format: ResponseFormat) -> str | None:
        return _referenced_schemas_display(format)

    @staticmethod
    def render_format(format: ResponseFormat) -> str:
        return _response_format_display(format)

    @staticmethod
    def render_tools(tool_schemas: list[dict[str, Any]]) -> str | None:
        return _tool_schemas_display(tool_schemas)

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
