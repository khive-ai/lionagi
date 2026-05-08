"""LNDL Formatter — Language Network Directive Language for structured output.

Uses lionagi.lndl lexer/parser for robust extraction. Renders response_format
as LNDL specs in the prompt. Parses LNDL-tagged responses back into validated
Pydantic models or dicts.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from lionagi.libs.schema.breakdown_pydantic_annotation import _is_pydantic_model_cls
from lionagi.lndl import Lexer, Lact, Lvar, Parser
from lionagi.lndl._parse_function_call import parse_function_call
from lionagi.lndl.prompt import LNDL_SYSTEM_PROMPT
from lionagi.lndl.types import ActionCall
from lionagi.ln.fuzzy import FuzzyMatchKeysParams

from ._json_formatter import _referenced_schemas_display, _tool_schemas_display


# ---------------------------------------------------------------------------
# LNDL response structure rendering (schema-driven, matches krons pattern)
# ---------------------------------------------------------------------------

def _render_lndl_response_structure(response_format: Any, has_tools: bool = False) -> str:
    """Generate a schema-driven LNDL guide from the response_format.

    Produces: specs line + typed field list + concrete example pattern + OUT{}.
    The model chooses its own aliases — fuzzy matching handles the rest.
    """
    if isinstance(response_format, BaseModel):
        return _render_lndl_response_structure(type(response_format), has_tools)

    if isinstance(response_format, dict):
        return _render_lndl_dict_structure(response_format, has_tools)

    if not _is_pydantic_model_cls(response_format):
        return ""

    fields = response_format.model_fields
    spec_parts = []
    nested_models: list[tuple[str, type[BaseModel]]] = []
    scalar_specs: list[str] = []

    for spec_name, info in fields.items():
        ann = info.annotation
        if hasattr(ann, "model_fields"):
            inner_fields = ", ".join(ann.model_fields.keys())
            type_name = ann.__name__
            spec_parts.append(f"{spec_name}({type_name}: {inner_fields})")
            nested_models.append((spec_name, ann))
        else:
            type_name = getattr(ann, "__name__", str(ann))
            spec_parts.append(f"{spec_name}({type_name})")
            scalar_specs.append(spec_name)

    specs_line = ", ".join(spec_parts)

    text = f"Specs: {specs_line}\n\n"

    example_lines: list[str] = []
    out_parts: list[str] = []
    alias_idx = 0

    def _next_alias() -> str:
        nonlocal alias_idx
        a = chr(ord("a") + alias_idx % 26)
        alias_idx += 1
        return a

    for spec_name, model in nested_models:
        model_fields = list(model.model_fields.keys())
        spec_aliases: list[str] = []
        for fname in model_fields:
            alias = _next_alias()
            spec_aliases.append(alias)
            if has_tools:
                example_lines.append(
                    f"<lvar {model.__name__}.{fname} {alias}>your value</lvar>"
                    f'  OR  <lact {model.__name__}.{fname} {alias}>tool(arg="val")</lact>'
                )
            else:
                example_lines.append(f"<lvar {model.__name__}.{fname} {alias}>your value</lvar>")
        out_parts.append(f"{spec_name}: [{', '.join(spec_aliases)}]")

    # Top-level fields (not nested models)
    for spec_name, info in fields.items():
        if spec_name not in [n for n, _ in nested_models]:
            ann = info.annotation
            if not hasattr(ann, "model_fields"):
                alias = _next_alias()
                if has_tools:
                    example_lines.append(
                        f"<lvar {response_format.__name__}.{spec_name} {alias}>your value</lvar>"
                        f'  OR  <lact {response_format.__name__}.{spec_name} {alias}>tool(arg="val")</lact>'
                    )
                else:
                    example_lines.append(
                        f"<lvar {response_format.__name__}.{spec_name} {alias}>your value</lvar>"
                    )
                out_parts.append(f"{spec_name}: [{alias}]")

    for spec_name in scalar_specs:
        if spec_name not in [p.split(":")[0].strip() for p in out_parts]:
            out_parts.append(f"{spec_name}: <value>")

    text += "Declare each field, then reference aliases in OUT{}:\n"
    if example_lines:
        text += "\n".join(example_lines) + "\n"
    text += f"\nOUT{{{', '.join(out_parts)}}}\n"
    if has_tools:
        text += '\nTool calls: use `<lact Model.field alias>tool(arg="val")</lact>` '
        text += "with keyword args. Result fills the field.\n"
    return text


def _render_lndl_dict_structure(response_format: dict, has_tools: bool = False) -> str:
    """Render LNDL structure for dict-based response_format."""
    spec_parts = []
    for k, v in response_format.items():
        if _is_pydantic_model_cls(v):
            fields = list(v.model_fields.keys())
            spec_parts.append(f"{k}({v.__name__}: {', '.join(fields)})")
        elif isinstance(v, type):
            spec_parts.append(f"{k}({v.__name__})")
        else:
            spec_parts.append(f"{k}({type(v).__name__})")

    text = f"Specs: {', '.join(spec_parts)}\n\n"

    example_lines: list[str] = []
    out_parts: list[str] = []
    alias_idx = 0

    def _next_alias() -> str:
        nonlocal alias_idx
        a = chr(ord("a") + alias_idx % 26)
        alias_idx += 1
        return a

    for k, v in response_format.items():
        if _is_pydantic_model_cls(v):
            model_fields = list(v.model_fields.keys())
            spec_aliases = []
            for fname in model_fields:
                alias = _next_alias()
                spec_aliases.append(alias)
                example_lines.append(f"<lvar {v.__name__}.{fname} {alias}>your value</lvar>")
            out_parts.append(f"{k}: [{', '.join(spec_aliases)}]")
        else:
            out_parts.append(f"{k}: <value>")

    text += "Declare each field, then reference aliases in OUT{}:\n"
    if example_lines:
        text += "\n".join(example_lines) + "\n"
    text += f"\nOUT{{{', '.join(out_parts)}}}\n"
    return text


# ---------------------------------------------------------------------------
# Parsing — uses real lexer/parser from lionagi.lndl
# ---------------------------------------------------------------------------

def _extract_lndl(text: str) -> tuple[dict[str, dict], dict[str, ActionCall], dict[str, Any]]:
    """Extract lvars, lacts, and OUT block using the LNDL lexer/parser.

    Returns:
        (lvars, lacts, out_fields)
        - lvars: alias → {model, field, value}
        - lacts: alias → ActionCall
        - out_fields: spec_name → [aliases] or scalar literal
    """
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source_text=text)
    program = parser.parse()

    lvars: dict[str, dict] = {}
    for lvar in program.lvars:
        if isinstance(lvar, Lvar):
            lvars[lvar.alias] = {
                "model": lvar.model,
                "field": lvar.field,
                "value": lvar.content,
            }
        else:
            lvars[lvar.alias] = {
                "model": None,
                "field": None,
                "value": lvar.content,
            }

    lacts: dict[str, ActionCall] = {}
    for lact in program.lacts:
        try:
            parsed = parse_function_call(lact.call)
            lacts[lact.alias] = ActionCall(
                name=lact.alias,
                function=parsed["operation"],
                arguments=parsed["arguments"],
                raw_call=lact.call,
            )
        except ValueError:
            pass

    out_fields: dict[str, Any] = {}
    if program.out_block:
        out_fields = dict(program.out_block.fields)

    return lvars, lacts, out_fields


def _resolve_values(
    lvars: dict[str, dict],
    lacts: dict[str, ActionCall],
    out_refs: list[str],
    out_fields: dict[str, Any] | None = None,
    model_class: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """Resolve OUT-referenced aliases into a field→value dict.

    Values can be:
    - str/int/float/bool from <lvar> tags
    - ActionCall from <lact> tags (to be executed by the framework)
    - scalar literals from OUT{} block
    """
    kwargs: dict[str, Any] = {}

    for alias in out_refs:
        if alias in lacts:
            action = lacts[alias]
            field = None
            # Check if lact was namespaced (Model.field)
            for lact_node_alias, ac in lacts.items():
                if lact_node_alias == alias:
                    # Look up field from the original program lact nodes
                    # The field info is embedded in the ActionCall name convention
                    break
            # Try to find field from lvar-style lookup in the lact metadata
            # For now, use alias as field name
            field = alias
            # Check if any lact in the program had a model.field namespace
            kwargs[field] = action
        elif alias in lvars:
            lvar = lvars[alias]
            field = lvar.get("field") or alias
            value = lvar["value"]

            if model_class and field in model_class.model_fields:
                field_info = model_class.model_fields[field]
                annotation = field_info.annotation
                if annotation is int:
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        pass
                elif annotation is float:
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        pass
                elif annotation is bool:
                    value = str(value).lower() in ("true", "1", "yes")
            kwargs[field] = value

    # Pull scalar literals from OUT{}
    if out_fields:
        for k, v in out_fields.items():
            if not isinstance(v, list) and k not in kwargs:
                kwargs[k] = v

    return kwargs


# ---------------------------------------------------------------------------
# LndlFormatter
# ---------------------------------------------------------------------------

class LndlFormatter:
    """LNDL formatter implementing the Formatter protocol.

    Usage:
        branch.operate(
            instruction="Analyze this code",
            response_format=AnalysisReport,
            formatter=LndlFormatter,
        )
    """

    @staticmethod
    def render_schema(format: Any) -> str | None:
        return _referenced_schemas_display(format)

    @staticmethod
    def render_format(format: Any, has_tools: bool = False) -> str:
        parts = [LNDL_SYSTEM_PROMPT.strip(), ""]
        parts.append(_render_lndl_response_structure(format, has_tools=has_tools))
        return "\n".join(parts)

    @staticmethod
    def render_tools(tool_schemas: list[dict[str, Any]]) -> str | None:
        return _tool_schemas_display(tool_schemas)

    @staticmethod
    def parse(
        text: str,
        response_format: Any,
        fuzzy_match_params: FuzzyMatchKeysParams | dict | None = None,
    ) -> Any:
        """Parse LNDL-tagged text into the target response_format.

        Returns:
            - BaseModel with ActionCall placeholders for lact fields (if model format)
            - dict with ActionCall values for lact keys (if dict format)

        The caller (operate pipeline) should check for ActionCall values,
        execute them, and call revalidate_with_action_results() if needed.
        """
        lvars, lacts, out_fields = _extract_lndl(text)

        if not lvars and not lacts and not out_fields:
            raise ValueError("No LNDL tags or OUT{} block found in response")

        # Collect all OUT-referenced aliases
        all_refs: list[str] = []
        for v in out_fields.values():
            if isinstance(v, list):
                all_refs.extend(v)

        # Filter to only OUT-referenced items
        active_lvars = {k: v for k, v in lvars.items() if k in all_refs}
        active_lacts = {k: v for k, v in lacts.items() if k in all_refs}

        if _is_pydantic_model_cls(response_format):
            model_name = response_format.__name__
            spec_name = model_name[0].lower() + model_name[1:]
            model_fields = set(response_format.model_fields.keys())

            refs = out_fields.get(spec_name) or out_fields.get(model_name)

            if isinstance(refs, list):
                # Single OUT entry for the model: OUT{answer: [a, b]}
                kwargs = _resolve_values(active_lvars, active_lacts, refs, out_fields, response_format)
            else:
                # Field-level OUT entries: OUT{q1: [a], q2: [b]}
                # Each OUT key maps to a model field
                kwargs = {}
                for field_name, field_refs in out_fields.items():
                    if field_name in model_fields and isinstance(field_refs, list):
                        for alias in field_refs:
                            if alias in active_lvars:
                                value = active_lvars[alias]["value"]
                                # Coerce types
                                ann = response_format.model_fields[field_name].annotation
                                if ann is int:
                                    try: value = int(value)
                                    except (ValueError, TypeError): pass
                                elif ann is float:
                                    try: value = float(value)
                                    except (ValueError, TypeError): pass
                                elif ann is bool:
                                    value = str(value).lower() in ("true", "1", "yes")
                                kwargs[field_name] = value
                            elif alias in active_lacts:
                                kwargs[field_name] = active_lacts[alias]
                    elif field_name in model_fields and not isinstance(field_refs, list):
                        # Scalar literal: OUT{score: 0.85}
                        kwargs[field_name] = field_refs

            has_actions = any(isinstance(v, ActionCall) for v in kwargs.values())
            if has_actions:
                return kwargs

            return response_format.model_validate(kwargs)

        if isinstance(response_format, BaseModel):
            return LndlFormatter.parse(text, type(response_format), fuzzy_match_params)

        if isinstance(response_format, dict):
            return _resolve_values(active_lvars, active_lacts, all_refs, out_fields)

        raise ValueError(f"Unsupported response_format type: {type(response_format)}")
