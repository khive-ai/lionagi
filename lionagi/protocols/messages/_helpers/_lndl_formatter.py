"""LNDL Formatter — Language Network Directive Language for structured output.

Uses lionagi.lndl lexer/parser/assembler. Renders response_format
as schema-driven LNDL specs in the prompt. Parses LNDL-tagged responses
back into validated Pydantic models or dicts.
"""

from __future__ import annotations

from typing import Any, get_args, get_origin

from pydantic import BaseModel

from lionagi.libs.schema.breakdown_pydantic_annotation import _is_pydantic_model_cls
from lionagi.ln.fuzzy import FuzzyMatchKeysParams
from lionagi.lndl import Lexer, Parser, assemble, collect_actions, normalize_lndl_text

from ._json_formatter import _referenced_schemas_display, _tool_schemas_display

# ---------------------------------------------------------------------------
# Schema-driven LNDL rendering
# ---------------------------------------------------------------------------


def _type_name(t: Any) -> str:
    """Best-effort short type name for a Python annotation."""
    origin = get_origin(t)
    args = get_args(t)
    if origin is list:
        if args and _is_pydantic_model_cls(args[0]):
            inner = args[0]
            inner_fields = ", ".join(inner.model_fields.keys())
            return f"list[{inner.__name__}: {inner_fields}]"
        if args:
            return f"list[{_type_name(args[0])}]"
        return "list"
    if origin is dict:
        if len(args) == 2:
            return f"dict[{_type_name(args[0])}, {_type_name(args[1])}]"
        return "dict"
    if _is_pydantic_model_cls(t):
        return f"{t.__name__}: {', '.join(t.model_fields.keys())}"
    return getattr(t, "__name__", None) or str(t)


def _spec_summary(field_name: str, ann: Any) -> str:
    return f"{field_name}({_type_name(ann)})"


def _render_field_example(
    field_name: str,
    ann: Any,
    has_tools: bool,
    alias_gen,
) -> tuple[list[str], str, str | None]:
    """Render LNDL example tags for a single response_format field.

    Returns (declaration_lines, named_out_entry, single_alias_or_None).
    The single_alias is non-None only when this field maps to exactly one alias
    — in that case, the OUT{} can use the shortcut form ``OUT{alias}``.
    """
    origin = get_origin(ann)
    args = get_args(ann)

    # list[X]
    if origin is list:
        elem = args[0] if args else Any
        if _is_pydantic_model_cls(elem):
            # list[Model] — show 2 sample items with nested-group OUT syntax.
            lines: list[str] = []
            grouped: list[list[str]] = []
            for i in range(2):
                item_aliases: list[str] = []
                for fname, finfo in elem.model_fields.items():
                    a = alias_gen()
                    item_aliases.append(a)
                    f_ann = finfo.annotation
                    f_origin = get_origin(f_ann)
                    if f_origin is list:
                        hint = " (JSON array)"
                    elif f_origin is dict:
                        hint = " (JSON object)"
                    else:
                        hint = ""
                    if has_tools and i == 0:
                        lines.append(
                            f"<lvar {elem.__name__}.{fname} {a}>...{hint}</lvar>  "
                            f'OR  <lact {elem.__name__}.{fname} {a}>tool(arg="val")</lact>'
                        )
                    else:
                        lines.append(
                            f"<lvar {elem.__name__}.{fname} {a}>...{hint}</lvar>"
                        )
                grouped.append(item_aliases)
            groups_str = ", ".join("[" + ", ".join(g) + "]" for g in grouped)
            return lines, f"{field_name}: [{groups_str}]", None
        # list[scalar]
        lines = []
        aliases = []
        for _ in range(2):
            a = alias_gen()
            aliases.append(a)
            lines.append(f"<lvar {a}>...</lvar>")
        return lines, f"{field_name}: [{', '.join(aliases)}]", None

    # dict[K, V] — declare as ``<lvar field.key alias>...</lvar>`` where the
    # second segment is the actual dictionary key the model picks for each
    # entry. Show two concrete sample keys so the model doesn't drift to
    # angle-bracket placeholders (which the lexer would treat as markup).
    if origin is dict:
        lines = []
        aliases = []
        for i in range(2):
            a = alias_gen()
            aliases.append(a)
            lines.append(
                f"<lvar {field_name}.key{i+1} {a}>...</lvar>  "
                "(replace 'key1'/'key2' with the actual dict key)"
            )
        return lines, f"{field_name}: [{', '.join(aliases)}]", None

    # nested model
    if _is_pydantic_model_cls(ann):
        lines = []
        aliases = []
        for fname, finfo in ann.model_fields.items():
            a = alias_gen()
            aliases.append(a)
            f_ann = finfo.annotation
            f_origin = get_origin(f_ann)
            # Hint that nested list fields take JSON arrays
            if f_origin is list:
                hint = ' (use JSON array: ["x", "y"])'
            elif f_origin is dict:
                hint = ' (use JSON object: {"k": "v"})'
            else:
                hint = ""
            if has_tools:
                lines.append(
                    f"<lvar {ann.__name__}.{fname} {a}>...{hint}</lvar>  "
                    f'OR  <lact {ann.__name__}.{fname} {a}>tool(arg="val")</lact>'
                )
            else:
                lines.append(f"<lvar {ann.__name__}.{fname} {a}>...{hint}</lvar>")
        return lines, f"{field_name}: [{', '.join(aliases)}]", None

    # scalar (int, float, str, bool, etc.)
    a = alias_gen()
    if has_tools:
        line = (
            f"<lvar {field_name} {a}>...</lvar>  "
            f'OR  <lact {field_name} {a}>tool(arg="val")</lact>'
        )
    else:
        line = f"<lvar {field_name} {a}>...</lvar>"
    return [line], f"{field_name}: [{a}]", a


def _render_lndl_response_structure(
    response_format: Any,
    has_tools: bool = False,
) -> str:
    """Generate a schema-driven LNDL guide from the response_format."""
    if isinstance(response_format, BaseModel):
        return _render_lndl_response_structure(type(response_format), has_tools)

    if isinstance(response_format, dict):
        return _render_lndl_dict_structure(response_format, has_tools)

    if not _is_pydantic_model_cls(response_format):
        return ""

    fields = response_format.model_fields

    # Skip framework-internal action fields — they're produced via <lact>
    skip = {"action_required", "action_requests", "action_responses"}

    spec_parts: list[str] = []
    decl_lines: list[str] = []
    out_entries: list[str] = []
    single_aliases: list[str | None] = []
    list_model_fields: list[tuple[str, type]] = []  # (spec_name, model_cls)
    counter = [0]

    def alias_gen() -> str:
        # Use 2-character aliases with explicit numeric suffix from index 0:
        # a1, b1, c1, ... z1, a2, b2, ...  This produces 676 distinct aliases
        # without any collision risk and trains the model — by example — that
        # aliases are multi-character. The earlier scheme started with single
        # letters (a, b, c) which models often duplicated mid-response on
        # large schemas (>10 fields). Two-char aliases are still short enough
        # to keep the OUT{} block readable.
        i = counter[0]
        counter[0] += 1
        letter = chr(ord("a") + (i % 26))
        suffix = (i // 26) + 1
        return f"{letter}{suffix}"

    for spec_name, info in fields.items():
        if spec_name in skip:
            continue
        ann = info.annotation
        spec_parts.append(_spec_summary(spec_name, ann))
        decls, out_entry, single_alias = _render_field_example(
            spec_name, ann, has_tools, alias_gen
        )
        decl_lines.extend(decls)
        out_entries.append(out_entry)
        single_aliases.append(single_alias)
        # Track list[Model] fields for targeted nudge
        if get_origin(ann) is list:
            inner = (get_args(ann) or (None,))[0]
            if _is_pydantic_model_cls(inner):
                list_model_fields.append((spec_name, inner))

    if not spec_parts:
        return ""

    text = f"Specs: {', '.join(spec_parts)}\n\n"
    text += (
        "Declare each field with <lvar> or <lact>, then commit aliases in OUT{}.\n"
        "Every spec MUST appear in OUT{} unless the schema gives it a default value. "
        "Nullable specs (those that accept None) may use null/None as the value, "
        "but they must still be present in OUT{} if they have no default.\n\n"
    )
    text += "\n".join(decl_lines) + "\n\n"

    # Always render the explicit, named form. The shortcut form (OUT{alias, ...})
    # is documented in the system prompt; rendering it here would push the model
    # to drop spec names from declarations, which breaks resolution.
    text += "OUT{" + ", ".join(out_entries) + "}\n"

    # When the schema has list[Model] fields, models often try to write each
    # item as a single string. Spell out the nested-group rule with a concrete
    # right/wrong example pinned to one of the actual specs.
    if list_model_fields:
        spec_name, model_cls = list_model_fields[0]
        field_names = list(model_cls.model_fields.keys())
        first_aliases = ", ".join(field_names[: min(3, len(field_names))])
        text += (
            "\nIMPORTANT — list[Model] fields use NESTED GROUPS in OUT{}, not a "
            "flat array of strings. Each inner array is one item, with one alias "
            "per model field in declaration order.\n"
            f'  WRONG: {spec_name}: ["...", "..."]   (strings instead of aliases)\n'
            f"  RIGHT: {spec_name}: [[a1, a2, a3], [b1, b2, b3], ...]   "
            f"(aliases for {first_aliases})\n"
            "Apply this to EVERY list[Model] field above. Add as many groups as "
            "you need; each group becomes one model instance.\n"
        )

    if has_tools:
        text += (
            '\nTool calls: <lact spec alias>tool(arg="val")</lact> for top-level '
            "scalar specs, <lact Model.field alias>tool(...)</lact> for model fields. "
            "The tool's result becomes the field's value — DON'T also write a separate <lvar>.\n"
        )
    return text


def _render_lndl_dict_structure(response_format: dict, has_tools: bool = False) -> str:
    """Render LNDL structure for dict-based response_format."""
    spec_parts: list[str] = []
    decl_lines: list[str] = []
    out_entries: list[str] = []
    counter = [0]

    def alias_gen() -> str:
        i = counter[0]
        counter[0] += 1
        return chr(ord("a") + (i % 26))

    for k, v in response_format.items():
        if isinstance(v, type) and _is_pydantic_model_cls(v):
            spec_parts.append(f"{k}({v.__name__}: {', '.join(v.model_fields.keys())})")
            ann = v
        elif isinstance(v, type):
            spec_parts.append(f"{k}({v.__name__})")
            ann = v
        else:
            spec_parts.append(f"{k}({type(v).__name__})")
            ann = type(v)
        decls, entry, _single = _render_field_example(k, ann, has_tools, alias_gen)
        decl_lines.extend(decls)
        out_entries.append(entry)

    text = f"Specs: {', '.join(spec_parts)}\n\n"
    text += "Declare each field, then commit aliases in OUT{}:\n"
    text += "\n".join(decl_lines) + "\n\n"
    text += "OUT{" + ", ".join(out_entries) + "}\n"
    return text


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class LndlFormatter:
    """LNDL formatter implementing the Formatter protocol.

    Usage:
        branch.operate(
            instruction="...",
            response_format=AnalysisReport,
            lndl=True,
        )
    """

    @staticmethod
    def render_schema(format: Any) -> str | None:
        return _referenced_schemas_display(format)

    @staticmethod
    def render_format(format: Any, has_tools: bool = False) -> str:
        # The full LNDL syntax/rules live in the system prompt (auto-injected
        # by operate when ``lndl=True``). The per-call instruction only
        # carries the schema — keeping it terse and stable across rounds.
        return _render_lndl_response_structure(format, has_tools=has_tools)

    @staticmethod
    def render_tools(tool_schemas: list[dict[str, Any]]) -> str | None:
        return _tool_schemas_display(tool_schemas)

    @staticmethod
    def parse(
        text: str,
        response_format: Any,
        fuzzy_match_params: FuzzyMatchKeysParams | dict | None = None,
    ) -> Any:
        """Parse LNDL-tagged text into a dict suitable for model_validate.

        Values may include ``ActionCall`` placeholders for <lact> aliases.
        Caller is responsible for executing actions and replacing placeholders.
        """
        text = normalize_lndl_text(text)
        try:
            lexer = Lexer(text)
            tokens = lexer.tokenize()
            parser = Parser(tokens, source_text=text)
            program = parser.parse()
        except Exception as e:
            raise ValueError(f"Failed to parse LNDL: {e}") from e

        if not program.lvars and not program.lacts and not program.out_block:
            raise ValueError("No LNDL tags or OUT{} block found in response")

        target = response_format
        if isinstance(target, BaseModel):
            target = type(target)

        output = assemble(program, target)

        # If response_format is a dict spec, return the assembled dict directly
        if isinstance(response_format, dict):
            return output

        # If output has ActionCall placeholders anywhere, return the dict —
        # the caller (operate pipeline) will execute and re-validate.
        if collect_actions(output):
            return output

        if _is_pydantic_model_cls(target):
            try:
                return target.model_validate(output)
            except Exception:
                # Fall back to dict — operate pipeline may still validate
                # against an extended (operative) response model.
                return output

        return output
