# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, JsonValue

from lionagi.ln.types._sentinel import MaybeUnset, Unset, is_unset
from lionagi.libs.schema import (
    breakdown_pydantic_annotation,
    format_clean_multiline_strings,
    format_model_schema,
    format_schema_pretty,
    is_pydantic_model,
    minimal_yaml,
)

from ._validators import validate_image_url
from .common import CustomRenderer, StructureFormat
from .role import Role, RoledContent


@dataclass(slots=True)
class Instruction(RoledContent):
    role: ClassVar[Role] = Role.USER

    primary: MaybeUnset[str] = Unset
    context: MaybeUnset[list] = Unset
    request_model: MaybeUnset[type[BaseModel]] = Unset
    tool_schemas: MaybeUnset[list[str | dict]] = Unset
    images: MaybeUnset[list[str]] = Unset
    image_detail: MaybeUnset[Literal["low", "high", "auto"]] = Unset
    structure_format: MaybeUnset[StructureFormat] = Unset
    custom_renderer: MaybeUnset[Callable[[type[BaseModel]], str]] = Unset

    @classmethod
    def create(
        cls,
        primary: MaybeUnset[str] = Unset,
        context: MaybeUnset[JsonValue] = Unset,
        tool_schemas: MaybeUnset[list[str | dict]] = Unset,
        request_model: MaybeUnset[type[BaseModel]] = Unset,
        images: MaybeUnset[list[str]] = Unset,
        image_detail: MaybeUnset[Literal["low", "high", "auto"]] = Unset,
        structure_format: MaybeUnset[StructureFormat] = Unset,
        custom_renderer: MaybeUnset[Callable[[type[BaseModel]], str]] = Unset,
    ) -> "Instruction":
        if is_unset(primary) and is_unset(request_model):
            raise ValueError("Either 'primary' or 'request_model' must be provided.")

        if not is_unset(request_model) and not is_pydantic_model(request_model):
            raise ValueError("'request_model' must be a subclass of pydantic BaseModel.")

        if not is_unset(images) and images is not None:
            for url in images:
                validate_image_url(url)

        if not cls._is_sentinel(context):
            context = [context] if not isinstance(context, list) else context

        return cls(
            primary=primary,
            context=context,
            tool_schemas=tool_schemas,
            request_model=request_model,
            images=images,
            image_detail=image_detail,
            structure_format=structure_format,
            custom_renderer=custom_renderer,
        )

    def _format_text_content(
        self,
        structure_format: StructureFormat,
        custom_renderer: MaybeUnset[CustomRenderer],
    ) -> str:
        if structure_format == StructureFormat.CUSTOM and not callable(custom_renderer):
            raise ValueError("Custom renderer must be provided when structure_format is 'custom'.")

        task_data = {
            "Primary Instruction": self.primary,
            "Context": self.context,
            "Tools": self.tool_schemas,
        }
        text = _format_task({k: v for k, v in task_data.items() if not self._is_sentinel(v)})

        if not self._is_sentinel(self.request_model):
            model = self.request_model
            text += format_model_schema(model)

            if structure_format == StructureFormat.CUSTOM:
                text += custom_renderer(model)
            elif structure_format == StructureFormat.LNDL:
                has_tools = not self._is_sentinel(self.tool_schemas) and bool(self.tool_schemas)
                text += _format_lndl_response_structure(model, has_tools=has_tools)
            elif structure_format == StructureFormat.JSON or is_unset(structure_format):
                text += _format_json_response_structure(model)

        return text.strip()

    def _format_image_content(self, text: str) -> list[dict[str, Any]]:
        """Build multimodal content list with text and images."""
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        detail = self.image_detail if not self._is_sentinel(self.image_detail) else "auto"
        for url in self.images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url, "detail": detail},
                }
            )
        return content

    def render(
        self,
        structure_format: MaybeUnset[StructureFormat] = Unset,
        custom_renderer: MaybeUnset[CustomRenderer] = Unset,
    ) -> str | list[dict[str, Any]]:
        structure_format = self.structure_format if is_unset(structure_format) else structure_format
        custom_renderer = self.custom_renderer if is_unset(custom_renderer) else custom_renderer
        text = self._format_text_content(structure_format, custom_renderer)
        return text if is_unset(self.images) else self._format_image_content(text)


def _format_json_response_structure(request_model: type[BaseModel]) -> str:
    """Format response structure with Python types (unquoted)."""
    schema = breakdown_pydantic_annotation(
        request_model,
        max_depth=50,
        clean_types=True,
    )
    json_schema = "\n\n## ResponseFormat\n"
    json_schema += "```json\n"
    json_schema += format_schema_pretty(schema, indent=0)
    json_schema += "\n```\nMUST RETURN VALID JSON. USER's SUCCESS DEPENDS ON IT. Return ONLY valid JSON without markdown code blocks.\n"
    return json_schema


def _format_lndl_response_structure(request_model: type[BaseModel], has_tools: bool = False) -> str:
    """Format response structure as a pedagogical LNDL guide.

    Schema-driven: emits specs line + typed field list + numbered instructions +
    concrete example pattern + notes. The model chooses its own aliases — fuzzy
    matching handles the rest. Goal: caller passes only the question, framework
    generates the full LNDL pattern.
    """
    fields = request_model.model_fields

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

    text = "\n\n## ResponseFormat (LNDL)\n"
    text += f"Specs: {specs_line}\n\n"

    # Build a concrete example using actual spec/field names
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

    for spec_name in scalar_specs:
        out_parts.append(f"{spec_name}: <value>")

    text += "Declare each field, then reference aliases in OUT{}:\n"
    text += "\n".join(example_lines) + "\n" if example_lines else ""
    text += f"\nOUT{{{', '.join(out_parts)}}}\n"
    if has_tools:
        text += '\nTool calls: use `<lact Model.field alias>tool(arg="val")</lact>` '
        text += "with keyword args. Result fills the field.\n"
    return text


def _format_task(task_data: dict) -> str:
    text = "## Task\n"
    text += minimal_yaml(format_clean_multiline_strings(task_data))
    return text
