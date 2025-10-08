import inspect
from dataclasses import dataclass, field
from typing import Any, Literal

import orjson
from pydantic import BaseModel, field_validator

from .message import MessageContent, MessageRole, RoledMessage


@dataclass(slots=True)
class InstructionContent(MessageContent):
    """Structured content for user instructions.

    Fields:
        instruction: Main instruction text
        guidance: Optional guidance or disclaimers
        context: Additional context items (list)
        plain_content: Raw text fallback (bypasses structured rendering)
        tool_schemas: Tool specifications for the assistant
        response_format: Example JSON payload for expected response
        response_schema: JSON Schema for response validation
        images: Image URLs, data URLs, or base64 strings
        image_detail: Detail level for image processing
    """

    instruction: str | None = None
    guidance: str | None = None
    context: list[Any] = field(default_factory=list)
    plain_content: str | None = None
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    response_format: dict[str, Any] | None = None
    response_schema: dict[str, Any] | None = None
    images: list[str] = field(default_factory=list)
    image_detail: Literal["low", "high", "auto"] | None = None

    @property
    def rendered(self) -> str | list[dict[str, Any]]:
        """Render content as text or text+images structure."""
        text = self._format_text_content()
        if not self.images:
            return text
        return self._format_image_content(text, self.images, self.image_detail)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InstructionContent":
        """Construct InstructionContent from dictionary with validation."""
        from lionagi.libs.schema.breakdown_pydantic_annotation import (
            breakdown_pydantic_annotation,
        )

        inst = cls()

        # Scalar fields
        for k in ("instruction", "guidance", "plain_content", "image_detail"):
            if k in data and data[k]:
                setattr(inst, k, data[k])

        # Determine how to apply context updates
        handle_context = data.get("handle_context", "extend")
        if handle_context not in {"extend", "replace"}:
            raise ValueError(
                "handle_context must be either 'extend' or 'replace'"
            )

        if "context" in data:
            ctx = data.get("context")
            if ctx is None:
                ctx_list: list[Any] = []
            elif isinstance(ctx, list):
                ctx_list = list(ctx)
            else:
                ctx_list = [ctx]
            if handle_context == "replace":
                inst.context = list(ctx_list)
            else:
                inst.context.extend(ctx_list)

        if ts := data.get("tool_schemas"):
            inst.tool_schemas.extend(ts if isinstance(ts, list) else [ts])

        if "images" in data:
            imgs = data.get("images") or []
            imgs_list = imgs if isinstance(imgs, list) else [imgs]
            inst.images.extend(imgs_list)
            inst.image_detail = (
                data.get("image_detail") or inst.image_detail or "auto"
            )

        # Response schema and format
        # request_model is an alias for response_schema
        schema_source = data.get("response_schema") or data.get(
            "request_model"
        )
        response_format_input = data.get("response_format")

        schema_instance: BaseModel | None = None
        schema_cls: type[BaseModel] | None = None
        response_format_value: dict[str, Any] | None = None

        if response_format_input is not None and not isinstance(
            response_format_input, dict
        ):
            if isinstance(response_format_input, BaseModel):
                schema_instance = response_format_input
                schema_cls = type(response_format_input)
                if schema_source is None:
                    schema_source = schema_cls
            elif inspect.isclass(response_format_input) and issubclass(
                response_format_input, BaseModel
            ):
                schema_cls = response_format_input
                if schema_source is None:
                    schema_source = schema_cls
            else:
                raise TypeError(
                    "response_format must be dict, BaseModel instance, or BaseModel class"
                )
        elif isinstance(response_format_input, dict):
            response_format_value = response_format_input

        if schema_source is not None:
            if isinstance(schema_source, BaseModel):
                schema_instance = schema_instance or schema_source
                schema_cls = type(schema_source)
                inst.response_schema = schema_source.model_json_schema()
            elif inspect.isclass(schema_source) and issubclass(
                schema_source, BaseModel
            ):
                schema_cls = schema_source
                inst.response_schema = schema_source.model_json_schema()
            elif isinstance(schema_source, dict):
                inst.response_schema = schema_source
            else:
                raise TypeError(
                    "response_schema must be dict, BaseModel instance, or BaseModel class"
                )
        elif schema_instance is not None:
            inst.response_schema = schema_instance.model_json_schema()

        if response_format_value is None:
            if schema_instance is not None:
                response_format_value = schema_instance.model_dump(
                    mode="json", exclude_none=True
                )
            elif schema_cls is not None:
                response_format_value = breakdown_pydantic_annotation(
                    schema_cls
                )

        if response_format_value is not None:
            inst.response_format = response_format_value

        return inst

    def _format_text_content(self) -> str:
        from lionagi.libs.schema.minimal_yaml import minimal_yaml

        if self.plain_content:
            return self.plain_content

        doc: dict[str, Any] = {
            "Guidance": self.guidance,
            "Instruction": self.instruction,
            "Context": self.context,
            "Tools": self.tool_schemas,
            "ResponseSchema": self.response_schema,
        }

        rf_text = self._format_response_format(self.response_format)
        if rf_text:
            doc["ResponseFormat"] = rf_text

        # strip empties
        doc = {k: v for k, v in doc.items() if v not in (None, "", [], {})}
        return minimal_yaml(doc).strip()

    @staticmethod
    def _format_response_format(
        response_format: dict[str, Any] | None,
    ) -> str | None:
        if not response_format:
            return None
        try:
            example = orjson.dumps(response_format).decode("utf-8")
        except Exception:
            example = str(response_format)
        return (
            "Return a **single JSON code block**.\n"
            "No prose before/after the block. Use exactly these keys.\n\n"
            f"```json\n{example}\n```"
        )

    @staticmethod
    def _format_image_item(idx: str, detail: str) -> dict[str, Any]:
        url = idx
        if not (
            idx.startswith("http://")
            or idx.startswith("https://")
            or idx.startswith("data:")
        ):
            url = f"data:image/jpeg;base64,{idx}"
        return {
            "type": "image_url",
            "image_url": {"url": url, "detail": detail},
        }

    @classmethod
    def _format_image_content(
        cls,
        text_content: str,
        images: list[str],
        image_detail: Literal["low", "high", "auto"],
    ) -> list[dict[str, Any]]:
        content = [{"type": "text", "text": text_content}]
        content.extend(cls._format_image_item(i, image_detail) for i in images)
        return content


class Instruction(RoledMessage):
    """User instruction message with structured content.

    Supports text, images, context, tool schemas, and response format specifications.
    """

    role: MessageRole = MessageRole.USER
    content: InstructionContent

    @field_validator("content", mode="before")
    def _validate_content(cls, v):
        if v is None:
            return InstructionContent()
        if isinstance(v, dict):
            return InstructionContent.from_dict(v)
        if isinstance(v, InstructionContent):
            return v
        raise TypeError("content must be dict or InstructionContent instance")
