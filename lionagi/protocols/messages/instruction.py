from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, field_validator

from lionagi.ln.types import ModelConfig

from ._helpers import Formatter, JsonFormatter
from .message import Message, MessageContent, MessageRole


@dataclass(slots=True)
class InstructionContent(MessageContent):
    """Structured content for user instructions.

    Fields:
        instruction: Main instruction text
        guidance: Optional guidance or disclaimers
        prompt_context: Additional context items for the prompt (list)
        plain_content: Raw text fallback (bypasses structured rendering)
        tool_schemas: Tool specifications for the assistant
        response_format: User's desired response format (BaseModel class, instance, or dict)
        formatter: Renderer/parser for the response format (default: JsonFormatter)
        images: Image URLs, data URLs, or base64 strings
        image_detail: Detail level for image processing
    """

    _config: ClassVar[ModelConfig] = ModelConfig(
        none_as_sentinel=True,
        serialize_exclude=frozenset({"response_format", "formatter"}),
    )

    instruction: str | None = None
    guidance: str | None = None
    prompt_context: list[Any] = field(default_factory=list)
    plain_content: str | None = None
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    response_format: type[BaseModel] | dict[str, Any] | BaseModel | None = None
    formatter: type[Formatter] = field(default=None, repr=False)
    images: list[str] = field(default_factory=list)
    image_detail: Literal["low", "high", "auto"] | None = None

    def __init__(
        self,
        instruction: str | None = None,
        guidance: str | None = None,
        prompt_context: list[Any] | None = None,
        context: list[Any] | None = None,  # backwards compat
        plain_content: str | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
        response_format: type[BaseModel] | dict[str, Any] | BaseModel | None = None,
        formatter: type[Formatter] | None = None,
        images: list[str] | None = None,
        image_detail: Literal["low", "high", "auto"] | None = None,
    ):
        if context is not None and prompt_context is None:
            prompt_context = context

        if response_format is not None and formatter is None:
            from ._helpers import JsonFormatter

            formatter = JsonFormatter

        data = {
            "instruction": instruction,
            "guidance": guidance,
            "prompt_context": prompt_context if prompt_context is not None else [],
            "plain_content": plain_content,
            "tool_schemas": tool_schemas if tool_schemas is not None else [],
            "response_format": response_format,
            "formatter": formatter,
            "images": images if images is not None else [],
            "image_detail": image_detail,
        }

        for key, value in data.items():
            object.__setattr__(self, key, value)

    @property
    def context(self) -> list[Any]:
        """Backwards compatibility accessor for prompt_context."""
        return self.prompt_context

    @property
    def response_model_cls(self) -> type[BaseModel] | None:
        """DEPRECATED: Will be removed in v1.0. Derive from response_format directly."""
        import warnings

        warnings.warn(
            "InstructionContent.response_model_cls is deprecated and will be removed in v1.0. "
            "Use response_format directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        rf = self.response_format
        if isinstance(rf, type) and issubclass(rf, BaseModel):
            return rf
        if isinstance(rf, BaseModel):
            return type(rf)
        return None

    @property
    def request_model(self) -> type[BaseModel] | None:
        """DEPRECATED: Will be removed in v1.0. Use response_format directly."""
        import warnings

        warnings.warn(
            "InstructionContent.request_model is deprecated and will be removed in v1.0. "
            "Use response_format directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.response_model_cls

    @property
    def schema_dict(self) -> dict[str, Any] | None:
        """DEPRECATED: Will be removed in v1.0. Use formatter.render() instead."""
        import warnings

        from lionagi.libs.schema.breakdown_pydantic_annotation import (
            breakdown_pydantic_annotation,
        )

        warnings.warn(
            "InstructionContent.schema_dict is deprecated and will be removed in v1.0. "
            "Use formatter.render(response_format) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        rf = self.response_format
        if rf is None:
            return None
        if isinstance(rf, dict):
            return rf
        if isinstance(rf, BaseModel):
            return rf.model_dump(mode="json", exclude_none=True)
        if isinstance(rf, type) and issubclass(rf, BaseModel):
            return breakdown_pydantic_annotation(rf)
        return None

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
        inst = cls()

        for k in ("instruction", "guidance", "plain_content", "image_detail"):
            if k in data and data[k]:
                setattr(inst, k, data[k])

        handle_context = data.get("handle_context", "extend")
        if handle_context not in {"extend", "replace"}:
            raise ValueError("handle_context must be either 'extend' or 'replace'")

        ctx_key = "context" if "context" in data else "prompt_context"
        if ctx_key in data:
            ctx = data.get(ctx_key)
            if ctx is None:
                ctx_list: list[Any] = []
            elif isinstance(ctx, list):
                ctx_list = list(ctx)
            else:
                ctx_list = [ctx]
            if handle_context == "replace":
                inst.prompt_context = list(ctx_list)
            else:
                inst.prompt_context.extend(ctx_list)

        if ts := data.get("tool_schemas"):
            inst.tool_schemas.extend(ts if isinstance(ts, list) else [ts])

        if "images" in data:
            imgs = data.get("images") or []
            imgs_list = imgs if isinstance(imgs, list) else [imgs]
            inst.images.extend(imgs_list)
            inst.image_detail = data.get("image_detail") or inst.image_detail or "auto"

        response_format = data.get("response_format") or data.get("request_model")
        if response_format is not None:
            valid = (
                isinstance(response_format, dict)
                or isinstance(response_format, BaseModel)
                or (
                    isinstance(response_format, type)
                    and issubclass(response_format, BaseModel)
                )
            )
            if valid:
                inst.response_format = response_format
        if pt := data.get("formatter"):
            inst.formatter = pt
        elif inst.response_format is not None and inst.formatter is None:
            from ._helpers import JsonFormatter

            inst.formatter = JsonFormatter
        return inst

    def _format_text_content(self) -> str:
        from lionagi.libs.schema.minimal_yaml import minimal_yaml

        if self.plain_content:
            return self.plain_content

        parts: list[str] = []
        tf = self.formatter

        if self.guidance:
            parts.append(f"## Guidance\n{self.guidance}")

        if self.instruction:
            parts.append(f"## Task Instruction\n{self.instruction}")

        if self.prompt_context:
            ctx_yaml = minimal_yaml(self.prompt_context).strip()
            parts.append(f"## Context\n{ctx_yaml}")

        if self.tool_schemas:
            from ._helpers import _tool_schemas_display

            if all(isinstance(t, str) for t in self.tool_schemas):
                parts.append("## Tools\n" + "\n".join(self.tool_schemas))
            else:
                tools_display = _tool_schemas_display(self.tool_schemas)
                parts.append(
                    f"## Tools\n{tools_display or minimal_yaml(self.tool_schemas).strip()}"
                )

        has_tools = bool(self.tool_schemas)

        def _render(fmt):
            try:
                return tf.render_format(fmt, has_tools=has_tools)
            except TypeError:
                # Back-compat: custom formatters that don't accept has_tools
                return tf.render_format(fmt)

        if not self._is_sentinel(self.response_format):
            schema = tf.render_schema(self.response_format)
            if schema:
                parts.append(f"## Schema\n{schema}")
            parts.append(f"## ResponseFormat\n{_render(self.response_format)}")
        elif tf is not None and tf is not JsonFormatter:
            parts.append(f"## ResponseFormat\n{_render(None)}")

        return "\n\n".join(parts)

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


class Instruction(Message):
    """User instruction message with structured content.

    Supports text, images, context, tool schemas, and response format specifications.
    """

    _role: ClassVar[MessageRole] = MessageRole.USER
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
