from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, field_validator

from lionagi.ln.types import ModelConfig

from ._helpers import Formatter, JsonTransformer
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
        prompt_transformer: Renderer/parser for the response format (default: JsonTransformer)
        images: Image URLs, data URLs, or base64 strings
        image_detail: Detail level for image processing
    """

    _config: ClassVar[ModelConfig] = ModelConfig(
        none_as_sentinel=True,
        serialize_exclude=frozenset({"response_format", "prompt_transformer"}),
    )

    instruction: str | None = None
    guidance: str | None = None
    prompt_context: list[Any] = field(default_factory=list)
    plain_content: str | None = None
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    response_format: type[BaseModel] | dict[str, Any] | BaseModel | None = None
    prompt_transformer: type[Formatter] = field(default=JsonTransformer, repr=False)
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
        prompt_transformer: type[Formatter] | None = None,
        images: list[str] | None = None,
        image_detail: Literal["low", "high", "auto"] | None = None,
    ):
        if context is not None and prompt_context is None:
            prompt_context = context

        object.__setattr__(self, "instruction", instruction)
        object.__setattr__(self, "guidance", guidance)
        object.__setattr__(
            self,
            "prompt_context",
            prompt_context if prompt_context is not None else [],
        )
        object.__setattr__(self, "plain_content", plain_content)
        object.__setattr__(
            self,
            "tool_schemas",
            tool_schemas if tool_schemas is not None else [],
        )
        object.__setattr__(self, "response_format", response_format)
        object.__setattr__(
            self,
            "prompt_transformer",
            prompt_transformer or JsonTransformer,
        )
        object.__setattr__(self, "images", images if images is not None else [])
        object.__setattr__(self, "image_detail", image_detail)

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
        """DEPRECATED: Will be removed in v1.0. Use prompt_transformer.render() instead."""
        import warnings

        from lionagi.libs.schema.breakdown_pydantic_annotation import (
            breakdown_pydantic_annotation,
        )

        warnings.warn(
            "InstructionContent.schema_dict is deprecated and will be removed in v1.0. "
            "Use prompt_transformer.render(response_format) instead.",
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

        if pt := data.get("prompt_transformer"):
            inst.prompt_transformer = pt

        return inst

    def _format_text_content(self) -> str:
        from lionagi.libs.schema.minimal_yaml import minimal_yaml

        if self.plain_content:
            return self.plain_content

        doc: dict[str, Any] = {
            "Guidance": self.guidance,
            "Instruction": self.instruction,
            "Context": self.prompt_context,
            "Tools": self.tool_schemas,
        }

        if not self._is_sentinel(self.response_format):
            doc["ResponseFormat"] = self.prompt_transformer.render(self.response_format)

        doc = {k: v for k, v in doc.items() if v not in (None, "", [], {})}
        return minimal_yaml(doc).strip()

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
