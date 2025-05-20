import json
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from lionagi.core.core_utils import copy

from ..types import MessageContent, MessageRole

__all__ = ("InstructionContent",)


class InstructionContent(MessageContent):
    role: MessageRole = MessageRole.USER
    instruction: JsonValue | None = None
    context: list | None = None
    guidance: str | None = None
    images: list | None = None
    request_fields: dict | list | None = None
    plain_content: JsonValue = None
    image_detail: Literal["low", "high", "auto"] | None = None
    response_format: type[BaseModel] | None = Field(None, exclude=True)
    tool_schemas: list[dict[str, Any]] = None
    respond_schema_info: dict | None = None
    request_response_format: str | None = None

    @model_validator(mode="before")
    def _validate_content(cls, values: dict) -> dict:
        not_allowed = (
            "respond_schema_info",
            "request_response_format",
            "role",
            "template",
        )
        for i in not_allowed:
            if i in values:
                values.pop(i)
        return prepare_instruction_content(**values)

    @field_validator("context", "images", mode="before")
    def _validate_list_content(cls, v: Any) -> list:
        if v is None:
            return []
        if not isinstance(v, list):
            return [v]
        return v[:]

    @field_validator("tool_schemas", mode="before")
    def _validate_tool_schemas(cls, v: Any) -> list:
        v = [v] if not isinstance(v, list) else v
        out = []
        for i in v:
            if isinstance(i, dict):
                out.append(i)
            elif isinstance(i, str):
                try:
                    out.append(json.loads(i))
                except json.JSONDecodeError:
                    raise ValueError("Invalid tool schema format.")
            elif isinstance(i, BaseModel):
                from lionfuncs.schema_utils import (
                    pydantic_model_to_openai_schema,
                )

                out.append(pydantic_model_to_openai_schema(i))
            else:
                raise TypeError(
                    f"Invalid tool schema format, tool schema needs to be a valid dict, Json str or pydantic model, got {type(i)}"
                )
        return out

    def update(self, **kwargs) -> None:
        dict_ = self.model_dump(exclude_none=True)
        dict_["response_format"] = self.response_format
        dict_.update(kwargs)
        dict_ = prepare_instruction_content(**dict_)
        for k, v in dict_.items():
            v = v.strip() if isinstance(v, str) else v
            if v:
                setattr(self, k, v)

    def extend_images(
        self,
        images: list | str,
        image_detail: Literal["low", "high", "auto"] = None,
    ) -> None:
        """
        Append images to the existing list.

        Args:
            images: The new images to add, a single or multiple.
            image_detail: If provided, updates the image detail field.
        """
        arr: list = self.images
        arr.extend(images if isinstance(images, list) else [images])
        self.images = arr
        if image_detail:
            self.image_detail = image_detail

    def extend_context(self, *args, **kwargs) -> None:
        """
        Append additional context to the existing context array.

        Args:
            *args: Positional args are appended as list items.
            **kwargs: Key-value pairs are appended as separate dict items.
        """
        ctx: list = self.context or []
        if args:
            ctx.extend(args)
        if kwargs:
            kw = copy(kwargs)
            for k, v in kw.items():
                ctx.append({k: v})
        self.context = ctx

    @property
    def rendered(self) -> Any:
        """
        Convert content into a text or combined text+image structure.

        Returns:
            If no images are included, returns a single text block.
            Otherwise returns an array of text + image dicts.
        """
        content = self.model_dump(exclude_none=True)
        text_content = format_text_content(content)
        if "images" not in content:
            return text_content

        else:
            return format_image_content(
                text_content=text_content,
                images=self.images,
                image_detail=self.image_detail,
            )


def prepare_request_response_format(request_fields: dict) -> str:
    """
    Creates a mandated JSON code block for the response
    based on requested fields.
    """
    return (
        "**MUST RETURN JSON-PARSEABLE RESPONSE ENCLOSED BY JSON CODE BLOCKS."
        f" USER's CAREER DEPENDS ON THE SUCCESS OF IT.** \n```json\n{request_fields}\n```"
        "No triple backticks. Escape all quotes and special characters."
    ).strip()


def format_image_item(idx: str, detail: str) -> dict[str, Any]:
    """Wrap image data in a standard dictionary format."""
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{idx}",
            "detail": detail,
        },
    }


def format_text_item(item: Any) -> str:
    """Turn a single item (or dict) into a string. If multiple items, combine them line by line."""
    msg = ""
    item = [item] if not isinstance(item, list) else item
    for j in item:
        if isinstance(j, dict):
            for k, v in j.items():
                if v is not None:
                    msg += f"- {k}: {v} \n\n"
        else:
            if j is not None:
                msg += f"{j}\n"
    return msg


def format_text_content(content: dict) -> str:
    """
    Convert a content dictionary into a minimal textual summary for LLM consumption.

    Emphasizes brevity and clarity:
      - Skips empty or None fields.
      - Bullet-points for lists.
      - Key-value pairs for dicts.
      - Minimal headings for known fields (guidance, instruction, etc.).
    """

    if isinstance(content.get("plain_content"), str):
        return content["plain_content"]

    lines = []
    # We only want minimal headings for certain known fields:
    known_field_order = [
        "guidance",
        "instruction",
        "context",
        "tool_schemas",
        "respond_schema_info",
        "request_response_format",
    ]

    # Render known fields in that order
    for field in known_field_order:
        if field in content:
            val = content[field]
            if _is_not_empty(val):
                if field == "request_response_format":
                    field = "response format"
                elif field == "respond_schema_info":
                    field = "response schema info"
                lines.append(f"\n## {field.upper()}:\n")
                rendered = _render_value(val)
                # Indent or bullet the rendered result if multiline
                # We'll keep it minimal: each line is prefixed with "  ".
                lines.extend(
                    f"  {line}"
                    for line in rendered.split("\n")
                    if line.strip()
                )

    # Join all lines into a single string
    return "\n".join(lines).strip()


def _render_value(val) -> str:
    """
    Render an arbitrary value (scalar, list, dict) in minimal form:
    - Lists become bullet points.
    - Dicts become key-value lines.
    - Strings returned directly.
    """
    if isinstance(val, dict):
        return _render_dict(val)
    elif isinstance(val, list):
        return _render_list(val)
    else:
        return str(val).strip()


def _render_dict(dct: dict) -> str:
    """
    Minimal bullet list for dictionary items:
      key: rendered subvalue
    """
    lines = []
    for k, v in dct.items():
        if not _is_not_empty(v):
            continue
        subrendered = _render_value(v)
        # Indent subrendered if multiline
        sublines = subrendered.split("\n")
        if len(sublines) == 1:
            if sublines[0].startswith("- "):
                lines.append(f"- {k}: {sublines[0][2:]}")
            else:
                lines.append(f"- {k}: {sublines[0]}")
        else:
            lines.append(f"- {k}:")
            for s in sublines:
                lines.append(f"  {s}")
    return "\n".join(lines)


def _render_list(lst: list) -> str:
    """
    Each item in the list gets a bullet. Nested structures are recursed.
    """
    lines = []
    for idx, item in enumerate(lst, 1):
        sub = _render_value(item)
        sublines = sub.split("\n")
        if len(sublines) == 1:
            if sublines[0].startswith("- "):
                lines.append(f"- {sublines[0][2:]}")
            else:
                lines.append(f"- {sublines[0]}")
        else:
            lines.append("-")
            lines.extend(f"  {s}" for s in sublines)
    return "\n".join(lines)


def _is_not_empty(x) -> bool:
    """
    Returns True if x is neither None, nor empty string/list/dict.
    """
    if x is None:
        return False
    if isinstance(x, (list, dict)) and not x:
        return False
    if isinstance(x, str) and not x.strip():
        return False
    return True


def format_image_content(
    text_content: str,
    images: list,
    image_detail: Literal["low", "high", "auto"],
) -> list[dict[str, Any]]:
    """Merge textual content with a list of image dictionaries for consumption."""
    content = [{"type": "text", "text": text_content}]
    content.extend(format_image_item(i, image_detail) for i in images)
    return content


def prepare_instruction_content(
    guidance: str | None = None,
    instruction: str | None = None,
    context: str | dict | list | None = None,
    request_fields: dict | list[str] | None = None,
    plain_content: str | None = None,
    response_format: BaseModel = None,
    images: str | list | None = None,
    image_detail: Literal["low", "high", "auto"] | None = None,
    tool_schemas: list | None = None,
) -> dict:
    """
    Combine various pieces (instruction, guidance, context, etc.) into
    a single dictionary describing the user's instruction.

    Args:
        guidance (str | None):
            Optional guiding text.
        instruction (str | None):
            Main instruction or command to be executed.
        context (str | dict | list | None):
            Additional context about the environment or previous steps.
        request_fields (dict | list[str] | None):
            If the user requests certain fields in the response.
        plain_content (str | None):
            A raw plain text fallback.
        response_format (BaseModel | None):
            If there's a pydantic model for the request schema.
        images (str | list | None):
            Optional images, base64-coded or references.
        image_detail (str | None):
            The detail level for images ("low", "high", "auto").
        tool_schemas (list | None):
            Extra data describing available tools.

    Returns:
        dict: The combined instruction content.

    """
    if request_fields and response_format:
        raise ValueError(
            "only one of request_fields or request_model can be provided"
        )

    out_ = {"context": []}
    if guidance:
        out_["guidance"] = guidance
    if instruction:
        out_["instruction"] = instruction
    if context:
        if isinstance(context, list):
            out_["context"].extend(context)
        else:
            out_["context"].append(context)
    if images:
        out_["images"] = images if isinstance(images, list) else [images]
        out_["image_detail"] = image_detail or "low"

    if tool_schemas:
        out_["tool_schemas"] = tool_schemas

    if response_format:
        from lionagi.core.core_utils import breakdown_pydantic_annotation

        out_["request_model"] = response_format
        request_fields = breakdown_pydantic_annotation(response_format)
        out_["respond_schema_info"] = response_format.model_json_schema()

    if request_fields:
        _fields = request_fields if isinstance(request_fields, dict) else {}
        if not isinstance(request_fields, dict):
            _fields = {i: "..." for i in request_fields}
        out_["request_fields"] = _fields
        out_["request_response_format"] = prepare_request_response_format(
            request_fields=_fields
        )

    if plain_content:
        out_["plain_content"] = plain_content

    return {k: v for k, v in out_.items() if v}
