# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from pydantic import BaseModel, JsonValue, field_serializer
from typing_extensions import override

from lionagi.utils import UNDEFINED, breakdown_pydantic_annotation, copy

from .base import MessageRole
from .message import RoledMessage, SenderRecipient



class Instruction(RoledMessage):
    """
    A user-facing message that conveys commands or tasks. It supports
    optional images, tool references, and schema-based requests.
    """

    @classmethod
    def create(
        cls,
        instruction: JsonValue = None,
        *,
        context: JsonValue = None,
        guidance: JsonValue = None,
        images: list = None,
        sender: SenderRecipient = None,
        recipient: SenderRecipient = None,
        request_fields: JsonValue = None,
        plain_content: JsonValue = None,
        image_detail: Literal["low", "high", "auto"] = None,
        request_model: BaseModel | type[BaseModel] = None,
        response_format: BaseModel | type[BaseModel] = None,
        tool_schemas: list[dict] = None,
    ) -> "Instruction":
        """
        Construct a new Instruction.

        Args:
            instruction (JsonValue, optional):
                The main user instruction.
            context (JsonValue, optional):
                Additional context or environment info.
            guidance (JsonValue, optional):
                Guidance or disclaimers for the instruction.
            images (list, optional):
                A set of images relevant to the instruction.
            request_fields (JsonValue, optional):
                The fields the user wants in the assistant's response.
            plain_content (JsonValue, optional):
                A raw plain text fallback.
            image_detail ("low"|"high"|"auto", optional):
                The detail level for included images.
            request_model (BaseModel|type[BaseModel], optional):
                A Pydantic schema for the request.
            response_format (BaseModel|type[BaseModel], optional):
                Alias for request_model.
            tool_schemas (list[dict] | dict, optional):
                Extra tool reference data.
            sender (SenderRecipient, optional):
                The sender role or ID.
            recipient (SenderRecipient, optional):
                The recipient role or ID.

        Returns:
            Instruction: A newly created instruction object.

        Raises:
            ValueError: If more than one of `request_fields`, `request_model`,
                or `response_format` is passed at once.
        """
        if (
            sum(
                bool(i)
                for i in [request_fields, request_model, response_format]
            )
            > 1
        ):
            raise ValueError(
                "only one of request_fields or request_model can be provided"
                "response_format is alias of request_model"
            )
        content = prepare_instruction_content(
            guidance=guidance,
            instruction=instruction,
            context=context,
            request_fields=request_fields,
            plain_content=plain_content,
            request_model=request_model or response_format,
            images=images,
            image_detail=image_detail,
            tool_schemas=tool_schemas,
        )
        return cls(
            role=MessageRole.USER,
            content=content,
            sender=sender,
            recipient=recipient,
        )

    @property
    def guidance(self) -> str | None:
        return self.content.get("guidance", None)

    @guidance.setter
    def guidance(self, guidance: str) -> None:
        if guidance is None:
            self.content.pop("guidance", None)
        else:
            self.content["guidance"] = str(guidance)

    @property
    def instruction(self) -> JsonValue | None:
        if "plain_content" in self.content:
            return self.content["plain_content"]
        return self.content.get("instruction", None)

    @instruction.setter
    def instruction(self, val: JsonValue) -> None:
        if val is None:
            self.content.pop("instruction", None)
        else:
            self.content["instruction"] = val

    @property
    def context(self) -> JsonValue | None:
        return self.content.get("context", None)

    @context.setter
    def context(self, ctx: JsonValue) -> None:
        if ctx is None:
            self.content["context"] = []
        else:
            self.content["context"] = (
                list(ctx) if isinstance(ctx, list) else [ctx]
            )

    @property
    def tool_schemas(self) -> JsonValue | None:
        return self.content.get("tool_schemas", None)

    @tool_schemas.setter
    def tool_schemas(self, val: list[dict] | dict) -> None:
        if not val:
            self.content.pop("tool_schemas", None)
            return
        self.content["tool_schemas"] = val

    @property
    def plain_content(self) -> str | None:
        return self.content.get("plain_content", None)

    @plain_content.setter
    def plain_content(self, pc: str) -> None:
        self.content["plain_content"] = pc

    @property
    def image_detail(self) -> Literal["low", "high", "auto"] | None:
        return self.content.get("image_detail", None)

    @image_detail.setter
    def image_detail(self, detail: Literal["low", "high", "auto"]) -> None:
        self.content["image_detail"] = detail

    @property
    def images(self) -> list:
        return self.content.get("images", [])

    @images.setter
    def images(self, imgs: list) -> None:
        self.content["images"] = imgs if isinstance(imgs, list) else [imgs]

    @property
    def request_fields(self) -> dict | None:
        return self.content.get("request_fields", None)

    @request_fields.setter
    def request_fields(self, fields: dict) -> None:
        self.content["request_fields"] = fields
        self.content["request_response_format"] = (
            prepare_request_response_format(fields)
        )

    @property
    def response_format(self) -> type[BaseModel] | None:
        return self.content.get("request_model", None)

    @response_format.setter
    def response_format(self, model: type[BaseModel]) -> None:
        if isinstance(model, BaseModel):
            self.content["request_model"] = type(model)
        else:
            self.content["request_model"] = model

        self.request_fields = {}
        self.extend_context(respond_schema_info=model.model_json_schema())
        self.request_fields = breakdown_pydantic_annotation(model)

    @property
    def respond_schema_info(self) -> dict | None:
        return self.content.get("respond_schema_info", None)

    @respond_schema_info.setter
    def respond_schema_info(self, info: dict) -> None:
        if info is None:
            self.content.pop("respond_schema_info", None)
        else:
            self.content["respond_schema_info"] = info

    @property
    def request_response_format(self) -> str | None:
        return self.content.get("request_response_format", None)

    @request_response_format.setter
    def request_response_format(self, val: str) -> None:
        if not val:
            self.content.pop("request_response_format", None)
        else:
            self.content["request_response_format"] = val

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

    def update(
        self,
        *,
        guidance: JsonValue = None,
        instruction: JsonValue = None,
        context: JsonValue = None,
        request_fields: JsonValue = None,
        plain_content: JsonValue = None,
        request_model: BaseModel | type[BaseModel] = None,
        response_format: BaseModel | type[BaseModel] = None,
        images: str | list = None,
        image_detail: Literal["low", "high", "auto"] = None,
        tool_schemas: dict = None,
        sender: SenderRecipient = None,
        recipient: SenderRecipient = None,
    ):
        """
        Batch-update this Instruction.

        Args:
            guidance (JsonValue): New guidance text.
            instruction (JsonValue): Main user instruction.
            request_fields (JsonValue): Updated request fields.
            plain_content (JsonValue): Plain text fallback.
            request_model (BaseModel|type[BaseModel]): Pydantic schema model.
            response_format (BaseModel|type[BaseModel]): Alias for request_model.
            images (list|str): Additional images to add.
            image_detail ("low"|"high"|"auto"): Image detail level.
            tool_schemas (dict): New tool schemas.
            sender (SenderRecipient): New sender.
            recipient (SenderRecipient): New recipient.

        Raises:
            ValueError: If request_model and request_fields are both set.
        """
        if response_format and request_model:
            raise ValueError(
                "only one of request_fields or request_model can be provided"
                "response_format is alias of request_model"
            )

        request_model = request_model or response_format

        if request_model and request_fields:
            raise ValueError(
                "You cannot pass both request_model and request_fields "
                "to create_instruction"
            )
        if guidance:
            self.guidance = guidance

        if instruction:
            self.instruction = instruction

        if plain_content:
            self.plain_content = plain_content

        if request_fields:
            self.request_fields = request_fields

        if request_model:
            self.response_format = request_model

        if images:
            self.images = images

        if image_detail:
            self.image_detail = image_detail

        if tool_schemas:
            self.tool_schemas = tool_schemas

        if sender:
            self.sender = sender

        if recipient:
            self.recipient = recipient

        if context:
            self.extend_context(context)

    @override
    @property
    def rendered(self) -> Any:
        """
        Convert content into a text or combined text+image structure.

        Returns:
            If no images are included, returns a single text block.
            Otherwise returns an array of text + image dicts.
        """
        content = copy(self.content)
        text_content = format_text_content(content)
        if "images" not in content:
            return text_content

        else:
            return format_image_content(
                text_content=text_content,
                images=self.images,
                image_detail=self.image_detail,
            )

    @field_serializer("content")
    def _serialize_content(self, values) -> dict:
        """
        Remove certain ephemeral fields before saving.

        Returns:
            dict: The sanitized content dictionary.
        """
        values.pop("request_model", None)
        values.pop("request_fields", None)

        return values


# File: lionagi/protocols/messages/instruction.py
