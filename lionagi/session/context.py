# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, JsonValue
from typing_extensions import TypedDict

from lionagi import Operation
from lionagi.fields import instruct
from lionagi.ln._async_call import AlcallParams
from lionagi.ln.fuzzy import FuzzyMatchKeysParams
from lionagi.ln.types import DataClass, Params
from lionagi.protocols.action.tool import ToolRef
from lionagi.protocols.types import ID, Instruction, SenderRecipient
from lionagi.service.imodel import iModel

HandleValidation = Literal["raise", "return_value", "return_none"]


class Invariant(ABC):
    pass


class Morphism:
    """Transformations, represents the process of operation execution."""

    async def pre(): ...

    async def post(): ...

    async def perform(): ...

    ...


@dataclass(slots=True, frozen=True, init=False)
class iModelParams(Params, Invariant):
    _none_as_sentinel: ClassVar[bool] = True
    imodel: iModel = None
    imodel_kw: dict = None


@dataclass(slots=True)
class InstructContext(DataClass):
    instruction: JsonValue | Instruction = None
    guidance: JsonValue = None
    context: JsonValue = None
    response_format: type[BaseModel] | dict = None


@dataclass(slots=True)
class ChatContext(InstructContext):
    """Context for an instruction in a chat."""

    sender: SenderRecipient = None
    recipient: SenderRecipient = None
    progression: ID.RefSeq = None
    tool_schemas: list[dict] = None
    images: list = None
    image_detail: Literal["low", "high", "auto"] = None
    plain_content: str = None
    include_token_usage_to_model: bool = False


@dataclass(slots=True)
class ParseContext(iModelContext, InstructContext):
    """Context for parsing a text into a structured format."""

    fuzzy_match_params: FuzzyMatchKeysParams | dict = None
    handle_validation: HandleValidation = "raise"
    alcall_params: AlcallParams | dict = None


@dataclass(slots=True)
class InterpretContext(iModelContext, InstructContext):
    domain: str = None
    style: str = None
    sample_writing: str = None
