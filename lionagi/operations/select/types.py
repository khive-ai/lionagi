# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, JsonValue

from lionagi import Operation
from lionagi.ln._async_call import AlcallParams
from lionagi.ln.fuzzy import FuzzyMatchKeysParams
from lionagi.ln.types import DataClass
from lionagi.protocols.action.tool import ToolRef
from lionagi.protocols.types import ID, Instruction, SenderRecipient
from lionagi.service.imodel import iModel
from lionagi.session.branch import Branch

HandleValidation = Literal["raise", "return_value", "return_none"]


class BaseiModelContext:
    imodel: iModel = None
    imodel_kw: dict = None


@dataclass(slots=True)
class ChatContext(DataClass):
    """Context for an instruction in a chat."""

    _none_as_sentinel: ClassVar[bool] = True

    instruction: JsonValue | Instruction = None
    guidance: JsonValue = None
    context: JsonValue = None
    response_format: type[BaseModel] | dict = None
    sender: SenderRecipient = None
    recipient: SenderRecipient = None
    progression: ID.RefSeq = None
    tool_schemas: list[dict] = None
    images: list = None
    image_detail: Literal["low", "high", "auto"] = None
    plain_content: str = None
    include_token_usage_to_model: bool = False


@dataclass(slots=True)
class ParseContext(DataClass):
    _none_as_sentinel: ClassVar[bool] = True

    fuzzy_match_params: FuzzyMatchKeysParams | dict = None
    handle_validation: HandleValidation = "raise"
    alcall_params: AlcallParams | dict = None
    imodel: iModel = None
    imodel_kw: dict = None


@dataclass(slots=True)
class InterpretContext(DataClass):
    _none_as_sentinel: ClassVar[bool] = True

    instruction: JsonValue = None
    guidance: JsonValue = None
    context: JsonValue = None
    domain: str = None
    style: str = None
    sample_writing: str = None
    imodel: iModel = None
    imodel_kw: dict = None


@dataclass(slots=True)
class ActionContext(DataClass):
    _none_as_sentinel: ClassVar[bool] = True
    action_call_params: AlcallParams = None
    tools: ToolRef = None
    strategy: Literal["concurrent", "sequential"] = "concurrent"
    suppress_errors: bool = True
    verbose_action: bool = False


@dataclass(slots=True)
class Context(DataClass):
    """A unified context object that can hold chat, parse, action, and interpret contexts."""

    chat: ChatContext = None
    parse: ParseContext = None
    action: ActionContext = None
    interpret: InterpretContext = None
    clear_messages: bool = False
    handle_validation: HandleValidation = "raise"


class Morphism:

    @classmethod
    async def pre(
        cls, branch: Branch, operation: Operation, ctx: Context
    ) -> Any: ...

    @classmethod
    @abstractmethod
    async def apply(
        cls, branch: Branch, operation: Operation, ctx: Context
    ) -> Any: ...

    @classmethod
    async def post(
        cls, branch: Branch, operation: Operation, ctx: Context, result: Any
    ) -> Any: ...

    ...


async def operate_interface(branch: Branch, params: dict, ctx: Context): ...


async def xxxxx(branch, params: dict, ctx: Context): ...
