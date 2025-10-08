# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import BaseModel, JsonValue

from lionagi.ln._async_call import AlcallParams
from lionagi.ln.fuzzy import FuzzyMatchKeysParams
from lionagi.ln.types import DataClass
from lionagi.protocols.action.tool import ToolRef
from lionagi.protocols.types import ID, SenderRecipient
from lionagi.service.imodel import iModel

HandleValidation = Literal["raise", "return_value", "return_none"]


@dataclass(slots=True)
class ChatContext(DataClass):
    """Context for an instruction in a chat."""

    _none_as_sentinel: ClassVar[bool] = True
    guidance: JsonValue = None
    context: JsonValue = None
    sender: SenderRecipient = None
    recipient: SenderRecipient = None
    response_format: type[BaseModel] | dict = None
    progression: ID.RefSeq = None
    tool_schemas: list[dict] = None
    images: list = None
    image_detail: Literal["low", "high", "auto"] = None
    plain_content: str = None
    include_token_usage_to_model: bool = False
    imodel: iModel = None
    imodel_kw: dict = None


@dataclass(slots=True)
class InterpretContext(DataClass):
    _none_as_sentinel: ClassVar[bool] = True
    domain: str = None
    style: str = None
    sample_writing: str = None
    imodel: iModel = None
    imodel_kw: dict = None


@dataclass(slots=True)
class ParseContext(DataClass):
    _none_as_sentinel: ClassVar[bool] = True
    response_format: type[BaseModel] | dict = None
    fuzzy_match_params: FuzzyMatchKeysParams | dict = None
    handle_validation: HandleValidation = "raise"
    alcall_params: AlcallParams | dict = None
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
