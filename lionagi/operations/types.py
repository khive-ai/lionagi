# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import BaseModel, JsonValue

from lionagi.ln._async_call import AlcallParams
from lionagi.ln.fuzzy import FuzzyMatchKeysParams
from lionagi.ln.types import DataClass
from lionagi.protocols.action.tool import ToolRef
from lionagi.protocols.types import ID, Instruction, SenderRecipient
from lionagi.service.imodel import iModel

HandleValidation = Literal["raise", "return_value", "return_none"]


@dataclass(slots=True)
class ChatContext(DataClass):
    """Context for an instruction in a chat."""

    _none_as_sentinel: ClassVar[bool] = True
    guidance: JsonValue
    context: JsonValue
    sender: SenderRecipient
    recipient: SenderRecipient
    response_format: type[BaseModel] | dict
    progression: ID.RefSeq
    tool_schemas: list[dict]
    images: list
    image_detail: Literal["low", "high", "auto"]
    plain_content: str
    include_token_usage_to_model: bool
    imodel: iModel
    imodel_kw: dict


@dataclass(slots=True)
class InterpretContext(DataClass):
    _none_as_sentinel: ClassVar[bool] = True
    domain: str
    style: str
    sample_writing: str
    imodel: iModel
    imodel_kw: dict


@dataclass(slots=True)
class ParseContext(DataClass):
    _none_as_sentinel: ClassVar[bool] = True
    response_format: type[BaseModel] | dict
    fuzzy_match_params: FuzzyMatchKeysParams | dict
    handle_validation: HandleValidation
    alcall_params: AlcallParams | dict
    imodel: iModel
    imodel_kw: dict


@dataclass(slots=True)
class ActionContext(DataClass):
    _none_as_sentinel: ClassVar[bool] = True
    action_call_params: AlcallParams
    tools: ToolRef
    strategy: Literal["concurrent", "sequential"] = "concurrent"
    suppress_errors: bool = True
    verbose_action: bool = False
