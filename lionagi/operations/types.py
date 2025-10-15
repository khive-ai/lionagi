# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal

from pydantic import BaseModel, JsonValue

from lionagi.ln import AlcallParams
from lionagi.ln.fuzzy import FuzzyMatchKeysParams
from lionagi.ln.types import Params
from lionagi.protocols.action.tool import ToolRef
from lionagi.protocols.types import ID, SenderRecipient
from lionagi.service.imodel import iModel

HandleValidation = Literal["raise", "return_value", "return_none"]


class ContextPolicy(str, Enum):
    """Policy for merging prompt context across morphism invocations.

    Attributes:
        REPLACE: New context completely replaces existing context
        EXTEND: New context is appended to existing context
        DEDUP: New context is appended but duplicates are removed
    """

    REPLACE = "replace"
    EXTEND = "extend"
    DEDUP = "dedup"


@dataclass(slots=True, frozen=True, init=False)
class MorphParam(Params):
    """Base class for morphism parameters (invariants).

    MorphParams represent the invariant properties that define a morphism
    in LionAGI's categorical framework. They are frozen (immutable) and
    hashable, enabling reproducible operations and efficient caching.

    Morphisms are the fundamental abstraction in LionAGI - they represent
    transformations between message states with well-defined parameters.
    """

    _none_as_sentinel: ClassVar[bool] = True


@dataclass(slots=True, frozen=True, init=False)
class ChatParam(MorphParam):
    """Parameters for chat/communicate morphism.

    Defines the invariant properties of a chat operation, including
    guidance, context, response format, and LLM-visible content.

    Note: 'context' field contains prompt context (LLM-visible facts).
    This gets mapped to InstructionContent.prompt_context during message creation.
    """

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


@dataclass(slots=True, frozen=True, init=False)
class InterpretParam(MorphParam):
    """Parameters for interpret morphism.

    Defines interpretation style, domain, and sample writing for
    transforming content according to specified guidelines.
    """

    _none_as_sentinel: ClassVar[bool] = True
    domain: str = None
    style: str = None
    sample_writing: str = None
    imodel: iModel = None
    imodel_kw: dict = None


@dataclass(slots=True, frozen=True, init=False)
class ParseParam(MorphParam):
    """Parameters for parse morphism.

    Defines parsing behavior including response format validation,
    fuzzy matching, and error handling strategies.
    """

    _none_as_sentinel: ClassVar[bool] = True
    response_format: type[BaseModel] | dict = None
    fuzzy_match_params: FuzzyMatchKeysParams | dict = None
    handle_validation: HandleValidation = "raise"
    alcall_params: AlcallParams | dict = None
    imodel: iModel = None
    imodel_kw: dict = None


@dataclass(slots=True, frozen=True, init=False)
class ActionParam(MorphParam):
    """Parameters for action/tool execution morphism.

    Defines tool execution strategy, error handling, and verbosity
    for action-based operations.
    """

    _none_as_sentinel: ClassVar[bool] = True
    action_call_params: AlcallParams = None
    tools: ToolRef = None
    strategy: Literal["concurrent", "sequential"] = "concurrent"
    suppress_errors: bool = True
    verbose_action: bool = False
