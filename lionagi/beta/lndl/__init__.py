# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""LNDL — Language Network Directive Language.

Structured output format for LLM responses with fuzzy parsing tolerance.
Replaces provider-specific tool schemas (OpenAI/Anthropic) with a universal
format that supports mixed reasoning + structured data + action invocations.

Usage:
    from lionagi.ln.types import Spec, Operable
    from lionagi.beta.lndl import parse_lndl_fuzzy

    operable = Operable([Spec(MyModel, name="result")])
    output = parse_lndl_fuzzy(llm_response, operable)
"""

from .ast import Identifier, Lact, Literal, Lvar, OutBlock, Program, RLvar
from .errors import (
    AmbiguousMatchError,
    InvalidConstructorError,
    LNDLError,
    MissingFieldError,
    MissingLvarError,
    MissingOutBlockError,
    TypeMismatchError,
)
from .fuzzy import normalize_lndl_text, parse_lndl_fuzzy
from .lexer import Lexer, Token, TokenType
from .parser import ParseError, Parser
from .prompt import LNDL_SYSTEM_PROMPT, get_lndl_system_prompt
from .resolver import parse_lndl, resolve_references_prefixed
from .types import (
    ActionCall,
    LactMetadata,
    LNDLOutput,
    LvarMetadata,
    ParsedConstructor,
    RLvarMetadata,
    Scalar,
    ensure_no_action_calls,
    has_action_calls,
    revalidate_with_action_results,
)

__all__ = (
    "LNDL_SYSTEM_PROMPT",
    "ActionCall",
    "AmbiguousMatchError",
    "Identifier",
    "InvalidConstructorError",
    "LNDLError",
    "LNDLOutput",
    "Lact",
    "LactMetadata",
    "Lexer",
    "Literal",
    "Lvar",
    "LvarMetadata",
    "MissingFieldError",
    "MissingLvarError",
    "MissingOutBlockError",
    "OutBlock",
    "ParseError",
    "ParsedConstructor",
    "Parser",
    "Program",
    "RLvar",
    "RLvarMetadata",
    "Scalar",
    "Token",
    "TokenType",
    "TypeMismatchError",
    "ensure_no_action_calls",
    "get_lndl_system_prompt",
    "has_action_calls",
    "normalize_lndl_text",
    "parse_lndl",
    "parse_lndl_fuzzy",
    "resolve_references_prefixed",
    "revalidate_with_action_results",
)
