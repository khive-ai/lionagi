# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""LNDL — Language Network Directive Language.

Structured output format for LLM responses. Tags allow models to mix
natural reasoning with structured data.

Core (no external deps beyond lionagi):
    lexer, parser, ast, resolver, extract, types, errors, prompt
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
from .extract import extract_lndl_blocks
from .lexer import Lexer, Token, TokenType
from .parser import ParseError, Parser
from .prompt import LNDL_SYSTEM_PROMPT, get_lndl_system_prompt
from .types import ActionCall, LNDLOutput, LvarMetadata, RLvarMetadata, Scalar

__all__ = (
    "LNDL_SYSTEM_PROMPT",
    "ActionCall",
    "AmbiguousMatchError",
    "Identifier",
    "InvalidConstructorError",
    "LNDLError",
    "LNDLOutput",
    "Lact",
    "Lexer",
    "Literal",
    "Lvar",
    "LvarMetadata",
    "MissingFieldError",
    "MissingLvarError",
    "MissingOutBlockError",
    "OutBlock",
    "ParseError",
    "Parser",
    "Program",
    "RLvar",
    "RLvarMetadata",
    "Scalar",
    "Token",
    "TokenType",
    "extract_lndl_blocks",
    "get_lndl_system_prompt",
)
