# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""LNDL — Language Network Directive Language.

Structured output format for LLM responses. Tags allow models to mix
natural reasoning with structured data.

Core (no external deps beyond lionagi):
    lexer, parser, ast, resolver, extract, types, errors, prompt
"""

from .assembler import (
    NOTE_NAMESPACE,
    assemble,
    assemble_spec_value,
    collect_actions,
    collect_notes,
    replace_actions,
)
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
from .normalize import normalize_lndl_text
from .parser import ParseError, Parser
from .prompt import LNDL_SYSTEM_PROMPT, get_lndl_system_prompt
from .round_outcome import Continue, Exhausted, Failed, Retry, RoundOutcome, Success
from .types import ActionCall, LNDLOutput, LvarMetadata, RLvarMetadata, Scalar

__all__ = (
    "LNDL_SYSTEM_PROMPT",
    "NOTE_NAMESPACE",
    "ActionCall",
    "AmbiguousMatchError",
    "Continue",
    "Exhausted",
    "Failed",
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
    "Retry",
    "RoundOutcome",
    "Scalar",
    "Success",
    "Token",
    "TokenType",
    "assemble",
    "assemble_spec_value",
    "collect_actions",
    "collect_notes",
    "extract_lndl_blocks",
    "get_lndl_system_prompt",
    "normalize_lndl_text",
    "replace_actions",
)
