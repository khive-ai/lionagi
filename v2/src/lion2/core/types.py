# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections.abc import Callable
from enum import Enum
from typing import Any, Dict, List, Set, Union
from uuid import UUID

import lionfuncs as ln
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from .utils import copy

__all__ = (
    "MessageContent",
    "MessageRole",
    "ToolRequest",
    "ToolResponse",
)

SenderRecipient = UUID | str


class MessageRole(str, Enum):
    """Enum for message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    ACTION = "action"


class GenericParams(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )


class MessageContent(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )

    @property
    def rendered(self) -> str:
        return NotImplemented

    def update(self, **kwargs) -> None:
        return NotImplemented


class ToolRequest(BaseModel):
    function: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("arguments", mode="before")
    def _validate_arguments(cls, v: dict | str | BaseModel) -> dict:
        if isinstance(v, dict):
            return copy(v)
        if isinstance(v, BaseModel):
            return v.model_dump()
        if isinstance(v, str):
            try:
                return ln.fuzzy_parse_json(v.strip(), strict=True)
            except Exception as e:
                raise ValueError("Arguments must be a dictionary.") from e

    @field_validator("function", mode="before")
    def _validate_function(cls, v: Any) -> str:
        if isinstance(v, Callable):
            v = v.__name__
        if hasattr(v, "function"):
            v = v.function
        if not isinstance(v, str):
            raise ValueError("Function must be a string or callable.")
        return v

    @classmethod
    def create(cls, data: Any) -> list[ToolRequest]:
        data = parse_tool_requests(data)
        if not data:
            return []
        return [cls(**item) for item in data]


class ToolResponse(BaseModel):
    function: str
    arguments: dict[str, Any]
    output: BaseModel | JsonValue | None


# Pre-compile regex for Python code blocks if not already done globally
_PYTHON_CODE_BLOCK_PATTERN = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)

# Define sets for faster key lookups and clearer normalized targets
_FUNC_TARGET_KEYS: Set[str] = {"name", "function", "recipient"}
_ARGS_TARGET_KEYS: Set[str] = {"parameter", "argument", "arg", "param"}

CANONICAL_ACTION_FUNCTION_KEY = "function"
CANONICAL_ACTION_ARGUMENTS_KEY = "arguments"
REFERENCE_ACTION_KEYS = [
    CANONICAL_ACTION_FUNCTION_KEY,
    CANONICAL_ACTION_ARGUMENTS_KEY,
]


def _normalize_action_key(key: str) -> str:
    """
    Applies the specific key normalizations from the original function.
    More robust normalization would be complex, this preserves original behavior.
    """
    k = key.replace("action_", "").replace("recipient_", "")
    if k.endswith("s") and k[:-1] in (
        "function",
        "recipient",
        "parameter",
        "argument",
        "arg",
        "param",
    ):
        k = k[:-1]
    return k


def parse_tool_requests(
    content: Union[str, Dict[str, Any], BaseModel],
) -> List[Dict[str, Any]]:
    """
    Extracts structured action requests from various input formats, optimized
    while preserving the core logic of the original function.

    Args:
        content: Input data (Pydantic model, dict, or string with JSON/code blocks).

    Returns:
        A list of valid action dictionaries, each with "function" and "arguments".
    """
    intermediate_json_blocks: List[Any] = []

    if isinstance(content, BaseModel):
        intermediate_json_blocks = [content.model_dump()]
    elif isinstance(content, str):
        # Attempt to parse the whole string using your to_json utility
        parsed_content = ln.to_json(content, fuzzy_parse=True)

        # `to_json` might return a single dict, a list, or None/empty list
        if isinstance(parsed_content, dict):
            intermediate_json_blocks.append(parsed_content)
        elif isinstance(parsed_content, list):
            intermediate_json_blocks.extend(parsed_content)
        # If still no blocks, try Python code block extraction
        if not intermediate_json_blocks or all(
            item is None for item in intermediate_json_blocks
        ):  # check if it only contains Nones
            python_block_matches = _PYTHON_CODE_BLOCK_PATTERN.findall(content)

            block_parse_results = []
            for match_str in python_block_matches:
                # `to_json` here will process the content of the python block
                parsed_py_block = ln.to_json(match_str, fuzzy_parse=True)
                if (
                    parsed_py_block is not None
                ):  # Add if to_json returned something
                    block_parse_results.append(parsed_py_block)

            # `block_parse_results` could be like `[dict, [dict, dict], None]`.
            # We need to flatten it and ensure only dicts proceed.
            # Your `to_list` function might be `to_list(block_parse_results, flatten=True, dropna=True)`
            # For this refinement, let's do a direct flatten and filter:
            flat_py_blocks: List[Dict[str, Any]] = []
            for item in block_parse_results:
                if isinstance(item, dict):
                    flat_py_blocks.append(item)
                elif isinstance(item, list):
                    for sub_item in item:
                        if isinstance(sub_item, dict):
                            flat_py_blocks.append(sub_item)
            intermediate_json_blocks = (
                flat_py_blocks  # Overwrite with Python block results
            )

    elif isinstance(
        content, dict
    ):  # content could be {} which is falsey but valid
        intermediate_json_blocks = [content]

    # Filter out Nones and ensure all items are dictionaries before processing
    # This replaces the original `to_list(json_blocks, dropna=True)` and subsequent safeguard
    json_dicts_to_process: List[Dict[str, Any]] = []
    if intermediate_json_blocks:  # Check if it's not empty
        for block in intermediate_json_blocks:
            if isinstance(block, dict):
                json_dicts_to_process.append(block)
            # If `to_json` can return lists of dicts directly at this level,
            # and they are part of intermediate_json_blocks, they need to be added individually.
            # The current logic after python block extraction creates a flat list of dicts.

    # Final actions list
    parsed_actions: List[Dict[str, Any]] = []

    for item_dict in json_dicts_to_process:
        current_action: Dict[str, Any] = {}

        # Handle potentially nested function name first (original logic)
        # Create a temporary dict to iterate to avoid issues if `item_dict["function"]` is modified
        # or iterate over items() which provides a snapshot.
        temp_item_dict = (
            item_dict.copy()
        )  # Work on a copy for this specific modification step
        if "function" in temp_item_dict and isinstance(
            temp_item_dict["function"], dict
        ):
            nested_func_dict = temp_item_dict["function"]
            if "name" in nested_func_dict and isinstance(
                nested_func_dict["name"], str
            ):
                # If successfully flattened, place it directly into current_action
                # And remove/overwrite "function" in temp_item_dict to avoid re-processing by alias
                current_action[CANONICAL_ACTION_FUNCTION_KEY] = (
                    nested_func_dict["name"]
                )
                temp_item_dict[CANONICAL_ACTION_FUNCTION_KEY] = (
                    nested_func_dict["name"]
                )  # ensure loops see the string

        for key, value in temp_item_dict.items():
            normalized_key = _normalize_action_key(key)

            # Check if this key (after normalization) maps to our canonical "function" key
            # And ensure we haven't already set the function from the nested structure logic
            if (
                normalized_key in _FUNC_TARGET_KEYS
                and CANONICAL_ACTION_FUNCTION_KEY not in current_action
            ):
                if isinstance(value, str):  # Function name should be a string
                    current_action[CANONICAL_ACTION_FUNCTION_KEY] = value

            # Check if this key maps to our canonical "arguments" key And ensure we haven't already set arguments
            elif (
                normalized_key in _ARGS_TARGET_KEYS
                and CANONICAL_ACTION_ARGUMENTS_KEY not in current_action
            ):

                args_dict = ln.to_dict(
                    value,
                    parse_strings=True,  # LLMs might provide args as JSON strings
                    str_type_for_parsing="json",
                    fuzzy_parse_strings=True,  # Be lenient with arg strings
                    suppress_errors=True,  # If args are malformed, default to {}
                    default_on_error={},
                    recursive=True,  # Important for nested args
                    use_enum_values=True,  # Sensible default for args
                )
                current_action[CANONICAL_ACTION_ARGUMENTS_KEY] = (
                    args_dict if isinstance(args_dict, dict) else {}
                )

        # Validate the constructed action: must have a function name (string)
        # and arguments must be a dictionary (can be empty, if original `and j["arguments"]` is relaxed).
        if CANONICAL_ACTION_FUNCTION_KEY in current_action and isinstance(
            current_action[CANONICAL_ACTION_FUNCTION_KEY], str
        ):

            # Ensure arguments key exists, defaulting to empty dict if not found by aliases
            if CANONICAL_ACTION_ARGUMENTS_KEY not in current_action:
                current_action[CANONICAL_ACTION_ARGUMENTS_KEY] = {}

            parsed_actions.append(current_action)

    return parsed_actions
