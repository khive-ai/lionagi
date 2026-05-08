# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Function call parser with khive-mcp extensions for unified tool paradigm.

Core parsing with support for:
- Service namespacing: cognition.remember_episodic(...)
- Batch parsing: [call1(...), call2(...)]
- Reserved keyword handling: from= -> from_= (Python keywords as args)
"""

from __future__ import annotations

import ast
import re
from typing import Any

# Python reserved keywords that might be used as field names
# These get mapped to underscore versions for parsing
RESERVED_KEYWORDS = {
    "from",
    "import",
    "class",
    "def",
    "return",
    "yield",
    "async",
    "await",
}

# Regex to match keyword arguments with reserved names
# Matches: from="value" or from='value' at word boundary
_RESERVED_KWARG_PATTERN = re.compile(
    r"\b(" + "|".join(RESERVED_KEYWORDS) + r")\s*=", re.MULTILINE
)

__all__ = (
    "parse_batch_function_calls",
    "parse_function_call",
)


def _escape_reserved_keywords(call_str: str) -> str:
    """Escape Python reserved keywords used as argument names.

    Converts `from=` to `from_=` so ast.parse can handle it.
    The underscore version is what Pydantic expects for aliased fields.

    Args:
        call_str: Function call string that may contain reserved keywords

    Returns:
        String with reserved keywords escaped
    """
    return _RESERVED_KWARG_PATTERN.sub(r"\1_=", call_str)


def _ast_to_value(node: ast.AST) -> Any:
    """Convert AST node to Python value with recursive dict/list processing.

    Handles nested dicts, lists, tuples, and JSON-style literals (true/false/null).
    Normalizes JSON literals: true->True, false->False, null->None.

    Args:
        node: AST node to convert

    Returns:
        Python value

    Raises:
        ValueError: If node cannot be converted to a value
    """
    # Handle JSON-style boolean/null names: true, false, null
    if isinstance(node, ast.Name):
        if node.id in ("true", "false", "null"):
            return {"true": True, "false": False, "null": None}[node.id]
        raise ValueError(f"Name '{node.id}' is not a valid literal")

    # Handle dict nodes: {key1: val1, key2: val2, ...}
    if isinstance(node, ast.Dict):
        return {
            _ast_to_value(k): _ast_to_value(v)
            for k, v in zip(node.keys, node.values, strict=False)
        }

    # Handle list nodes: [elem1, elem2, ...]
    if isinstance(node, ast.List):
        return [_ast_to_value(elem) for elem in node.elts]

    # Handle tuple nodes: (elem1, elem2, ...)
    if isinstance(node, ast.Tuple):
        return tuple(_ast_to_value(elem) for elem in node.elts)

    # Handle simple literals (str, int, float, bool, None) via ast.literal_eval
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert AST node: {type(node).__name__}") from e


def parse_function_call(call_str: str) -> dict[str, Any]:
    """Parse Python function call syntax into unified tool format.

    Supports service namespacing for unified tool paradigm:
        - Simple: search("query") -> {operation: "search", arguments: {...}}
        - Namespaced: cognition.remember("...") -> {service: "cognition", ...}
        - Deep: recall.search("...") -> {service: "recall", operation: "search", ...}

    Examples:
        >>> parse_function_call('search("AI news")')
        {'operation': 'search', 'arguments': {'query': 'AI news'}}

        >>> parse_function_call('cognition.remember_episodic(content="...")')
        {'service': 'cognition', 'operation': 'remember_episodic', 'arguments': {'content': '...'}}

    Args:
        call_str: Python function call as string

    Returns:
        Dict with 'operation', optional 'service', and 'arguments' keys
        Legacy 'tool' key also included for backward compatibility

    Raises:
        ValueError: If the string is not a valid function call
    """
    try:
        # Escape reserved keywords before parsing (e.g., from= -> from_=)
        escaped_str = _escape_reserved_keywords(call_str)

        # Parse the call as a Python expression
        tree = ast.parse(escaped_str, mode="eval")
        call = tree.body

        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # Extract function name and service namespace
        service = None
        operation = None

        if isinstance(call.func, ast.Name):
            # Simple call: search(...)
            operation = call.func.id
        elif isinstance(call.func, ast.Attribute):
            # Namespaced call: cognition.remember(...) or recall.search(...)
            operation = call.func.attr

            # Walk up the attribute chain to get service name
            node = call.func.value
            if isinstance(node, ast.Name):
                service = node.id
            elif isinstance(node, ast.Attribute):
                # Multi-level: could be module.service.operation
                # For now, take the last attribute as service
                service = node.attr
        else:
            raise ValueError(f"Unsupported function type: {type(call.func)}")

        # Extract arguments
        arguments = {}

        # Positional arguments (will be mapped by parameter order in schema)
        for i, arg in enumerate(call.args):
            # For now, use position-based keys; will be mapped to param names later
            arguments[f"_pos_{i}"] = _ast_to_value(arg)

        # Keyword arguments
        for keyword in call.keywords:
            if keyword.arg is None:
                # **kwargs syntax
                raise ValueError("**kwargs not supported")
            arguments[keyword.arg] = _ast_to_value(keyword.value)

        # Build result with new unified format
        result = {
            "operation": operation,
            "arguments": arguments,
            "tool": operation,  # Backward compatibility
        }

        if service:
            result["service"] = service

        return result

    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Invalid function call syntax: {e}") from e


def parse_batch_function_calls(batch_str: str) -> list[dict[str, Any]]:
    """Parse batch function calls (array of function calls).

    Supports:
        - Same service batch: [remember(...), recall(...)]
        - Cross-service batch: [cognition.remember(...), waves.check_in()]

    Examples:
        >>> parse_batch_function_calls('[search("A"), search("B")]')
        [
            {'operation': 'search', 'arguments': {'query': 'A'}},
            {'operation': 'search', 'arguments': {'query': 'B'}}
        ]

        >>> parse_batch_function_calls('[cognition.remember(...), waves.check_in()]')
        [
            {'service': 'cognition', 'operation': 'remember', 'arguments': {...}},
            {'service': 'waves', 'operation': 'check_in', 'arguments': {}}
        ]

    Args:
        batch_str: String containing array of function calls

    Returns:
        List of parsed function call dicts

    Raises:
        ValueError: If the string is not a valid array of function calls
    """
    try:
        # Remove whitespace for easier parsing
        batch_str = batch_str.strip()

        # Must start with [ and end with ]
        if not (batch_str.startswith("[") and batch_str.endswith("]")):
            raise ValueError("Batch call must be enclosed in [ ]")

        # Escape reserved keywords before parsing (e.g., from= -> from_=)
        escaped_str = _escape_reserved_keywords(batch_str)

        # Parse as Python list expression
        tree = ast.parse(escaped_str, mode="eval")
        if not isinstance(tree.body, ast.List):
            raise ValueError("Not a list expression")

        results = []
        for element in tree.body.elts:
            if not isinstance(element, ast.Call):
                raise ValueError(
                    f"List element is not a function call: {ast.dump(element)}"
                )

            # Convert the Call node back to source code and parse it
            call_str = ast.unparse(element)
            parsed = parse_function_call(call_str)
            results.append(parsed)

        return results

    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Invalid batch function call syntax: {e}") from e
