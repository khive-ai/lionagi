from __future__ import annotations

import ast
import re
from typing import Any

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

_RESERVED_KWARG_PATTERN = re.compile(
    r"\b(" + "|".join(RESERVED_KEYWORDS) + r")\s*=", re.MULTILINE
)

__all__ = (
    "parse_batch_function_calls",
    "parse_function_call",
)


def _escape_reserved_keywords(call_str: str) -> str:
    return _RESERVED_KWARG_PATTERN.sub(r"\1_=", call_str)


def _ast_to_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Name):
        if node.id in ("true", "false", "null"):
            return {"true": True, "false": False, "null": None}[node.id]
        raise ValueError(f"Name '{node.id}' is not a valid literal")

    if isinstance(node, ast.Dict):
        return {
            _ast_to_value(k): _ast_to_value(v)
            for k, v in zip(node.keys, node.values, strict=False)
        }

    if isinstance(node, ast.List):
        return [_ast_to_value(elem) for elem in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_ast_to_value(elem) for elem in node.elts)

    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert AST node: {type(node).__name__}") from e


def parse_function_call(call_str: str) -> dict[str, Any]:
    try:
        escaped_str = _escape_reserved_keywords(call_str)
        tree = ast.parse(escaped_str, mode="eval")
        call = tree.body

        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        service = None
        operation = None

        if isinstance(call.func, ast.Name):
            operation = call.func.id
        elif isinstance(call.func, ast.Attribute):
            operation = call.func.attr
            node = call.func.value
            if isinstance(node, ast.Name):
                service = node.id
            elif isinstance(node, ast.Attribute):
                service = node.attr
        else:
            raise ValueError(f"Unsupported function type: {type(call.func)}")

        arguments = {}

        for i, arg in enumerate(call.args):
            arguments[f"_pos_{i}"] = _ast_to_value(arg)

        for keyword in call.keywords:
            if keyword.arg is None:
                raise ValueError("**kwargs not supported")
            arguments[keyword.arg] = _ast_to_value(keyword.value)

        result = {
            "operation": operation,
            "arguments": arguments,
            "tool": operation,
        }

        if service:
            result["service"] = service

        return result

    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Invalid function call syntax: {e}") from e


def parse_batch_function_calls(batch_str: str) -> list[dict[str, Any]]:
    try:
        batch_str = batch_str.strip()

        if not (batch_str.startswith("[") and batch_str.endswith("]")):
            raise ValueError("Batch call must be enclosed in [ ]")

        escaped_str = _escape_reserved_keywords(batch_str)
        tree = ast.parse(escaped_str, mode="eval")
        if not isinstance(tree.body, ast.List):
            raise ValueError("Not a list expression")

        results = []
        for element in tree.body.elts:
            if not isinstance(element, ast.Call):
                raise ValueError(
                    f"List element is not a function call: {ast.dump(element)}"
                )

            call_str = ast.unparse(element)
            parsed = parse_function_call(call_str)
            results.append(parsed)

        return results

    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Invalid batch function call syntax: {e}") from e
