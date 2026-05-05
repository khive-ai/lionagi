# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.libs.parse.function_call — parse_function_call,
parse_batch_function_calls, _ast_to_value, _escape_reserved_keywords."""

import ast

import pytest

from lionagi.libs.parse.function_call import (
    _ast_to_value,
    _escape_reserved_keywords,
    parse_batch_function_calls,
    parse_function_call,
)

# ---------------------------------------------------------------------------
# _escape_reserved_keywords
# ---------------------------------------------------------------------------


class TestEscapeReservedKeywords:
    def test_escapes_from_kwarg(self):
        result = _escape_reserved_keywords("foo(from=1)")
        assert "from_=" in result
        assert "from=" not in result

    def test_escapes_import_kwarg(self):
        result = _escape_reserved_keywords("foo(import=1)")
        assert "import_=" in result

    def test_escapes_class_kwarg(self):
        result = _escape_reserved_keywords("foo(class=1)")
        assert "class_=" in result

    def test_escapes_return_kwarg(self):
        result = _escape_reserved_keywords("foo(return=1)")
        assert "return_=" in result

    def test_non_reserved_not_escaped(self):
        original = "foo(x=1, y=2)"
        result = _escape_reserved_keywords(original)
        assert result == original

    def test_mixed_reserved_and_regular(self):
        result = _escape_reserved_keywords("foo(from=1, x=2)")
        assert "from_=" in result
        assert "x=2" in result


# ---------------------------------------------------------------------------
# _ast_to_value
# ---------------------------------------------------------------------------


class TestAstToValue:
    def test_name_true(self):
        node = ast.parse("true", mode="eval").body
        assert _ast_to_value(node) is True

    def test_name_false(self):
        node = ast.parse("false", mode="eval").body
        assert _ast_to_value(node) is False

    def test_name_null(self):
        node = ast.parse("null", mode="eval").body
        assert _ast_to_value(node) is None

    def test_name_unknown_raises_value_error(self):
        node = ast.parse("SomeUnknown", mode="eval").body
        with pytest.raises(ValueError, match="not a valid literal"):
            _ast_to_value(node)

    def test_dict_node(self):
        node = ast.parse('{"key": "value"}', mode="eval").body
        result = _ast_to_value(node)
        assert result == {"key": "value"}

    def test_dict_with_nested(self):
        node = ast.parse('{"a": 1, "b": 2}', mode="eval").body
        result = _ast_to_value(node)
        assert result == {"a": 1, "b": 2}

    def test_list_node(self):
        node = ast.parse("[1, 2, 3]", mode="eval").body
        result = _ast_to_value(node)
        assert result == [1, 2, 3]

    def test_list_empty(self):
        node = ast.parse("[]", mode="eval").body
        result = _ast_to_value(node)
        assert result == []

    def test_tuple_node(self):
        node = ast.parse("(1, 2, 3)", mode="eval").body
        result = _ast_to_value(node)
        assert result == (1, 2, 3)

    def test_tuple_empty(self):
        node = ast.parse("()", mode="eval").body
        result = _ast_to_value(node)
        assert result == ()

    def test_literal_integer(self):
        node = ast.parse("42", mode="eval").body
        assert _ast_to_value(node) == 42

    def test_literal_string(self):
        node = ast.parse('"hello"', mode="eval").body
        assert _ast_to_value(node) == "hello"

    def test_literal_float(self):
        node = ast.parse("3.14", mode="eval").body
        assert _ast_to_value(node) == pytest.approx(3.14)

    def test_nested_list_in_dict(self):
        node = ast.parse('{"items": [1, 2]}', mode="eval").body
        result = _ast_to_value(node)
        assert result == {"items": [1, 2]}


# ---------------------------------------------------------------------------
# parse_function_call
# ---------------------------------------------------------------------------


class TestParseFunctionCall:
    def test_basic_kwargs(self):
        result = parse_function_call("foo(a=1, b=2)")
        assert result["operation"] == "foo"
        assert result["tool"] == "foo"
        assert result["arguments"] == {"a": 1, "b": 2}

    def test_no_args(self):
        result = parse_function_call("foo()")
        assert result["operation"] == "foo"
        assert result["arguments"] == {}

    def test_service_operation(self):
        result = parse_function_call("service.operation(x=1)")
        assert result["operation"] == "operation"
        assert result["service"] == "service"
        assert result["arguments"] == {"x": 1}

    def test_double_attribute_service(self):
        result = parse_function_call("a.b.c(x=1)")
        assert result["operation"] == "c"
        assert result["service"] == "b"

    def test_reserved_keyword_from(self):
        result = parse_function_call("foo(from=1)")
        assert result["arguments"]["from_"] == 1

    def test_boolean_literals(self):
        result = parse_function_call("foo(a=true, b=false)")
        assert result["arguments"]["a"] is True
        assert result["arguments"]["b"] is False

    def test_null_literal(self):
        result = parse_function_call("foo(c=null)")
        assert result["arguments"]["c"] is None

    def test_string_argument(self):
        result = parse_function_call('foo(name="hello")')
        assert result["arguments"]["name"] == "hello"

    def test_list_argument(self):
        result = parse_function_call("foo(items=[1, 2, 3])")
        assert result["arguments"]["items"] == [1, 2, 3]

    def test_dict_argument(self):
        result = parse_function_call('foo(cfg={"k": "v"})')
        assert result["arguments"]["cfg"] == {"k": "v"}

    def test_positional_args_encoded_with_pos_prefix(self):
        result = parse_function_call("foo(1, 2, x=3)")
        assert result["arguments"]["_pos_0"] == 1
        assert result["arguments"]["_pos_1"] == 2
        assert result["arguments"]["x"] == 3

    def test_not_a_call_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid function call syntax"):
            parse_function_call("not_a_call")

    def test_expression_not_call_raises(self):
        with pytest.raises(ValueError, match="Invalid function call syntax"):
            parse_function_call("1 + 2")

    def test_invalid_syntax_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid function call syntax"):
            parse_function_call("foo(a=@@@)")

    def test_result_has_operation_tool_arguments_keys(self):
        result = parse_function_call("bar(x=99)")
        assert "operation" in result
        assert "tool" in result
        assert "arguments" in result

    def test_service_key_absent_when_no_service(self):
        result = parse_function_call("bar(x=1)")
        assert "service" not in result

    def test_mixed_json_like_literals(self):
        result = parse_function_call("foo(a=true, b=null, c=false)")
        assert result["arguments"] == {"a": True, "b": None, "c": False}

    def test_float_argument(self):
        result = parse_function_call("foo(pi=3.14)")
        assert result["arguments"]["pi"] == pytest.approx(3.14)

    def test_nested_dict_argument(self):
        result = parse_function_call('foo(cfg={"nested": {"x": 1}})')
        assert result["arguments"]["cfg"]["nested"]["x"] == 1

    def test_kwargs_double_star_raises(self):
        with pytest.raises(ValueError):
            parse_function_call("foo(**kwargs)")


# ---------------------------------------------------------------------------
# parse_batch_function_calls
# ---------------------------------------------------------------------------


class TestParseBatchFunctionCalls:
    def test_basic_two_calls(self):
        result = parse_batch_function_calls("[foo(a=1), bar(b=2)]")
        assert len(result) == 2
        assert result[0]["operation"] == "foo"
        assert result[0]["arguments"]["a"] == 1
        assert result[1]["operation"] == "bar"
        assert result[1]["arguments"]["b"] == 2

    def test_empty_list(self):
        result = parse_batch_function_calls("[]")
        assert result == []

    def test_single_call_in_list(self):
        result = parse_batch_function_calls("[foo(x=42)]")
        assert len(result) == 1
        assert result[0]["arguments"]["x"] == 42

    def test_not_enclosed_in_brackets_raises(self):
        with pytest.raises(ValueError, match="Invalid batch function call syntax"):
            parse_batch_function_calls("foo(a=1)\nbar(b=2)")

    def test_bare_function_call_raises(self):
        with pytest.raises(ValueError, match="Invalid batch function call syntax"):
            parse_batch_function_calls("foo(a=1)")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Invalid batch function call syntax"):
            parse_batch_function_calls("")

    def test_list_with_non_call_element_raises(self):
        with pytest.raises(ValueError, match="Invalid batch function call syntax"):
            parse_batch_function_calls("[foo(a=1), 42]")

    def test_reserved_keyword_in_batch(self):
        result = parse_batch_function_calls("[foo(from=1)]")
        assert result[0]["arguments"]["from_"] == 1

    def test_multiple_calls_preserve_order(self):
        result = parse_batch_function_calls("[a(), b(), c()]")
        ops = [r["operation"] for r in result]
        assert ops == ["a", "b", "c"]
