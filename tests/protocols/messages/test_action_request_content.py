"""Surgical gap-fill tests for ActionRequestContent missing branches.

Targets ~14 missing statements in lionagi/protocols/messages/action_request.py:
Lines: 43-48, 53, 63-65, 69, 87, 96-97, 126, 131, 136
"""

import pytest

from lionagi.ln.types._sentinel import Undefined
from lionagi.protocols.messages.action_request import (
    ActionRequest,
    ActionRequestContent,
)
from lionagi.protocols.messages.message import MessageRole

# ---------------------------------------------------------------------------
# render_compact (lines 43-48)
# ---------------------------------------------------------------------------


def test_render_compact_basic():
    """Lines 43-48: render_compact formats function call."""
    c = ActionRequestContent(function="my_func", arguments={"key": "val", "num": 42})
    result = c.render_compact()
    assert "my_func" in result
    assert "key='val'" in result
    assert "num=42" in result


def test_render_compact_empty_args():
    """Lines 43-48: render_compact with no arguments."""
    c = ActionRequestContent(function="no_args")
    result = c.render_compact()
    assert result == "no_args()"


def test_render_compact_string_arg_uses_repr():
    """Line 46: string values use !r format."""
    c = ActionRequestContent(function="f", arguments={"x": "hello"})
    result = c.render_compact()
    assert "x='hello'" in result


def test_render_compact_non_string_no_repr():
    """Line 47: non-string values use f format (no repr)."""
    c = ActionRequestContent(function="f", arguments={"n": 123})
    result = c.render_compact()
    assert "n=123" in result


# ---------------------------------------------------------------------------
# role property (line 53)
# ---------------------------------------------------------------------------


def test_role_property():
    """Line 53: role returns MessageRole.ACTION."""
    c = ActionRequestContent(function="test")
    assert c.role == MessageRole.ACTION


# ---------------------------------------------------------------------------
# create classmethod — sentinel handling (lines 63-65)
# ---------------------------------------------------------------------------


def test_create_with_none_arguments():
    """Line 63-65: None arguments → empty dict."""
    c = ActionRequestContent.create(function="f", arguments=None)
    assert c.arguments == {}


def test_create_with_sentinel_arguments():
    """Lines 63-65: sentinel arguments → empty dict."""
    c = ActionRequestContent.create(function="f", arguments=Undefined)
    assert c.arguments == {}


# ---------------------------------------------------------------------------
# render method (line 69)
# ---------------------------------------------------------------------------


def test_render_delegates_to_rendered():
    """Line 69: render() delegates to rendered."""
    c = ActionRequestContent(function="foo", arguments={"a": 1})
    assert c.render() == c.rendered


# ---------------------------------------------------------------------------
# from_dict — callable function (line 87)
# ---------------------------------------------------------------------------


def test_from_dict_with_callable_function():
    """Line 87: callable function extracts __name__."""

    def my_function():
        pass

    data = {"function": my_function, "arguments": {"x": 1}}
    c = ActionRequestContent.from_dict(data)
    assert c.function == "my_function"


# ---------------------------------------------------------------------------
# from_dict — arguments normalization (lines 93-99)
# ---------------------------------------------------------------------------


def test_from_dict_arguments_as_json_string():
    """Lines 94-97: arguments as JSON string → parsed to dict."""
    data = {"function": "f", "arguments": '{"x": 1, "y": 2}'}
    c = ActionRequestContent.from_dict(data)
    assert c.arguments == {"x": 1, "y": 2}


# ---------------------------------------------------------------------------
# ActionRequest validator (lines 119-126)
# ---------------------------------------------------------------------------


def test_action_request_validate_content_none():
    """Line 121: None content → default ActionRequestContent."""
    ar = ActionRequest(content=None)
    assert ar.content.function == ""
    assert ar.content.arguments == {}


def test_action_request_validate_content_from_dict():
    """Line 122-123: dict content → from_dict."""
    ar = ActionRequest(content={"function": "my_func", "arguments": {"a": 1}})
    assert ar.content.function == "my_func"
    assert ar.content.arguments == {"a": 1}


def test_action_request_validate_content_raises_on_bad_type():
    """Line 126: TypeError for unsupported content type."""
    with pytest.raises(TypeError, match="content must be dict or ActionRequestContent"):
        ActionRequest(content="bad string")


# ---------------------------------------------------------------------------
# ActionRequest properties (lines 131, 136)
# ---------------------------------------------------------------------------


def test_action_request_function_property():
    """Line 131: function property delegates to content.function."""
    ar = ActionRequest(content=ActionRequestContent(function="do_thing", arguments={}))
    assert ar.function == "do_thing"


def test_action_request_arguments_property():
    """Line 136: arguments property delegates to content.arguments."""
    ar = ActionRequest(content=ActionRequestContent(function="f", arguments={"k": "v"}))
    assert ar.arguments == {"k": "v"}


# ---------------------------------------------------------------------------
# Additional gap fill: lines 87 and 97
# ---------------------------------------------------------------------------


def test_from_dict_function_object_with_function_attr():
    """Line 87: object with .function attribute gets extracted."""

    class FunctionLike:
        function = "extracted_name"

    data = {"function": FunctionLike(), "arguments": {}}
    c = ActionRequestContent.from_dict(data)
    assert c.function == "extracted_name"


def test_from_dict_arguments_as_list_of_dicts():
    """Line 97: to_dict returns list → first element is used."""
    # Pass a JSON string that, when parsed, is a list with a dict as first element
    # to_dict with fuzzy_parse on a string like '[{"x": 1}]' → list → first element
    data = {"function": "f", "arguments": '[{"x": 1}]'}
    c = ActionRequestContent.from_dict(data)
    # to_dict parses the JSON list, then takes first element
    assert isinstance(c.arguments, dict)
    assert c.arguments.get("x") == 1
