# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Gap-fill tests for ActionResponseContent missing branches."""

from __future__ import annotations

import pytest

from lionagi.protocols.messages.action_response import (
    ActionResponse,
    ActionResponseContent,
)
from lionagi.protocols.messages.message import MessageRole


class TestActionResponseContentRenderSummary:
    def test_render_summary_error_returns_error_string(self):
        content = ActionResponseContent(function="f", arguments={}, error="oops")
        result = content.render_summary()
        assert "error" in result
        assert "oops" in result

    def test_render_summary_error_none_message(self):
        content = ActionResponseContent(function="f", arguments={}, error="")
        result = content.render_summary()
        assert "unknown" in result

    def test_render_summary_output_none_returns_ok(self):
        content = ActionResponseContent(function="f", arguments={}, output=None)
        result = content.render_summary()
        assert result == "ok"

    def test_render_summary_output_is_string(self):
        content = ActionResponseContent(
            function="f", arguments={}, output="result text"
        )
        result = content.render_summary()
        assert result == "result text"

    def test_render_summary_output_is_dict(self):
        content = ActionResponseContent(
            function="f", arguments={}, output={"key": "val"}
        )
        result = content.render_summary()
        assert "key" in result
        assert "val" in result

    def test_render_summary_output_is_list(self):
        content = ActionResponseContent(function="f", arguments={}, output=["a", "b"])
        result = content.render_summary()
        assert "a" in result

    def test_render_summary_output_non_str_dict_list(self):
        content = ActionResponseContent(function="f", arguments={}, output=42)
        result = content.render_summary()
        assert result == "42"

    def test_success_property_true(self):
        content = ActionResponseContent(function="f", arguments={}, output="done")
        assert content.success is True

    def test_success_property_false_when_error(self):
        content = ActionResponseContent(function="f", arguments={}, error="failed")
        assert content.success is False


class TestActionResponseContentCreate:
    def test_create_with_result(self):
        content = ActionResponseContent.create(
            function="search", arguments={"q": "test"}, result={"data": []}
        )
        assert content.function == "search"
        assert content.output == {"data": []}
        assert content.error is None

    def test_create_with_error(self):
        from lionagi.ln.types._sentinel import Undefined

        content = ActionResponseContent.create(
            function="search", arguments={}, result=Undefined, error="not found"
        )
        assert content.error == "not found"
        assert content.output is None

    def test_create_both_sentinel(self):
        from lionagi.ln.types._sentinel import Undefined

        content = ActionResponseContent.create(
            function="f", arguments={}, result=Undefined, error=Undefined
        )
        assert content.output is None
        assert content.error is None


class TestActionResponseContentFromDict:
    def test_from_dict_nested_action_response_format(self):
        data = {
            "action_response": {
                "function": "lookup",
                "arguments": {"key": "x"},
                "output": 42,
            }
        }
        content = ActionResponseContent.from_dict(data)
        assert content.function == "lookup"
        assert content.output == 42

    def test_from_dict_flat_format(self):
        data = {"function": "compute", "arguments": {}, "output": "result"}
        content = ActionResponseContent.from_dict(data)
        assert content.function == "compute"
        assert content.output == "result"

    def test_from_dict_with_action_request_id(self):
        import uuid

        req_id = str(uuid.uuid4())
        data = {
            "function": "f",
            "arguments": {},
            "output": None,
            "action_request_id": req_id,
        }
        content = ActionResponseContent.from_dict(data)
        assert content.action_request_id == req_id

    def test_from_dict_with_error(self):
        data = {"function": "f", "arguments": {}, "output": None, "error": "timeout"}
        content = ActionResponseContent.from_dict(data)
        assert content.error == "timeout"


class TestActionResponseContent:
    def test_role_property(self):
        content = ActionResponseContent(function="f", arguments={})
        assert content.role == MessageRole.ACTION

    def test_result_alias(self):
        content = ActionResponseContent(function="f", arguments={}, output="done")
        assert content.result == "done"

    def test_request_id_alias(self):
        content = ActionResponseContent(
            function="f", arguments={}, action_request_id="abc123"
        )
        assert content.request_id == "abc123"


class TestActionResponse:
    def test_validate_content_none_creates_empty(self):
        msg = ActionResponse(content=None)
        assert isinstance(msg.content, ActionResponseContent)

    def test_validate_content_from_dict(self):
        msg = ActionResponse(content={"function": "f", "arguments": {}, "output": 1})
        assert msg.content.function == "f"

    def test_validate_content_passthrough(self):
        c = ActionResponseContent(function="g", arguments={})
        msg = ActionResponse(content=c)
        assert msg.content is c

    def test_validate_content_raises_on_bad_type(self):
        with pytest.raises(TypeError):
            ActionResponse(content="bad")

    def test_function_property(self):
        msg = ActionResponse(content=ActionResponseContent(function="go", arguments={}))
        assert msg.function == "go"

    def test_arguments_property(self):
        msg = ActionResponse(
            content=ActionResponseContent(function="f", arguments={"x": 1})
        )
        assert msg.arguments == {"x": 1}

    def test_output_property(self):
        msg = ActionResponse(
            content=ActionResponseContent(function="f", arguments={}, output=99)
        )
        assert msg.output == 99

    def test_error_property(self):
        msg = ActionResponse(
            content=ActionResponseContent(function="f", arguments={}, error="err")
        )
        assert msg.error == "err"
