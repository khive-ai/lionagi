# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for beta/operations/specs.py — Action, ActionResult, Instruct, spec factories."""

from __future__ import annotations

import pytest

from lionagi.beta.operations.specs import (
    Action,
    ActionResult,
    Instruct,
    _normalize_action_keys,
    _parse_action_blocks,
    get_action_result_spec,
    get_action_spec,
    get_instruct_spec,
)

# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class TestAction:
    def test_basic_construction(self):
        a = Action(function="my_tool", arguments={"x": 1})
        assert a.function == "my_tool"
        assert a.arguments == {"x": 1}

    def test_default_arguments(self):
        a = Action(function="tool")
        assert a.arguments == {}

    def test_coerce_arguments_from_json_string(self):
        a = Action(function="tool", arguments='{"k": "v"}')
        assert a.arguments.get("k") == "v"

    def test_coerce_arguments_from_dict(self):
        a = Action(function="tool", arguments={"key": "val"})
        assert a.arguments["key"] == "val"

    def test_create_from_dict(self):
        actions = Action.create({"function": "foo", "arguments": {"x": 1}})
        assert len(actions) == 1
        assert actions[0].function == "foo"

    def test_create_from_string_with_json(self):
        text = '{"function": "bar", "arguments": {}}'
        actions = Action.create(text)
        assert len(actions) == 1

    def test_create_returns_empty_on_failure(self):
        actions = Action.create("not valid json at all %%%")
        assert actions == []

    def test_create_from_pydantic_model(self):
        from pydantic import BaseModel

        class MyModel(BaseModel):
            function: str = "baz"
            arguments: dict = {}

        actions = Action.create(MyModel())
        assert len(actions) == 1
        assert actions[0].function == "baz"


# ---------------------------------------------------------------------------
# ActionResult
# ---------------------------------------------------------------------------


class TestActionResult:
    def test_basic(self):
        r = ActionResult(function="foo", result="ok")
        assert r.success is True
        assert r.function == "foo"

    def test_error_makes_success_false(self):
        r = ActionResult(function="foo", error="something went wrong")
        assert r.success is False

    def test_both_none_defaults(self):
        r = ActionResult(function="bar")
        assert r.result is None
        assert r.error is None
        assert r.success is True


# ---------------------------------------------------------------------------
# Instruct
# ---------------------------------------------------------------------------


class TestInstruct:
    def test_defaults(self):
        i = Instruct()
        assert i.instruction is None
        assert i.reason is False
        assert i.actions is False
        assert i.action_strategy == "concurrent"

    def test_with_instruction(self):
        i = Instruct(instruction="Do the thing")
        assert i.instruction == "Do the thing"

    def test_action_strategy_invalid_defaults_to_concurrent(self):
        i = Instruct(action_strategy="invalid_value")
        assert i.action_strategy == "concurrent"

    def test_action_strategy_sequential(self):
        i = Instruct(action_strategy="sequential")
        assert i.action_strategy == "sequential"

    def test_with_context_and_guidance(self):
        i = Instruct(context={"key": "val"}, guidance="be careful")
        assert i.context["key"] == "val"


# ---------------------------------------------------------------------------
# get_action_spec / get_action_result_spec / get_instruct_spec
# ---------------------------------------------------------------------------


class TestSpecFactories:
    def test_get_action_spec_returns_spec(self):
        from lionagi.ln.types import Spec

        spec = get_action_spec()
        assert isinstance(spec, Spec)
        assert spec.name == "action_requests"

    def test_get_action_spec_is_cached(self):
        s1 = get_action_spec()
        s2 = get_action_spec()
        assert s1 is s2

    def test_get_action_result_spec_returns_spec(self):
        from lionagi.ln.types import Spec

        spec = get_action_result_spec()
        assert isinstance(spec, Spec)
        assert spec.name == "action_results"

    def test_get_instruct_spec_returns_spec(self):
        from lionagi.ln.types import Spec

        spec = get_instruct_spec()
        assert isinstance(spec, Spec)
        assert spec.name == "instruct_model"

    def test_get_action_result_spec_cached(self):
        s1 = get_action_result_spec()
        s2 = get_action_result_spec()
        assert s1 is s2

    def test_get_instruct_spec_cached(self):
        s1 = get_instruct_spec()
        s2 = get_instruct_spec()
        assert s1 is s2


# ---------------------------------------------------------------------------
# _normalize_action_keys
# ---------------------------------------------------------------------------


class TestNormalizeActionKeys:
    def test_standard_function_arguments(self):
        d = {"function": "do_thing", "arguments": {"x": 1}}
        result = _normalize_action_keys(d)
        assert result["function"] == "do_thing"
        assert result["arguments"] == {"x": 1}

    def test_nested_function_name(self):
        d = {"function": {"name": "nested_fn"}, "arguments": {}}
        result = _normalize_action_keys(d)
        assert result["function"] == "nested_fn"

    def test_action_name_key(self):
        d = {"action_name": "do_thing", "arguments": {}}
        result = _normalize_action_keys(d)
        assert result["function"] == "do_thing"

    def test_recipient_name_key(self):
        d = {"recipient_name": "do_thing", "arguments": {}}
        result = _normalize_action_keys(d)
        assert result["function"] == "do_thing"

    def test_parameters_key(self):
        d = {"function": "f", "parameters": {"k": "v"}}
        result = _normalize_action_keys(d)
        assert "arguments" in result

    def test_missing_function_returns_none(self):
        d = {"arguments": {"x": 1}}
        result = _normalize_action_keys(d)
        assert result is None

    def test_default_empty_arguments(self):
        d = {"function": "f"}
        result = _normalize_action_keys(d)
        assert result["arguments"] == {}


# ---------------------------------------------------------------------------
# _parse_action_blocks
# ---------------------------------------------------------------------------


class TestParseActionBlocks:
    def test_from_dict(self):
        d = {"function": "do_it", "arguments": {}}
        result = _parse_action_blocks(d)
        assert len(result) == 1

    def test_from_json_string(self):
        text = '{"function": "do_it", "arguments": {}}'
        result = _parse_action_blocks(text)
        assert len(result) == 1

    def test_from_base_model(self):
        from pydantic import BaseModel

        class M(BaseModel):
            function: str = "fn"
            arguments: dict = {}

        result = _parse_action_blocks(M())
        assert len(result) == 1
        assert result[0]["function"] == "fn"

    def test_from_python_code_fence(self):
        text = '```python\n{"function": "fn", "arguments": {}}\n```'
        result = _parse_action_blocks(text)
        # May find it in python block fallback
        assert isinstance(result, list)

    def test_invalid_returns_empty(self):
        result = _parse_action_blocks("no json here %%%")
        assert result == []
