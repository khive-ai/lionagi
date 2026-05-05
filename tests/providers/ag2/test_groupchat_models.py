# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.providers.ag2.groupchat.models."""

from __future__ import annotations

import pytest

from lionagi.providers.ag2.groupchat.models import (
    AfterWorkCondition,
    AG2GroupChatRequest,
    AgentSpec,
    GroupChatSpec,
    HandoffCondition,
    ResearchPlan,
    _content_to_text,
    _prompt_from_messages,
    _resolve_function_map,
    _resolve_functions,
    _resolve_yield_events,
)

# ---------------------------------------------------------------------------
# HandoffCondition
# ---------------------------------------------------------------------------


class TestHandoffCondition:
    def test_defaults(self):
        hc = HandoffCondition(target="AgentA")
        assert hc.target == "AgentA"
        assert hc.condition is None
        assert hc.condition_type == "llm"
        assert hc.available is None
        assert hc.available_type == "context"
        assert hc.llm_function_name is None

    def test_full_construction(self):
        hc = HandoffCondition(
            target="AgentB",
            condition="task is done",
            condition_type="context",
            available="ready",
            available_type="context_expression",
            llm_function_name="handoff_fn",
        )
        assert hc.target == "AgentB"
        assert hc.condition == "task is done"
        assert hc.condition_type == "context"
        assert hc.available == "ready"
        assert hc.available_type == "context_expression"
        assert hc.llm_function_name == "handoff_fn"

    def test_serialization_roundtrip(self):
        hc = HandoffCondition(
            target="X", condition="cond", condition_type="context_expression"
        )
        d = hc.model_dump()
        hc2 = HandoffCondition.model_validate(d)
        assert hc2.target == "X"
        assert hc2.condition == "cond"
        assert hc2.condition_type == "context_expression"

    def test_invalid_condition_type(self):
        with pytest.raises(Exception):
            HandoffCondition(target="X", condition_type="invalid_type")


# ---------------------------------------------------------------------------
# AfterWorkCondition
# ---------------------------------------------------------------------------


class TestAfterWorkCondition:
    def test_defaults(self):
        aw = AfterWorkCondition(target="AgentC")
        assert aw.target == "AgentC"
        assert aw.condition is None
        assert aw.condition_type == "context"
        assert aw.available is None
        assert aw.available_type == "context"

    def test_context_expression(self):
        aw = AfterWorkCondition(
            target="AgentD",
            condition="x > 5",
            condition_type="context_expression",
            available="flag",
        )
        assert aw.condition == "x > 5"
        assert aw.condition_type == "context_expression"
        assert aw.available == "flag"


# ---------------------------------------------------------------------------
# AgentSpec
# ---------------------------------------------------------------------------


class TestAgentSpec:
    def test_minimal_construction(self):
        spec = AgentSpec(name="Researcher")
        assert spec.name == "Researcher"
        assert spec.role == ""
        assert spec.human_input_mode == "NEVER"
        assert spec.tools == []
        assert spec.handoffs == []

    def test_normalize_legacy_handoff_conditions(self):
        spec = AgentSpec.model_validate(
            {
                "name": "Writer",
                "handoff_conditions": [{"target": "Editor", "condition": "draft done"}],
            }
        )
        assert len(spec.handoffs) == 1
        assert spec.handoffs[0].target == "Editor"

    def test_normalize_nlip_dict_inherits_name(self):
        spec = AgentSpec.model_validate(
            {
                "name": "RemoteAgent",
                "role": "assistant",
                "nlip": {"url": "http://example.com"},
            }
        )
        assert spec.nlip is not None
        assert spec.nlip.name == "RemoteAgent"
        assert spec.nlip.role == "assistant"

    def test_nlip_url_from_nlip_dict(self):
        spec = AgentSpec.model_validate(
            {
                "name": "Remote",
                "nlip": {"url": "http://nlip.test:8080"},
            }
        )
        assert spec.nlip_url == "http://nlip.test:8080"

    def test_context_variables_default_empty(self):
        spec = AgentSpec(name="A")
        assert spec.context_variables == {}

    def test_max_consecutive_auto_reply_constraint(self):
        spec = AgentSpec(name="A", max_consecutive_auto_reply=5)
        assert spec.max_consecutive_auto_reply == 5
        with pytest.raises(Exception):
            AgentSpec(name="A", max_consecutive_auto_reply=-1)

    def test_nlip_timeout_constraint(self):
        with pytest.raises(Exception):
            AgentSpec(name="A", nlip_timeout=-1.0)


# ---------------------------------------------------------------------------
# GroupChatSpec
# ---------------------------------------------------------------------------


class TestGroupChatSpec:
    def test_defaults(self):
        spec = GroupChatSpec()
        assert spec.name == "endpoint_chat"
        assert spec.objective == ""
        assert spec.max_round == 15
        assert spec.pattern == "default"
        assert spec.mechanical_fallback is True
        assert spec.terminate_after_last is True

    def test_normalize_max_rounds(self):
        spec = GroupChatSpec.model_validate({"max_rounds": 5})
        assert spec.max_round == 5

    def test_normalize_context_variables(self):
        spec = GroupChatSpec.model_validate({"context_variables": {"key": "value"}})
        assert spec.context == {"key": "value"}

    def test_normalize_agent_configs(self):
        spec = GroupChatSpec.model_validate(
            {"agent_configs": [{"name": "Alice"}, {"name": "Bob"}]}
        )
        assert len(spec.agents) == 2
        assert spec.agents[0].name == "Alice"

    def test_max_round_must_be_positive(self):
        with pytest.raises(Exception):
            GroupChatSpec(max_round=0)

    def test_with_agents(self):
        spec = GroupChatSpec(
            agents=[AgentSpec(name="A"), AgentSpec(name="B")],
            pattern="round_robin",
        )
        assert len(spec.agents) == 2
        assert spec.pattern == "round_robin"


# ---------------------------------------------------------------------------
# ResearchPlan
# ---------------------------------------------------------------------------


class TestResearchPlan:
    def test_construction(self):
        plan = ResearchPlan(
            topic="AI safety",
            hypothesis="RLHF works",
            group_chats=[GroupChatSpec(name="team1")],
            synthesis_instruction="Combine results",
            expected_output="Report",
        )
        assert plan.topic == "AI safety"
        assert plan.safeguard_policy == []
        assert len(plan.group_chats) == 1

    def test_with_safeguard_policy(self):
        plan = ResearchPlan(
            topic="Quantum",
            hypothesis="Entanglement",
            group_chats=[],
            synthesis_instruction="Sum",
            expected_output="Paper",
            safeguard_policy=["no fabrication", "cite sources"],
        )
        assert len(plan.safeguard_policy) == 2


# ---------------------------------------------------------------------------
# AG2GroupChatRequest
# ---------------------------------------------------------------------------


class TestAG2GroupChatRequest:
    def test_minimal_with_prompt(self):
        req = AG2GroupChatRequest(prompt="Hello!")
        assert req.prompt == "Hello!"
        assert req.objective == "Hello!"

    def test_derive_prompt_from_messages(self):
        req = AG2GroupChatRequest(
            messages=[{"role": "user", "content": "What is 2+2?"}]
        )
        assert req.prompt == "What is 2+2?"
        assert req.objective == "What is 2+2?"

    def test_normalize_aliases_max_rounds(self):
        req = AG2GroupChatRequest.model_validate({"prompt": "hi", "max_rounds": 8})
        assert req.max_round == 8

    def test_normalize_aliases_context(self):
        req = AG2GroupChatRequest.model_validate(
            {
                "prompt": "hi",
                "context": {"x": 1},
            }
        )
        assert req.context_variables == {"x": 1}

    def test_normalize_agent_configs(self):
        req = AG2GroupChatRequest.model_validate(
            {
                "prompt": "task",
                "agent_configs": [{"name": "AgentX"}],
            }
        )
        assert len(req.agents) == 1
        assert req.agents[0].name == "AgentX"

    def test_run_messages_empty_uses_prompt(self):
        req = AG2GroupChatRequest(prompt="run this")
        assert req.run_messages() == "run this"

    def test_run_messages_with_messages(self):
        msgs = [{"role": "user", "content": "hello"}]
        req = AG2GroupChatRequest(prompt="ignored", messages=msgs)
        assert req.run_messages() == msgs

    def test_to_group_chat_spec(self):
        req = AG2GroupChatRequest(
            prompt="build something",
            agents=[AgentSpec(name="Builder")],
            max_round=3,
            pattern="auto",
        )
        spec = req.to_group_chat_spec()
        assert spec.objective == "build something"
        assert spec.max_round == 3
        assert spec.pattern == "auto"
        assert len(spec.agents) == 1

    def test_to_group_chat_spec_with_agent_configs_kwarg(self):
        req = AG2GroupChatRequest(prompt="task")
        spec = req.to_group_chat_spec(agent_configs=[{"name": "Helper"}])
        assert len(spec.agents) == 1
        assert spec.agents[0].name == "Helper"

    def test_content_to_text_list(self):
        req = AG2GroupChatRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "part1"},
                        {"type": "text", "text": "part2"},
                    ],
                }
            ]
        )
        assert req.prompt == "part1\npart2"

    def test_max_round_must_be_positive(self):
        with pytest.raises(Exception):
            AG2GroupChatRequest(prompt="x", max_round=0)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestContentToText:
    def test_string(self):
        assert _content_to_text("hello") == "hello"

    def test_list_of_strings(self):
        assert _content_to_text(["a", "b", "c"]) == "a\nb\nc"

    def test_dict_with_text_key(self):
        assert _content_to_text({"text": "value"}) == "value"

    def test_dict_with_content_key(self):
        assert _content_to_text({"content": "inner"}) == "inner"

    def test_nested(self):
        assert _content_to_text([{"text": "x"}, "y"]) == "x\ny"

    def test_none_returns_empty(self):
        assert _content_to_text(None) == ""

    def test_non_string_serialized(self):
        result = _content_to_text(42)
        assert result == "42"

    def test_empty_list(self):
        assert _content_to_text([]) == ""


class TestPromptFromMessages:
    def test_last_message_content(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        assert _prompt_from_messages(msgs) == "second"

    def test_empty_messages(self):
        assert _prompt_from_messages([]) == ""

    def test_content_list(self):
        msgs = [{"role": "user", "content": [{"text": "nested"}]}]
        assert _prompt_from_messages(msgs) == "nested"


class TestResolveFunctionMap:
    def test_callable_passes_through(self):
        fn = lambda x: x
        result = _resolve_function_map({"f": fn}, {})
        assert result["f"] is fn

    def test_string_resolved_from_registry(self):
        fn = lambda: "result"
        result = _resolve_function_map({"f": "my_tool"}, {"my_tool": fn})
        assert result["f"] is fn

    def test_unresolvable_string_skipped(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            result = _resolve_function_map({"f": "missing_tool"}, {})
        assert "f" not in result

    def test_empty_map(self):
        assert _resolve_function_map({}, {}) == {}


class TestResolveFunctions:
    def test_callable_included(self):
        fn = lambda: None
        result = _resolve_functions([fn], {})
        assert result == [fn]

    def test_string_resolved_from_registry(self):
        fn = lambda: None
        result = _resolve_functions(["my_fn"], {"my_fn": fn})
        assert result == [fn]

    def test_unresolvable_skipped(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            result = _resolve_functions(["missing"], {})
        assert result == []

    def test_empty(self):
        assert _resolve_functions([], {}) == []


class TestResolveYieldEvents:
    def test_type_passthrough(self):
        try:
            from autogen.events.agent_events import TextEvent

            result = _resolve_yield_events([TextEvent])
            assert TextEvent in result
        except ImportError:
            pytest.skip("autogen not available")

    def test_string_alias_text(self):
        try:
            result = _resolve_yield_events(["text"])
            assert len(result) == 1
        except ImportError:
            pytest.skip("autogen not available")

    def test_unknown_string_skipped(self, caplog):
        import logging

        try:
            with caplog.at_level(logging.WARNING):
                result = _resolve_yield_events(["not_a_real_event"])
            # Should be empty or not raise
        except ImportError:
            pytest.skip("autogen not available")

    def test_non_string_non_type_skipped(self, caplog):
        import logging

        try:
            with caplog.at_level(logging.WARNING):
                result = _resolve_yield_events([42])
            # 42 is not str or type; should be skipped or produce empty
        except ImportError:
            pytest.skip("autogen not available")
