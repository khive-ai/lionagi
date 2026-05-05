# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.providers.ag2.nlip.models."""

from __future__ import annotations

import pytest

from lionagi.providers.ag2.nlip.models import (
    AG2NlipRequest,
    NlipRemoteAgentSpec,
    _content_to_text,
    _prompt_from_messages,
    nlip_agent_config,
)


class TestNlipRemoteAgentSpec:
    def test_minimal(self):
        spec = NlipRemoteAgentSpec(url="http://agent.test/")
        assert spec.url == "http://agent.test/"
        assert spec.name == "remote"
        assert spec.role == "remote agent"
        assert spec.timeout == 60.0
        assert spec.max_retries == 3
        assert spec.context_variables == {}
        assert spec.client_tools == []

    def test_full_construction(self):
        spec = NlipRemoteAgentSpec(
            url="http://nlip.test:8000",
            name="my-agent",
            role="researcher",
            description="Does research",
            silent=True,
            timeout=30.0,
            max_retries=5,
            context_variables={"project": "lion"},
            client_tools=[{"name": "search"}],
        )
        assert spec.name == "my-agent"
        assert spec.role == "researcher"
        assert spec.description == "Does research"
        assert spec.silent is True
        assert spec.timeout == 30.0
        assert spec.max_retries == 5
        assert spec.context_variables == {"project": "lion"}

    def test_normalize_agent_name_alias(self):
        spec = NlipRemoteAgentSpec.model_validate(
            {
                "url": "http://x.com",
                "agent_name": "my-name",
            }
        )
        assert spec.name == "my-name"

    def test_normalize_context_alias(self):
        spec = NlipRemoteAgentSpec.model_validate(
            {
                "url": "http://x.com",
                "context": {"key": "val"},
            }
        )
        assert spec.context_variables == {"key": "val"}

    def test_timeout_must_be_positive(self):
        with pytest.raises(Exception):
            NlipRemoteAgentSpec(url="http://x.com", timeout=-1.0)

    def test_max_retries_constraint(self):
        with pytest.raises(Exception):
            NlipRemoteAgentSpec(url="http://x.com", max_retries=0)

    def test_to_agent_config_minimal(self):
        spec = NlipRemoteAgentSpec(url="http://agent.test/")
        cfg = spec.to_agent_config()
        assert cfg["nlip_url"] == "http://agent.test/"
        assert cfg["name"] == "remote"
        assert cfg["role"] == "remote agent"
        assert cfg["nlip_timeout"] == 60.0
        assert cfg["nlip_max_retries"] == 3

    def test_to_agent_config_with_optional_fields(self):
        spec = NlipRemoteAgentSpec(
            url="http://x.com",
            description="desc",
            silent=False,
            context_variables={"x": 1},
            client_tools=[{"name": "t"}],
        )
        cfg = spec.to_agent_config()
        assert cfg["description"] == "desc"
        assert "nlip_silent" in cfg
        assert cfg["context_variables"] == {"x": 1}
        assert len(cfg["nlip_client_tools"]) == 1

    def test_to_agent_config_with_overrides(self):
        spec = NlipRemoteAgentSpec(url="http://x.com", name="old")
        cfg = spec.to_agent_config(name="new-name")
        assert cfg["name"] == "new-name"

    def test_description_not_included_when_none(self):
        spec = NlipRemoteAgentSpec(url="http://x.com")
        cfg = spec.to_agent_config()
        assert "description" not in cfg

    def test_silent_not_included_when_none(self):
        spec = NlipRemoteAgentSpec(url="http://x.com")
        cfg = spec.to_agent_config()
        assert "nlip_silent" not in cfg


class TestAG2NlipRequest:
    def test_minimal_with_prompt(self):
        req = AG2NlipRequest(prompt="hello")
        assert req.prompt == "hello"
        assert req.url is None
        assert req.agent_name is None

    def test_normalize_name_alias(self):
        req = AG2NlipRequest.model_validate(
            {
                "prompt": "hi",
                "name": "my-agent",
            }
        )
        assert req.agent_name == "my-agent"

    def test_normalize_context_alias(self):
        req = AG2NlipRequest.model_validate(
            {
                "prompt": "hi",
                "context": {"key": "val"},
            }
        )
        assert req.context_variables == {"key": "val"}

    def test_derive_prompt_from_messages(self):
        req = AG2NlipRequest(
            messages=[
                {"role": "user", "content": "first msg"},
                {"role": "assistant", "content": "response"},
                {"role": "user", "content": "second msg"},
            ]
        )
        assert req.prompt == "second msg"

    def test_derive_prompt_skips_tool_messages(self):
        req = AG2NlipRequest(
            messages=[
                {"role": "user", "content": "real prompt"},
                {"role": "tool", "content": "tool result"},
            ]
        )
        assert req.prompt == "real prompt"

    def test_messages_for_call_with_messages(self):
        msgs = [{"role": "user", "content": "x"}]
        req = AG2NlipRequest(messages=msgs)
        assert req.messages_for_call() == msgs

    def test_messages_for_call_without_messages(self):
        req = AG2NlipRequest(prompt="ask me something")
        result = req.messages_for_call()
        assert result == [{"role": "user", "content": "ask me something"}]

    def test_to_agent_spec_requires_url(self):
        req = AG2NlipRequest(prompt="x")
        with pytest.raises(ValueError, match="url"):
            req.to_agent_spec()

    def test_to_agent_spec_with_url(self):
        req = AG2NlipRequest(prompt="x", url="http://agent.test/")
        spec = req.to_agent_spec()
        assert spec.url == "http://agent.test/"
        assert spec.timeout == 60.0

    def test_to_agent_spec_with_overrides(self):
        req = AG2NlipRequest(prompt="x", url="http://x.com", timeout=30.0)
        spec = req.to_agent_spec(agent_name="override_name", timeout=10.0)
        assert spec.name == "override_name"
        assert spec.timeout == 10.0

    def test_timeout_must_be_positive(self):
        with pytest.raises(Exception):
            AG2NlipRequest(prompt="x", timeout=0.0)

    def test_max_retries_constraint(self):
        with pytest.raises(Exception):
            AG2NlipRequest(prompt="x", max_retries=0)


class TestNlipAgentConfig:
    def test_basic(self):
        cfg = nlip_agent_config("http://x.com/")
        assert cfg["nlip_url"] == "http://x.com/"
        assert cfg["name"] == "remote"
        assert cfg["nlip_timeout"] == 60.0

    def test_with_kwargs(self):
        cfg = nlip_agent_config(
            "http://x.com/",
            name="bot",
            role="helper",
            timeout=30.0,
            max_retries=5,
        )
        assert cfg["name"] == "bot"
        assert cfg["role"] == "helper"
        assert cfg["nlip_timeout"] == 30.0
        assert cfg["nlip_max_retries"] == 5


class TestHelpers:
    def test_content_to_text_string(self):
        assert _content_to_text("hello") == "hello"

    def test_content_to_text_list(self):
        assert _content_to_text(["a", "b"]) == "a\nb"

    def test_content_to_text_dict_text(self):
        assert _content_to_text({"text": "val"}) == "val"

    def test_content_to_text_dict_content(self):
        assert _content_to_text({"content": "inner"}) == "inner"

    def test_content_to_text_none(self):
        assert _content_to_text(None) == ""

    def test_prompt_from_messages_last_user(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "follow up"},
        ]
        assert _prompt_from_messages(msgs) == "follow up"

    def test_prompt_from_messages_skips_tool_role(self):
        msgs = [
            {"role": "user", "content": "question"},
            {"role": "tool", "content": "tool data"},
        ]
        assert _prompt_from_messages(msgs) == "question"

    def test_prompt_from_messages_skips_none_content(self):
        msgs = [
            {"role": "user", "content": "real"},
            {"role": "user", "content": None},
        ]
        result = _prompt_from_messages(msgs)
        # None converts to "None" string, should be skipped due to != "None" check
        assert result == "real"

    def test_prompt_from_messages_empty(self):
        assert _prompt_from_messages([]) == ""
