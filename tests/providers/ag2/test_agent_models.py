# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.providers.ag2.agent.models."""

from __future__ import annotations

import pytest

from lionagi.providers.ag2.agent.models import (
    AG2AgentRequest,
    AgentConfig,
    _content_to_text,
)


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.name == "agent"
        assert cfg.prompt == "You are a helpful assistant."
        assert cfg.tools == []
        assert cfg.enable_subtasks is False
        assert cfg.knowledge is False
        assert cfg.observers == []
        assert cfg.policies == []
        assert cfg.response_schema is None

    def test_custom_values(self):
        cfg = AgentConfig(
            name="MyAgent",
            prompt="You are an expert.",
            tools=["search", "read"],
            enable_subtasks=True,
            knowledge=True,
            observers=["loop_detector"],
            policies=["sliding_window", "token_budget"],
        )
        assert cfg.name == "MyAgent"
        assert cfg.tools == ["search", "read"]
        assert cfg.enable_subtasks is True
        assert cfg.knowledge is True
        assert len(cfg.observers) == 1
        assert len(cfg.policies) == 2

    def test_prompt_as_list(self):
        cfg = AgentConfig(prompt=["System msg 1", "System msg 2"])
        assert isinstance(cfg.prompt, list)
        assert len(cfg.prompt) == 2

    def test_response_schema_accepts_pydantic_model(self):
        from pydantic import BaseModel

        class Answer(BaseModel):
            text: str

        cfg = AgentConfig(response_schema=Answer)
        assert cfg.response_schema is Answer


class TestAG2AgentRequest:
    def test_minimal_with_prompt(self):
        req = AG2AgentRequest(prompt="What is the capital of France?")
        assert req.prompt == "What is the capital of France?"
        assert req.messages == []
        assert req.agent_config is None

    def test_derive_prompt_from_messages_string(self):
        req = AG2AgentRequest(messages=[{"role": "user", "content": "hello world"}])
        assert req.prompt == "hello world"

    def test_derive_prompt_from_messages_list_content(self):
        req = AG2AgentRequest(
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "nested text"}]}
            ]
        )
        assert req.prompt == "nested text"

    def test_derive_prompt_from_messages_dict_content(self):
        req = AG2AgentRequest(
            messages=[{"role": "user", "content": {"text": "dict text"}}]
        )
        assert req.prompt == "dict text"

    def test_explicit_prompt_overrides_messages(self):
        req = AG2AgentRequest(
            prompt="explicit",
            messages=[{"role": "user", "content": "from messages"}],
        )
        assert req.prompt == "explicit"

    def test_empty_messages_and_no_prompt(self):
        req = AG2AgentRequest()
        assert req.prompt == ""

    def test_with_agent_config(self):
        cfg = AgentConfig(name="bot", tools=["search"])
        req = AG2AgentRequest(prompt="search for news", agent_config=cfg)
        assert req.agent_config.name == "bot"
        assert req.agent_config.tools == ["search"]


class TestContentToText:
    def test_string(self):
        assert _content_to_text("hello") == "hello"

    def test_list(self):
        result = _content_to_text(["a", "b"])
        assert result == "a\nb"

    def test_dict_text_key(self):
        assert _content_to_text({"text": "value"}) == "value"

    def test_dict_content_key(self):
        assert _content_to_text({"content": "inner"}) == "inner"

    def test_none(self):
        assert _content_to_text(None) == ""

    def test_number(self):
        assert _content_to_text(42) == "42"

    def test_empty_list_items_filtered(self):
        result = _content_to_text(["a", None, "b"])
        # None becomes "" which is filtered, so result is "a\nb"
        assert "a" in result
        assert "b" in result
