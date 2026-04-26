# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.service.token_budget — TokenBudget dataclass and helpers."""

import types

import pytest

from lionagi.service.token_budget import TokenBudget, get_context_window


class TestTokenBudget:
    def test_remaining_clamps_at_zero_when_over_limit(self):
        budget = TokenBudget(used=150, limit=100, model="m")
        assert budget.remaining == 0

    def test_remaining_positive_when_under_limit(self):
        budget = TokenBudget(used=40, limit=100)
        assert budget.remaining == 60

    def test_usage_pct_over_limit(self):
        budget = TokenBudget(used=150, limit=100, model="m")
        assert budget.usage_pct == 1.5

    def test_is_critical_when_over_limit(self):
        budget = TokenBudget(used=150, limit=100, model="m")
        assert budget.is_critical is True

    def test_is_warning_at_70_pct(self):
        budget = TokenBudget(used=70, limit=100)
        assert budget.is_warning is True
        assert budget.is_critical is False

    def test_is_not_warning_below_70_pct(self):
        budget = TokenBudget(used=69, limit=100)
        assert budget.is_warning is False

    def test_zero_limit_does_not_raise(self):
        budget = TokenBudget(used=0, limit=0)
        assert budget.usage_pct == 0.0
        assert budget.remaining == 0


class TestGetContextWindow:
    def test_falls_back_to_default_when_endpoint_access_raises(self):
        class BadConfig:
            @property
            def context_window(self):
                raise AttributeError("no context_window")

        class BadEndpoint:
            config = BadConfig()

        class FakeBranch:
            class chat_model:
                endpoint = BadEndpoint()

        result = get_context_window(FakeBranch())
        assert result == 128_000

    def test_falls_back_to_default_when_chat_model_attribute_missing(self):
        class FakeBranch:
            class chat_model:
                @property
                def endpoint(self):
                    raise AttributeError("no endpoint")

        result = get_context_window(FakeBranch())
        assert result == 128_000

    def test_respects_explicit_context_window_in_config(self):
        class FakeBranch:
            class chat_model:
                endpoint = types.SimpleNamespace(
                    config=types.SimpleNamespace(
                        context_window=32_000,
                        kwargs={},
                        provider="openai",
                    )
                )

        result = get_context_window(FakeBranch())
        assert result == 32_000
