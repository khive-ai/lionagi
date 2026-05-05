# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi/beta/rules/common/choice.py — ChoiceRule."""

from __future__ import annotations

import pytest

from lionagi._errors import ValidationError
from lionagi.beta.rules.common.choice import ChoiceRule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_rule(choices, case_sensitive=True):
    return ChoiceRule(choices=choices, case_sensitive=case_sensitive)


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


class TestChoiceRuleValidate:
    @pytest.mark.asyncio
    async def test_valid_choice_passes(self):
        rule = make_rule({"low", "medium", "high"})
        await rule.validate("low", str)  # no exception

    @pytest.mark.asyncio
    async def test_invalid_choice_raises_value_error(self):
        rule = make_rule({"low", "medium", "high"})
        with pytest.raises(ValueError, match="Invalid choice"):
            await rule.validate("critical", str)

    @pytest.mark.asyncio
    async def test_case_sensitive_mismatch_raises(self):
        rule = make_rule({"low", "medium", "high"}, case_sensitive=True)
        with pytest.raises(ValueError):
            await rule.validate("HIGH", str)

    @pytest.mark.asyncio
    async def test_int_choice_valid(self):
        rule = make_rule({1, 2, 3})
        await rule.validate(2, int)

    @pytest.mark.asyncio
    async def test_int_choice_invalid(self):
        rule = make_rule({1, 2, 3})
        with pytest.raises(ValueError):
            await rule.validate(99, int)


# ---------------------------------------------------------------------------
# perform_fix()
# ---------------------------------------------------------------------------


class TestChoiceRulePerformFix:
    @pytest.mark.asyncio
    async def test_already_in_choices_returns_as_is(self):
        rule = make_rule({"low", "medium", "high"}, case_sensitive=False)
        result = await rule.perform_fix("low", str)
        assert result == "low"

    @pytest.mark.asyncio
    async def test_case_insensitive_fix_high_to_canonical(self):
        rule = make_rule({"low", "medium", "high"}, case_sensitive=False)
        result = await rule.perform_fix("HIGH", str)
        assert result == "high"

    @pytest.mark.asyncio
    async def test_case_insensitive_fix_mixed_case(self):
        rule = make_rule({"Low", "Medium", "High"}, case_sensitive=False)
        result = await rule.perform_fix("low", str)
        assert result == "Low"

    @pytest.mark.asyncio
    async def test_case_sensitive_unfixable_raises(self):
        rule = make_rule({"low", "medium", "high"}, case_sensitive=True)
        with pytest.raises(ValueError, match="Cannot fix"):
            await rule.perform_fix("HIGH", str)

    @pytest.mark.asyncio
    async def test_case_insensitive_completely_wrong_raises(self):
        rule = make_rule({"low", "medium", "high"}, case_sensitive=False)
        with pytest.raises(ValueError, match="Cannot fix"):
            await rule.perform_fix("critical", str)

    @pytest.mark.asyncio
    async def test_non_string_not_in_choices_raises(self):
        rule = make_rule({1, 2, 3}, case_sensitive=False)
        with pytest.raises(ValueError):
            await rule.perform_fix(99, int)


# ---------------------------------------------------------------------------
# invoke()
# ---------------------------------------------------------------------------


class TestChoiceRuleInvoke:
    @pytest.mark.asyncio
    async def test_invoke_valid_auto_fix_true_returns_as_is(self):
        rule = make_rule({"low", "medium", "high"})
        result = await rule.invoke("priority", "low", str, auto_fix=True)
        assert result == "low"

    @pytest.mark.asyncio
    async def test_invoke_case_mismatch_auto_fix_true_returns_canonical(self):
        rule = make_rule({"low", "medium", "high"}, case_sensitive=False)
        result = await rule.invoke("priority", "HIGH", str, auto_fix=True)
        assert result == "high"

    @pytest.mark.asyncio
    async def test_invoke_invalid_auto_fix_true_raises_validation_error(self):
        rule = make_rule({"low", "medium", "high"})
        with pytest.raises(ValidationError):
            await rule.invoke("priority", "critical", str, auto_fix=True)

    @pytest.mark.asyncio
    async def test_invoke_invalid_auto_fix_false_raises_validation_error(self):
        rule = make_rule({"low", "medium", "high"})
        with pytest.raises(ValidationError):
            await rule.invoke("priority", "critical", str, auto_fix=False)

    @pytest.mark.asyncio
    async def test_invoke_case_mismatch_auto_fix_false_raises(self):
        rule = make_rule({"low", "medium", "high"}, case_sensitive=False)
        with pytest.raises(ValidationError):
            await rule.invoke("priority", "HIGH", str, auto_fix=False)

    @pytest.mark.asyncio
    async def test_default_auto_fix_is_true(self):
        # ChoiceRule sets auto_fix=True in default params
        rule = make_rule({"a", "b"}, case_sensitive=False)
        result = await rule.invoke("x", "A", str)
        assert result == "a"
