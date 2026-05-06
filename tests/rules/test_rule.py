# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for beta/rules/rule.py — Rule, RuleParams, RuleQualifier."""

from __future__ import annotations

import pytest

from lionagi._errors import ValidationError
from lionagi.rules.rule import (
    Rule,
    RuleParams,
    RuleQualifier,
    _decide_qualifier_order,
)

# ---------------------------------------------------------------------------
# RuleQualifier
# ---------------------------------------------------------------------------


class TestRuleQualifier:
    def test_values(self):
        assert RuleQualifier.FIELD < RuleQualifier.ANNOTATION < RuleQualifier.CONDITION

    def test_from_str_field(self):
        assert RuleQualifier.from_str("field") == RuleQualifier.FIELD

    def test_from_str_annotation(self):
        assert RuleQualifier.from_str("ANNOTATION") == RuleQualifier.ANNOTATION

    def test_from_str_condition(self):
        assert RuleQualifier.from_str("condition") == RuleQualifier.CONDITION

    def test_from_str_with_whitespace(self):
        assert RuleQualifier.from_str("  FIELD  ") == RuleQualifier.FIELD

    def test_from_str_unknown_raises(self):
        with pytest.raises(ValueError):
            RuleQualifier.from_str("UNKNOWN")


# ---------------------------------------------------------------------------
# _decide_qualifier_order
# ---------------------------------------------------------------------------


class TestDecideQualifierOrder:
    def test_none_returns_default(self):
        order = _decide_qualifier_order(None)
        assert order == [
            RuleQualifier.FIELD,
            RuleQualifier.ANNOTATION,
            RuleQualifier.CONDITION,
        ]

    def test_field_first(self):
        order = _decide_qualifier_order(RuleQualifier.FIELD)
        assert order[0] == RuleQualifier.FIELD

    def test_annotation_first(self):
        order = _decide_qualifier_order(RuleQualifier.ANNOTATION)
        assert order[0] == RuleQualifier.ANNOTATION
        assert len(order) == 3

    def test_condition_first(self):
        order = _decide_qualifier_order(RuleQualifier.CONDITION)
        assert order[0] == RuleQualifier.CONDITION

    def test_from_str_qualifier(self):
        order = _decide_qualifier_order("field")
        assert order[0] == RuleQualifier.FIELD


# ---------------------------------------------------------------------------
# RuleParams
# ---------------------------------------------------------------------------


class TestRuleParams:
    def test_defaults(self):
        p = RuleParams()
        assert p.apply_types == set()
        assert p.apply_fields == set()
        assert p.auto_fix is False
        assert p.default_qualifier == RuleQualifier.FIELD

    def test_with_fields(self):
        p = RuleParams(apply_fields={"name", "age"})
        assert "name" in p.apply_fields

    def test_with_types(self):
        p = RuleParams(apply_types={str})
        assert str in p.apply_types

    def test_kw_default_empty(self):
        p = RuleParams()
        assert p.kw == {}


# ---------------------------------------------------------------------------
# Rule (concrete subclass for testing)
# ---------------------------------------------------------------------------


class StrictPositiveRule(Rule):
    """Validates that a numeric value is positive."""

    async def validate(self, v: float, t: type, **kw) -> None:
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError(f"Expected positive number, got {v!r}")

    async def perform_fix(self, v, t):
        return abs(v)


class TestRule:
    def setup_method(self):
        params = RuleParams(
            apply_types={int, float},
            apply_fields={"amount"},
            auto_fix=False,
        )
        self.rule = StrictPositiveRule(params)

    def test_repr(self):
        r = repr(self.rule)
        assert "StrictPositiveRule" in r

    def test_properties(self):
        assert self.rule.auto_fix is False
        assert int in self.rule.apply_types
        assert "amount" in self.rule.apply_fields

    @pytest.mark.asyncio
    async def test_apply_by_field(self):
        result = await self.rule.apply("amount", -1, int)
        assert result is True

    @pytest.mark.asyncio
    async def test_apply_by_annotation(self):
        result = await self.rule.apply("other_field", 5, int)
        assert result is True

    @pytest.mark.asyncio
    async def test_apply_no_match(self):
        result = await self.rule.apply("unknown_field", "hello", str)
        assert result is False

    @pytest.mark.asyncio
    async def test_invoke_valid_value(self):
        result = await self.rule.invoke("amount", 5.0, float)
        assert result == 5.0

    @pytest.mark.asyncio
    async def test_invoke_invalid_raises(self):
        with pytest.raises(ValidationError):
            await self.rule.invoke("amount", -1.0, float)

    @pytest.mark.asyncio
    async def test_invoke_with_auto_fix(self):
        params = RuleParams(apply_types={int, float}, auto_fix=True)
        rule = StrictPositiveRule(params)
        result = await rule.invoke("x", -5, int)
        assert result == 5

    @pytest.mark.asyncio
    async def test_rule_condition_not_implemented(self):
        with pytest.raises(NotImplementedError):
            await self.rule.rule_condition("k", "v", str)

    @pytest.mark.asyncio
    async def test_apply_explicit_qualifier_field(self):
        result = await self.rule.apply("amount", -1, int, qualifier=RuleQualifier.FIELD)
        assert result is True

    @pytest.mark.asyncio
    async def test_apply_explicit_qualifier_annotation(self):
        result = await self.rule.apply(
            "other", 5, int, qualifier=RuleQualifier.ANNOTATION
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_invoke_auto_fix_fails_raises(self):
        class AlwaysFailFix(Rule):
            async def validate(self, v, t, **kw):
                raise ValueError("always fails")

            async def perform_fix(self, v, t):
                raise RuntimeError("fix also fails")

        rule = AlwaysFailFix(RuleParams(auto_fix=True))
        with pytest.raises(ValidationError):
            await rule.invoke("k", "v", str)
