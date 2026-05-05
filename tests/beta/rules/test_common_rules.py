# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for beta/rules/common — BooleanRule, NumberRule, StringRule."""

from __future__ import annotations

import pytest

from lionagi._errors import ValidationError
from lionagi.beta.rules.common.boolean import BooleanRule
from lionagi.beta.rules.common.number import NumberRule
from lionagi.beta.rules.common.string import StringRule

# ---------------------------------------------------------------------------
# BooleanRule
# ---------------------------------------------------------------------------


class TestBooleanRule:
    def setup_method(self):
        self.rule = BooleanRule()

    @pytest.mark.asyncio
    async def test_validate_true(self):
        await self.rule.validate(True, bool)

    @pytest.mark.asyncio
    async def test_validate_false(self):
        await self.rule.validate(False, bool)

    @pytest.mark.asyncio
    async def test_validate_string_raises(self):
        with pytest.raises(ValueError):
            await self.rule.validate("true", bool)

    @pytest.mark.asyncio
    async def test_perform_fix_true_strings(self):
        for s in ["true", "yes", "1", "on", "TRUE", "YES"]:
            result = await self.rule.perform_fix(s, bool)
            assert result is True, f"Expected True for {s!r}"

    @pytest.mark.asyncio
    async def test_perform_fix_false_strings(self):
        for s in ["false", "no", "0", "off", "FALSE", "NO"]:
            result = await self.rule.perform_fix(s, bool)
            assert result is False, f"Expected False for {s!r}"

    @pytest.mark.asyncio
    async def test_perform_fix_invalid_string_raises(self):
        with pytest.raises(ValueError):
            await self.rule.perform_fix("maybe", bool)

    @pytest.mark.asyncio
    async def test_perform_fix_int_zero(self):
        result = await self.rule.perform_fix(0, bool)
        assert result is False

    @pytest.mark.asyncio
    async def test_perform_fix_int_nonzero(self):
        result = await self.rule.perform_fix(1, bool)
        assert result is True

    @pytest.mark.asyncio
    async def test_invoke_valid(self):
        result = await self.rule.invoke("active", True, bool)
        assert result is True

    @pytest.mark.asyncio
    async def test_invoke_auto_fix_string(self):
        # BooleanRule has auto_fix=True by default
        result = await self.rule.invoke("active", "true", bool)
        assert result is True

    @pytest.mark.asyncio
    async def test_invoke_unfixable_raises(self):
        with pytest.raises(ValidationError):
            await self.rule.invoke("active", "maybe", bool)


# ---------------------------------------------------------------------------
# NumberRule
# ---------------------------------------------------------------------------


class TestNumberRule:
    def setup_method(self):
        self.rule = NumberRule()

    @pytest.mark.asyncio
    async def test_validate_int(self):
        await self.rule.validate(5, int)

    @pytest.mark.asyncio
    async def test_validate_float(self):
        await self.rule.validate(3.14, float)

    @pytest.mark.asyncio
    async def test_validate_string_raises(self):
        with pytest.raises(ValueError):
            await self.rule.validate("42", float)

    @pytest.mark.asyncio
    async def test_validate_ge_passes(self):
        rule = NumberRule(ge=0.0)
        await rule.validate(0.0, float)
        await rule.validate(1.0, float)

    @pytest.mark.asyncio
    async def test_validate_ge_fails(self):
        rule = NumberRule(ge=0.0)
        with pytest.raises(ValueError):
            await rule.validate(-0.1, float)

    @pytest.mark.asyncio
    async def test_validate_gt_fails(self):
        rule = NumberRule(gt=0.0)
        with pytest.raises(ValueError):
            await rule.validate(0.0, float)

    @pytest.mark.asyncio
    async def test_validate_le_passes(self):
        rule = NumberRule(le=10.0)
        await rule.validate(10.0, float)

    @pytest.mark.asyncio
    async def test_validate_le_fails(self):
        rule = NumberRule(le=10.0)
        with pytest.raises(ValueError):
            await rule.validate(10.1, float)

    @pytest.mark.asyncio
    async def test_validate_lt_fails(self):
        rule = NumberRule(lt=10.0)
        with pytest.raises(ValueError):
            await rule.validate(10.0, float)

    @pytest.mark.asyncio
    async def test_perform_fix_from_string_float(self):
        result = await self.rule.perform_fix("3.14", float)
        assert result == 3.14

    @pytest.mark.asyncio
    async def test_perform_fix_from_string_int(self):
        result = await self.rule.perform_fix("5", int)
        assert result == 5

    @pytest.mark.asyncio
    async def test_perform_fix_invalid_raises(self):
        with pytest.raises(ValueError):
            await self.rule.perform_fix("not_a_number", float)

    @pytest.mark.asyncio
    async def test_perform_fix_validates_constraints(self):
        rule = NumberRule(ge=0.0)
        with pytest.raises((ValueError, ValidationError)):
            await rule.perform_fix("-5", float)

    @pytest.mark.asyncio
    async def test_invoke_auto_fix_string(self):
        # NumberRule has auto_fix=True by default
        result = await self.rule.invoke("amount", "42", float)
        assert result == 42.0

    @pytest.mark.asyncio
    async def test_invoke_valid(self):
        result = await self.rule.invoke("amount", 42.0, float)
        assert result == 42.0


# ---------------------------------------------------------------------------
# StringRule
# ---------------------------------------------------------------------------


class TestStringRule:
    def setup_method(self):
        self.rule = StringRule()

    @pytest.mark.asyncio
    async def test_validate_valid_string(self):
        await self.rule.validate("hello", str)

    @pytest.mark.asyncio
    async def test_validate_non_string_raises(self):
        with pytest.raises(ValueError):
            await self.rule.validate(42, str)

    @pytest.mark.asyncio
    async def test_min_length_passes(self):
        rule = StringRule(min_length=3)
        await rule.validate("abc", str)

    @pytest.mark.asyncio
    async def test_min_length_fails(self):
        rule = StringRule(min_length=5)
        with pytest.raises(ValueError):
            await rule.validate("hi", str)

    @pytest.mark.asyncio
    async def test_max_length_passes(self):
        rule = StringRule(max_length=10)
        await rule.validate("hello", str)

    @pytest.mark.asyncio
    async def test_max_length_fails(self):
        rule = StringRule(max_length=3)
        with pytest.raises(ValueError):
            await rule.validate("toolong", str)

    @pytest.mark.asyncio
    async def test_pattern_match(self):
        rule = StringRule(pattern=r"^[A-Za-z]+$")
        await rule.validate("Hello", str)

    @pytest.mark.asyncio
    async def test_pattern_no_match(self):
        rule = StringRule(pattern=r"^[A-Za-z]+$")
        with pytest.raises(ValueError):
            await rule.validate("Hello123", str)

    @pytest.mark.asyncio
    async def test_pattern_too_long_input(self):
        rule = StringRule(pattern=r"^.*$", regex_max_input_length=5)
        with pytest.raises(ValueError):
            await rule.validate("toolong", str)

    def test_redos_pattern_raises(self):
        with pytest.raises(ValueError):
            StringRule(pattern="(.*)* ")

    def test_invalid_regex_raises(self):
        with pytest.raises(ValueError):
            StringRule(pattern="[unclosed")

    @pytest.mark.asyncio
    async def test_perform_fix_converts_int(self):
        result = await self.rule.perform_fix(42, str)
        assert result == "42"

    @pytest.mark.asyncio
    async def test_perform_fix_validates_constraints(self):
        rule = StringRule(min_length=10)
        with pytest.raises((ValueError, ValidationError)):
            await rule.perform_fix("short", str)

    @pytest.mark.asyncio
    async def test_invoke_auto_fix_int(self):
        result = await self.rule.invoke("name", 99, str)
        assert result == "99"

    @pytest.mark.asyncio
    async def test_invoke_valid(self):
        result = await self.rule.invoke("name", "Alice", str)
        assert result == "Alice"
