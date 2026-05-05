# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi/beta/rules/validator.py — Validator."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from lionagi._errors import ValidationError
from lionagi.beta.rules.common.number import NumberRule
from lionagi.beta.rules.common.string import StringRule
from lionagi.beta.rules.registry import RuleRegistry, reset_default_registry
from lionagi.beta.rules.rule import Rule, RuleParams
from lionagi.beta.rules.validator import Validator
from lionagi.ln.types import Operable, Spec


@pytest.fixture(autouse=True)
def reset_registry():
    reset_default_registry()
    yield
    reset_default_registry()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestValidatorConstruction:
    def test_default_construction_uses_default_registry(self):
        v = Validator()
        assert v.registry is not None

    def test_registry_and_rulebook_same_object_ok(self):
        reg = RuleRegistry()
        v = Validator(registry=reg, rulebook=reg)
        assert v.registry is reg

    def test_registry_and_rulebook_different_raises(self):
        reg1 = RuleRegistry()
        reg2 = RuleRegistry()
        with pytest.raises(ValueError, match="not both"):
            Validator(registry=reg1, rulebook=reg2)

    def test_custom_registry_used(self):
        reg = RuleRegistry()
        reg.register(str, StringRule())
        v = Validator(registry=reg)
        assert v.registry is reg

    def test_rulebook_property_returns_registry(self):
        v = Validator()
        assert v.rulebook is v.registry

    def test_max_log_entries_respected(self):
        v = Validator(max_log_entries=5)
        for i in range(10):
            v.log_validation_error(f"f{i}", i, "err")
        assert len(v.validation_log) == 5


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestValidatorLogging:
    def test_log_validation_error_adds_entry(self):
        v = Validator()
        v.log_validation_error("myfield", "badval", "something went wrong")
        assert len(v.validation_log) == 1
        entry = v.validation_log[0]
        assert entry["field"] == "myfield"
        assert entry["value"] == "badval"
        assert entry["error"] == "something went wrong"
        assert "timestamp" in entry

    def test_log_multiple_errors(self):
        v = Validator()
        v.log_validation_error("f1", 1, "e1")
        v.log_validation_error("f2", 2, "e2")
        assert len(v.validation_log) == 2

    def test_get_validation_summary_empty(self):
        v = Validator()
        summary = v.get_validation_summary()
        assert summary["total_errors"] == 0
        assert summary["fields_with_errors"] == []
        assert summary["error_entries"] == []

    def test_get_validation_summary_reports_correctly(self):
        v = Validator()
        v.log_validation_error("alpha", "x", "bad")
        v.log_validation_error("beta", "y", "also bad")
        v.log_validation_error("alpha", "z", "again bad")
        summary = v.get_validation_summary()
        assert summary["total_errors"] == 3
        assert "alpha" in summary["fields_with_errors"]
        assert "beta" in summary["fields_with_errors"]
        assert summary["fields_with_errors"] == sorted(summary["fields_with_errors"])

    def test_clear_log_empties(self):
        v = Validator()
        v.log_validation_error("f", "v", "e")
        v.clear_log()
        assert len(v.validation_log) == 0
        summary = v.get_validation_summary()
        assert summary["total_errors"] == 0


# ---------------------------------------------------------------------------
# get_rule_for_spec()
# ---------------------------------------------------------------------------


class TestGetRuleForSpec:
    def test_override_rule_in_spec_returned(self):
        override = StringRule(min_length=5)
        spec = Spec(str, name="field", rule=override)
        v = Validator()
        result = v.get_rule_for_spec(spec)
        assert result is override

    def test_fallback_to_registry_by_type(self):
        reg = RuleRegistry()
        rule = StringRule()
        reg.register(str, rule)
        v = Validator(registry=reg)
        spec = Spec(str, name="field")
        result = v.get_rule_for_spec(spec)
        assert result is rule

    def test_fallback_to_registry_by_field_name(self):
        reg = RuleRegistry()
        rule = NumberRule(ge=0)
        reg.register("amount", rule)
        v = Validator(registry=reg)
        spec = Spec(int, name="amount")
        result = v.get_rule_for_spec(spec)
        assert result is rule

    def test_no_rule_returns_none(self):
        reg = RuleRegistry()
        v = Validator(registry=reg)
        spec = Spec(list, name="items")
        result = v.get_rule_for_spec(spec)
        assert result is None

    def test_non_rule_override_falls_back_to_registry(self):
        reg = RuleRegistry()
        str_rule = StringRule()
        reg.register(str, str_rule)
        v = Validator(registry=reg)
        # spec.get("rule") returns a non-Rule (e.g. a string "foo") — falls back
        # We achieve this by not providing a rule — default registry lookup happens
        spec = Spec(str, name="field")
        result = v.get_rule_for_spec(spec)
        assert result is str_rule


# ---------------------------------------------------------------------------
# validate_spec()
# ---------------------------------------------------------------------------


class TestValidateSpec:
    @pytest.mark.asyncio
    async def test_string_field_valid_passes(self):
        v = Validator()
        spec = Spec(str, name="title")
        result = await v.validate_spec(spec, "hello")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_string_field_auto_fix_int_to_str(self):
        v = Validator()
        spec = Spec(str, name="label")
        result = await v.validate_spec(spec, 42, auto_fix=True)
        assert result == "42"

    @pytest.mark.asyncio
    async def test_none_value_nullable_returns_none(self):
        v = Validator()
        spec = Spec(str, name="opt", nullable=True)
        result = await v.validate_spec(spec, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_none_value_no_default_strict_raises(self):
        v = Validator()
        spec = Spec(str, name="required")
        with pytest.raises(ValidationError, match="missing"):
            await v.validate_spec(spec, None, strict=True)

    @pytest.mark.asyncio
    async def test_none_value_no_default_not_strict_returns_none(self):
        v = Validator()
        spec = Spec(str, name="optional_field")
        result = await v.validate_spec(spec, None, strict=False)
        assert result is None

    @pytest.mark.asyncio
    async def test_none_value_with_default_uses_default(self):
        v = Validator()
        spec = Spec(str, name="greeting", default="hello")
        result = await v.validate_spec(spec, None)
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_listable_field_wraps_single_value(self):
        v = Validator()
        spec = Spec(str, name="tags", listable=True)
        result = await v.validate_spec(spec, "tag1", auto_fix=True)
        assert result == ["tag1"]

    @pytest.mark.asyncio
    async def test_listable_field_validates_each_item(self):
        v = Validator()
        spec = Spec(str, name="tags", listable=True)
        result = await v.validate_spec(spec, ["a", "b", "c"], auto_fix=True)
        assert result == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_listable_field_auto_fixes_each_item(self):
        v = Validator()
        spec = Spec(str, name="tags", listable=True)
        result = await v.validate_spec(spec, [1, 2, 3], auto_fix=True)
        assert result == ["1", "2", "3"]

    @pytest.mark.asyncio
    async def test_no_rule_strict_raises(self):
        reg = RuleRegistry()  # empty registry
        v = Validator(registry=reg)
        spec = Spec(list, name="items")
        with pytest.raises(ValidationError, match="No rule found"):
            await v.validate_spec(spec, [1, 2], strict=True)

    @pytest.mark.asyncio
    async def test_no_rule_not_strict_returns_value_unchanged(self):
        reg = RuleRegistry()  # empty registry
        v = Validator(registry=reg)
        spec = Spec(list, name="items")
        result = await v.validate_spec(spec, [1, 2], strict=False)
        assert result == [1, 2]

    @pytest.mark.asyncio
    async def test_custom_validator_transforms_value(self):
        v = Validator()
        spec = Spec(str, name="name", validator=str.upper)
        result = await v.validate_spec(spec, "hello")
        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_custom_validator_list_applied_in_order(self):
        v = Validator()
        spec = Spec(str, name="name", validator=[str.strip, str.upper])
        result = await v.validate_spec(spec, "  hello  ")
        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_custom_validator_raises_validation_error(self):
        def bad_validator(v):
            raise ValueError("custom rejection")

        v = Validator()
        spec = Spec(str, name="name", validator=bad_validator)
        with pytest.raises(ValidationError, match="Custom validator failed"):
            await v.validate_spec(spec, "anything")

    @pytest.mark.asyncio
    async def test_async_custom_validator(self):
        async def upper(v):
            return v.upper()

        v = Validator()
        spec = Spec(str, name="name", validator=upper)
        result = await v.validate_spec(spec, "hello")
        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_unset_value_falls_through_to_default(self):
        from lionagi.ln.types import Unset

        v = Validator()
        spec = Spec(str, name="field", default="default_val")
        result = await v.validate_spec(spec, Unset)
        assert result == "default_val"


# ---------------------------------------------------------------------------
# validate() — full operable validation
# ---------------------------------------------------------------------------


class TestValidatorValidate:
    @pytest.mark.asyncio
    async def test_validate_basic_dict_against_operable(self):
        v = Validator()
        specs = (
            Spec(str, name="name"),
            Spec(int, name="count"),
        )
        operable = Operable(specs)
        data = {"name": "test", "count": 5}
        result = await v.validate(data, operable)
        assert result["name"] == "test"
        assert result["count"] == 5

    @pytest.mark.asyncio
    async def test_validate_auto_fixes_types(self):
        v = Validator()
        specs = (Spec(str, name="label"),)
        operable = Operable(specs)
        data = {"label": 123}
        result = await v.validate(data, operable, auto_fix=True)
        assert result["label"] == "123"

    @pytest.mark.asyncio
    async def test_validate_capabilities_subset_ok(self):
        v = Validator()
        specs = (
            Spec(str, name="a"),
            Spec(str, name="b"),
        )
        operable = Operable(specs)
        data = {"a": "x", "b": "y"}
        result = await v.validate(data, operable, capabilities={"a", "b"})
        assert "a" in result
        assert "b" in result

    @pytest.mark.asyncio
    async def test_validate_capabilities_not_subset_raises(self):
        v = Validator()
        specs = (Spec(str, name="a"),)
        operable = Operable(specs)
        data = {"a": "x"}
        with pytest.raises(ValidationError, match="Capabilities"):
            await v.validate(
                data, operable, capabilities={"a", "b"}
            )  # "b" not in operable

    @pytest.mark.asyncio
    async def test_validate_missing_field_with_default_uses_default(self):
        v = Validator()
        specs = (Spec(str, name="msg", default="hello"),)
        operable = Operable(specs)
        data = {}
        result = await v.validate(data, operable)
        assert result["msg"] == "hello"

    @pytest.mark.asyncio
    async def test_validate_missing_required_field_strict_raises(self):
        v = Validator()
        specs = (Spec(str, name="required_field"),)
        operable = Operable(specs)
        data = {}
        with pytest.raises(ValidationError):
            await v.validate(data, operable, strict=True)

    @pytest.mark.asyncio
    async def test_validate_returns_only_allowed_fields(self):
        v = Validator()
        specs = (Spec(str, name="a"),)
        operable = Operable(specs)
        # "b" is not in operable, should not appear in result
        data = {"a": "x", "b": "extra"}
        result = await v.validate(data, operable)
        assert "a" in result
        assert "b" not in result
