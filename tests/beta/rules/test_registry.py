# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi/beta/rules/registry.py — RuleRegistry, get_default_registry, reset."""

from __future__ import annotations

import pytest

from lionagi.beta.rules.common.boolean import BooleanRule
from lionagi.beta.rules.common.mapping import MappingRule
from lionagi.beta.rules.common.model import BaseModelRule
from lionagi.beta.rules.common.number import NumberRule
from lionagi.beta.rules.common.string import StringRule
from lionagi.beta.rules.registry import (
    RuleBook,
    RuleRegistry,
    get_default_registry,
    reset_default_registry,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure the singleton is reset before and after each test."""
    reset_default_registry()
    yield
    reset_default_registry()


# ---------------------------------------------------------------------------
# Basic registration
# ---------------------------------------------------------------------------


class TestRuleRegistryRegister:
    def test_register_str_key_and_get_by_field_name(self):
        reg = RuleRegistry()
        rule = StringRule()
        reg.register("my_field", rule)
        result = reg.get_rule(field_name="my_field")
        assert result is rule

    def test_register_type_key_and_get_by_type(self):
        reg = RuleRegistry()
        rule = NumberRule()
        reg.register(int, rule)
        result = reg.get_rule(base_type=int)
        assert result is rule

    def test_register_duplicate_str_raises_without_replace(self):
        reg = RuleRegistry()
        reg.register("field", StringRule())
        with pytest.raises(ValueError, match="already registered"):
            reg.register("field", StringRule())

    def test_register_duplicate_type_raises_without_replace(self):
        reg = RuleRegistry()
        reg.register(int, NumberRule())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(int, NumberRule())

    def test_register_duplicate_str_with_replace_ok(self):
        reg = RuleRegistry()
        rule1 = StringRule(min_length=1)
        rule2 = StringRule(min_length=5)
        reg.register("field", rule1)
        reg.register("field", rule2, replace=True)
        assert reg.get_rule(field_name="field") is rule2

    def test_register_duplicate_type_with_replace_ok(self):
        reg = RuleRegistry()
        rule1 = NumberRule()
        rule2 = NumberRule(ge=0)
        reg.register(int, rule1)
        reg.register(int, rule2, replace=True)
        assert reg.get_rule(base_type=int) is rule2


# ---------------------------------------------------------------------------
# Lookup priority and inheritance
# ---------------------------------------------------------------------------


class TestRuleRegistryGetRule:
    def test_field_name_takes_priority_over_type(self):
        reg = RuleRegistry()
        name_rule = StringRule(min_length=1)
        type_rule = StringRule(min_length=10)
        reg.register("myfield", name_rule)
        reg.register(str, type_rule)
        # field_name priority over base_type
        result = reg.get_rule(base_type=str, field_name="myfield")
        assert result is name_rule

    def test_exact_type_found(self):
        reg = RuleRegistry()
        rule = NumberRule()
        reg.register(float, rule)
        assert reg.get_rule(base_type=float) is rule

    def test_subclass_inherits_parent_rule(self):
        reg = RuleRegistry()
        rule = StringRule()
        reg.register(str, rule)

        class MyStr(str):
            pass

        result = reg.get_rule(base_type=MyStr)
        assert result is rule

    def test_get_rule_returns_none_when_not_found(self):
        reg = RuleRegistry()
        assert reg.get_rule(base_type=list) is None
        assert reg.get_rule(field_name="nonexistent") is None

    def test_get_rule_no_args_returns_none(self):
        reg = RuleRegistry()
        reg.register(str, StringRule())
        assert reg.get_rule() is None


# ---------------------------------------------------------------------------
# has_rule, list_types, list_names
# ---------------------------------------------------------------------------


class TestRuleRegistryQueries:
    def test_has_rule_for_str_key(self):
        reg = RuleRegistry()
        reg.register("amount", NumberRule())
        assert reg.has_rule("amount") is True
        assert reg.has_rule("other") is False

    def test_has_rule_for_type_key(self):
        reg = RuleRegistry()
        reg.register(bool, BooleanRule())
        assert reg.has_rule(bool) is True
        assert reg.has_rule(str) is False

    def test_list_types(self):
        reg = RuleRegistry()
        reg.register(str, StringRule())
        reg.register(int, NumberRule())
        types = reg.list_types()
        assert str in types
        assert int in types
        assert len(types) == 2

    def test_list_names(self):
        reg = RuleRegistry()
        reg.register("alpha", StringRule())
        reg.register("beta", NumberRule())
        names = reg.list_names()
        assert "alpha" in names
        assert "beta" in names
        assert len(names) == 2

    def test_list_types_empty(self):
        reg = RuleRegistry()
        assert reg.list_types() == []

    def test_list_names_empty(self):
        reg = RuleRegistry()
        assert reg.list_names() == []


# ---------------------------------------------------------------------------
# Singleton and reset
# ---------------------------------------------------------------------------


class TestDefaultRegistry:
    def test_get_default_registry_returns_same_instance(self):
        r1 = get_default_registry()
        r2 = get_default_registry()
        assert r1 is r2

    def test_get_default_registry_has_str_rule(self):
        reg = get_default_registry()
        rule = reg.get_rule(base_type=str)
        assert isinstance(rule, StringRule)

    def test_get_default_registry_has_int_rule(self):
        reg = get_default_registry()
        rule = reg.get_rule(base_type=int)
        assert isinstance(rule, NumberRule)

    def test_get_default_registry_has_float_rule(self):
        reg = get_default_registry()
        rule = reg.get_rule(base_type=float)
        assert isinstance(rule, NumberRule)

    def test_get_default_registry_has_bool_rule(self):
        reg = get_default_registry()
        rule = reg.get_rule(base_type=bool)
        assert isinstance(rule, BooleanRule)

    def test_get_default_registry_has_dict_rule(self):
        reg = get_default_registry()
        rule = reg.get_rule(base_type=dict)
        assert isinstance(rule, MappingRule)

    def test_get_default_registry_has_basemodel_rule(self):
        from pydantic import BaseModel

        reg = get_default_registry()
        rule = reg.get_rule(base_type=BaseModel)
        assert isinstance(rule, BaseModelRule)

    def test_reset_default_registry_forces_fresh_creation(self):
        r1 = get_default_registry()
        reset_default_registry()
        r2 = get_default_registry()
        assert r1 is not r2

    def test_reset_then_fresh_has_default_rules(self):
        reset_default_registry()
        reg = get_default_registry()
        assert isinstance(reg.get_rule(base_type=str), StringRule)


# ---------------------------------------------------------------------------
# RuleBook alias
# ---------------------------------------------------------------------------


class TestRuleBookAlias:
    def test_rulebook_is_ruleregistry(self):
        assert RuleBook is RuleRegistry

    def test_rulebook_instance_works(self):
        rb = RuleBook()
        rb.register("field", StringRule())
        assert rb.has_rule("field") is True
