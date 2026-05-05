# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi/beta/rules/common/mapping.py — MappingRule."""

from __future__ import annotations

import pytest

from lionagi._errors import ValidationError
from lionagi.beta.rules.common.mapping import MappingRule

# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


class TestMappingRuleValidate:
    @pytest.mark.asyncio
    async def test_validate_plain_dict_ok(self):
        rule = MappingRule()
        await rule.validate({"a": 1}, dict)  # no exception

    @pytest.mark.asyncio
    async def test_validate_non_dict_raises(self):
        rule = MappingRule()
        with pytest.raises(ValueError, match="expected dict/Mapping"):
            await rule.validate("not a dict", dict)

    @pytest.mark.asyncio
    async def test_validate_non_dict_list_raises(self):
        rule = MappingRule()
        with pytest.raises(ValueError):
            await rule.validate([1, 2, 3], dict)

    @pytest.mark.asyncio
    async def test_validate_missing_required_keys_raises(self):
        rule = MappingRule(required_keys={"name", "value"})
        with pytest.raises(ValueError, match="Missing required keys"):
            await rule.validate({"name": "test"}, dict)

    @pytest.mark.asyncio
    async def test_validate_all_required_keys_present_ok(self):
        rule = MappingRule(required_keys={"name", "value"})
        await rule.validate({"name": "test", "value": 42}, dict)

    @pytest.mark.asyncio
    async def test_validate_extra_keys_ok_no_restriction(self):
        rule = MappingRule(required_keys={"name"})
        await rule.validate({"name": "test", "extra": True}, dict)

    @pytest.mark.asyncio
    async def test_validate_mapping_subtype_ok(self):
        from collections import OrderedDict

        rule = MappingRule()
        await rule.validate(OrderedDict(a=1), dict)


# ---------------------------------------------------------------------------
# perform_fix()
# ---------------------------------------------------------------------------


class TestMappingRulePerformFix:
    @pytest.mark.asyncio
    async def test_perform_fix_json_string_to_dict(self):
        rule = MappingRule()
        result = await rule.perform_fix('{"key": "value"}', dict)
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_perform_fix_invalid_json_raises(self):
        rule = MappingRule()
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            await rule.perform_fix("{not valid json}", dict)

    @pytest.mark.asyncio
    async def test_perform_fix_non_string_non_mapping_raises(self):
        rule = MappingRule()
        with pytest.raises(ValueError, match="Cannot convert"):
            await rule.perform_fix(12345, dict)

    @pytest.mark.asyncio
    async def test_perform_fix_list_raises(self):
        rule = MappingRule()
        with pytest.raises(ValueError, match="Cannot convert"):
            await rule.perform_fix([1, 2], dict)

    @pytest.mark.asyncio
    async def test_perform_fix_fuzzy_keys_normalizes_case(self):
        rule = MappingRule(
            required_keys={"name", "value"},
            fuzzy_keys=True,
        )
        # Input has "Name" and "Value" — should normalize to canonical keys
        result = await rule.perform_fix({"Name": "test", "Value": 42}, dict)
        assert "name" in result
        assert "value" in result

    @pytest.mark.asyncio
    async def test_perform_fix_fuzzy_from_json_string(self):
        rule = MappingRule(
            required_keys={"name"},
            fuzzy_keys=True,
        )
        result = await rule.perform_fix('{"Name": "hello"}', dict)
        assert "name" in result
        assert result["name"] == "hello"

    @pytest.mark.asyncio
    async def test_perform_fix_missing_required_key_raises_after_normalize(self):
        rule = MappingRule(required_keys={"name", "value"}, fuzzy_keys=True)
        # Only one required key present even after normalization
        with pytest.raises(ValueError, match="Missing required keys"):
            await rule.perform_fix({"Name": "test"}, dict)


# ---------------------------------------------------------------------------
# invoke()
# ---------------------------------------------------------------------------


class TestMappingRuleInvoke:
    @pytest.mark.asyncio
    async def test_invoke_valid_dict_returns_dict(self):
        rule = MappingRule()
        result = await rule.invoke("config", {"x": 1}, dict, auto_fix=False)
        assert result == {"x": 1}

    @pytest.mark.asyncio
    async def test_invoke_json_string_auto_fix_returns_dict(self):
        rule = MappingRule()
        result = await rule.invoke("config", '{"x": 1}', dict, auto_fix=True)
        assert result == {"x": 1}

    @pytest.mark.asyncio
    async def test_invoke_invalid_string_auto_fix_raises(self):
        rule = MappingRule()
        with pytest.raises(ValidationError):
            await rule.invoke("config", "not json", dict, auto_fix=True)

    @pytest.mark.asyncio
    async def test_invoke_non_mapping_no_auto_fix_raises(self):
        rule = MappingRule()
        with pytest.raises(ValidationError):
            await rule.invoke("config", 42, dict, auto_fix=False)
