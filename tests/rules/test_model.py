# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi/beta/rules/common/model.py — BaseModelRule."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from lionagi._errors import ValidationError
from lionagi.rules.common.model import BaseModelRule

# ---------------------------------------------------------------------------
# Test model
# ---------------------------------------------------------------------------


class MyModel(BaseModel):
    name: str
    value: int


class OtherModel(BaseModel):
    x: float


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


class TestBaseModelRuleValidate:
    @pytest.mark.asyncio
    async def test_validate_correct_instance_passes(self):
        rule = BaseModelRule()
        instance = MyModel(name="test", value=42)
        await rule.validate(instance, MyModel)  # no exception

    @pytest.mark.asyncio
    async def test_validate_correct_dict_passes(self):
        rule = BaseModelRule()
        await rule.validate({"name": "test", "value": 42}, MyModel)

    @pytest.mark.asyncio
    async def test_validate_invalid_dict_raises(self):
        rule = BaseModelRule()
        with pytest.raises(ValueError, match="Dict validation failed"):
            await rule.validate(
                {"name": "test", "value": "not_an_int_coerceable_fail"}, MyModel
            )

    @pytest.mark.asyncio
    async def test_validate_wrong_dict_missing_key_raises(self):
        rule = BaseModelRule()
        with pytest.raises(ValueError):
            await rule.validate({"name": "test"}, MyModel)  # missing 'value'

    @pytest.mark.asyncio
    async def test_validate_non_model_non_dict_raises(self):
        rule = BaseModelRule()
        with pytest.raises(ValueError, match="Cannot validate"):
            await rule.validate("a string", MyModel)

    @pytest.mark.asyncio
    async def test_validate_t_not_basemodel_subclass_raises(self):
        rule = BaseModelRule()
        with pytest.raises(ValueError, match="BaseModel subclass"):
            await rule.validate("x", str)

    @pytest.mark.asyncio
    async def test_validate_t_plain_type_not_basemodel_raises(self):
        rule = BaseModelRule()
        with pytest.raises(ValueError, match="BaseModel subclass"):
            await rule.validate(42, int)

    @pytest.mark.asyncio
    async def test_validate_wrong_instance_type_raises(self):
        rule = BaseModelRule()
        other = OtherModel(x=1.0)
        with pytest.raises(ValueError):
            await rule.validate(other, MyModel)

    @pytest.mark.asyncio
    async def test_validate_dict_that_pydantic_coerces_ok(self):
        # Pydantic coerces "42" -> 42 for int fields
        rule = BaseModelRule()
        await rule.validate({"name": "test", "value": "42"}, MyModel)


# ---------------------------------------------------------------------------
# perform_fix()
# ---------------------------------------------------------------------------


class TestBaseModelRulePerformFix:
    @pytest.mark.asyncio
    async def test_perform_fix_from_valid_dict_returns_instance(self):
        rule = BaseModelRule()
        result = await rule.perform_fix({"name": "test", "value": 42}, MyModel)
        assert isinstance(result, MyModel)
        assert result.name == "test"
        assert result.value == 42

    @pytest.mark.asyncio
    async def test_perform_fix_already_correct_instance_returns_it(self):
        rule = BaseModelRule()
        instance = MyModel(name="x", value=1)
        result = await rule.perform_fix(instance, MyModel)
        assert result is instance

    @pytest.mark.asyncio
    async def test_perform_fix_invalid_dict_raises(self):
        rule = BaseModelRule()
        with pytest.raises(ValueError, match="Cannot convert"):
            await rule.perform_fix({"name": "test"}, MyModel)  # missing value

    @pytest.mark.asyncio
    async def test_perform_fix_non_dict_non_model_raises(self):
        rule = BaseModelRule()
        with pytest.raises(ValueError, match="Cannot convert"):
            await rule.perform_fix(12345, MyModel)

    @pytest.mark.asyncio
    async def test_perform_fix_t_not_basemodel_subclass_raises(self):
        rule = BaseModelRule()
        with pytest.raises(ValueError, match="BaseModel subclass"):
            await rule.perform_fix({"x": 1}, int)

    @pytest.mark.asyncio
    async def test_perform_fix_coerces_string_value(self):
        rule = BaseModelRule()
        result = await rule.perform_fix({"name": "test", "value": "99"}, MyModel)
        assert isinstance(result, MyModel)
        assert result.value == 99


# ---------------------------------------------------------------------------
# invoke()
# ---------------------------------------------------------------------------


class TestBaseModelRuleInvoke:
    @pytest.mark.asyncio
    async def test_invoke_valid_instance_no_fix(self):
        rule = BaseModelRule()
        instance = MyModel(name="a", value=1)
        result = await rule.invoke("m", instance, MyModel, auto_fix=False)
        assert result is instance

    @pytest.mark.asyncio
    async def test_invoke_dict_validate_passes_returns_dict(self):
        # validate() succeeds for a valid dict (calls model_validate internally but
        # does not return the instance). invoke() returns the original value when
        # validate passes — not the model instance.
        rule = BaseModelRule()
        data = {"name": "b", "value": 2}
        result = await rule.invoke("m", data, MyModel, auto_fix=True)
        assert result == data

    @pytest.mark.asyncio
    async def test_invoke_invalid_no_auto_fix_raises_validation_error(self):
        rule = BaseModelRule()
        with pytest.raises(ValidationError):
            await rule.invoke("m", "bad input", MyModel, auto_fix=False)

    @pytest.mark.asyncio
    async def test_invoke_invalid_auto_fix_still_fails_raises_validation_error(self):
        rule = BaseModelRule()
        with pytest.raises(ValidationError):
            await rule.invoke("m", "bad input", MyModel, auto_fix=True)
