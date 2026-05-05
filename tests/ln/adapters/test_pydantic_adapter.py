# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for PydanticSpecAdapter: Spec <-> Pydantic FieldInfo/BaseModel."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from lionagi.ln.types import Operable, Spec
from lionagi.ln.types.adapters._pydantic import PydanticSpecAdapter

# ---------------------------------------------------------------------------
# create_field
# ---------------------------------------------------------------------------


class TestCreateField:
    def test_basic_string_field(self):
        spec = Spec(str, name="name")
        fi = PydanticSpecAdapter.create_field(spec)
        assert isinstance(fi, FieldInfo)
        assert fi.annotation == str

    def test_field_with_description(self):
        spec = Spec(str, name="name", description="A name field")
        fi = PydanticSpecAdapter.create_field(spec)
        assert fi.description == "A name field"

    def test_field_with_default_value(self):
        spec = Spec(int, name="count")
        spec_with_default = Spec(int, name="count", default=0)
        fi = PydanticSpecAdapter.create_field(spec_with_default)
        assert fi.default == 0

    def test_field_with_default_factory(self):
        def my_factory():
            return []

        spec = Spec(list, name="items", default_factory=my_factory)
        fi = PydanticSpecAdapter.create_field(spec)
        assert fi.default_factory is my_factory

    def test_field_with_alias(self):
        spec = Spec(str, name="my_field", alias="myField")
        fi = PydanticSpecAdapter.create_field(spec)
        assert fi.alias == "myField"

    def test_field_with_min_length(self):
        spec = Spec(str, name="name", min_length=1)
        fi = PydanticSpecAdapter.create_field(spec)
        # min_length is in metadata constraints
        meta_types = [type(m).__name__ for m in fi.metadata]
        assert "MinLen" in meta_types

    def test_field_with_max_length(self):
        spec = Spec(str, name="name", max_length=100)
        fi = PydanticSpecAdapter.create_field(spec)
        meta_types = [type(m).__name__ for m in fi.metadata]
        assert "MaxLen" in meta_types

    def test_nullable_field_gets_none_default(self):
        # is_nullable is True only when nullable=True is set explicitly (not via str | None union)
        spec = Spec(str, name="opt_name", nullable=True)
        fi = PydanticSpecAdapter.create_field(spec)
        assert fi.default is None

    def test_nullable_required_field_no_default(self):
        from pydantic_core import PydanticUndefined

        # nullable=True + required=True: the required flag should prevent default=None being set
        spec = Spec(str, name="req_nullable", nullable=True, required=True)
        fi = PydanticSpecAdapter.create_field(spec)
        assert fi.default is PydanticUndefined

    def test_custom_key_goes_to_json_schema_extra(self):
        spec = Spec(str, name="field", custom_key="custom_val")
        fi = PydanticSpecAdapter.create_field(spec)
        assert fi.json_schema_extra is not None
        assert fi.json_schema_extra["custom_key"] == "custom_val"

    def test_field_with_examples(self):
        spec = Spec(str, name="name", examples=["Alice", "Bob"])
        fi = PydanticSpecAdapter.create_field(spec)
        assert fi.examples == ["Alice", "Bob"]


# ---------------------------------------------------------------------------
# create_field_validator
# ---------------------------------------------------------------------------


class TestCreateFieldValidator:
    def test_no_validator_returns_none(self):
        spec = Spec(str, name="no_val")
        result = PydanticSpecAdapter.create_field_validator(spec)
        assert result is None

    def test_with_validator_returns_dict(self):
        def my_validator(cls, v):
            return v.strip()

        spec = Spec(str, name="trimmed", validator=my_validator)
        result = PydanticSpecAdapter.create_field_validator(spec)
        assert result is not None
        assert isinstance(result, dict)
        assert "_trimmed_validator" in result

    def test_validator_key_uses_spec_name(self):
        def my_validator(cls, v):
            return v

        spec = Spec(str, name="my_field", validator=my_validator)
        result = PydanticSpecAdapter.create_field_validator(spec)
        assert "_my_field_validator" in result


# ---------------------------------------------------------------------------
# compose_structure
# ---------------------------------------------------------------------------


class TestComposeStructure:
    def test_basic_model_creation(self):
        op = Operable([Spec(str, name="name"), Spec(int, name="age")])
        MyModel = PydanticSpecAdapter.compose_structure(op, "MyModel")
        assert issubclass(MyModel, BaseModel)
        assert "name" in MyModel.model_fields
        assert "age" in MyModel.model_fields

    def test_model_name_is_set(self):
        op = Operable([Spec(str, name="x")])
        MyModel = PydanticSpecAdapter.compose_structure(op, "SpecificName")
        assert MyModel.__name__ == "SpecificName"

    def test_model_can_be_instantiated(self):
        op = Operable([Spec(str, name="name"), Spec(int, name="age", default=0)])
        MyModel = PydanticSpecAdapter.compose_structure(op, "MyModel")
        instance = MyModel(name="Alice", age=30)
        assert instance.name == "Alice"
        assert instance.age == 30

    def test_model_with_doc(self):
        op = Operable([Spec(str, name="name")])
        MyModel = PydanticSpecAdapter.compose_structure(
            op, "DocModel", doc="My documentation"
        )
        assert MyModel.__doc__ == "My documentation"

    def test_model_with_base_type(self):
        class MyBase(BaseModel):
            extra_field: str = "base"

        op = Operable([Spec(str, name="name")])
        MyModel = PydanticSpecAdapter.compose_structure(op, "MyModel", base_type=MyBase)
        instance = MyModel(name="Alice")
        assert instance.extra_field == "base"
        assert instance.name == "Alice"

    def test_model_with_include(self):
        op = Operable([Spec(str, name="a"), Spec(int, name="b"), Spec(float, name="c")])
        MyModel = PydanticSpecAdapter.compose_structure(
            op, "MyModel", include={"a", "b"}
        )
        fields = set(MyModel.model_fields.keys())
        assert "a" in fields
        assert "b" in fields
        assert "c" not in fields

    def test_model_with_exclude(self):
        op = Operable([Spec(str, name="a"), Spec(int, name="b"), Spec(float, name="c")])
        MyModel = PydanticSpecAdapter.compose_structure(op, "MyModel", exclude={"c"})
        fields = set(MyModel.model_fields.keys())
        assert "a" in fields
        assert "b" in fields
        assert "c" not in fields

    def test_model_with_validator_applied(self):
        def upper_validator(cls, v):
            return v.upper()

        op = Operable([Spec(str, name="name", validator=upper_validator)])
        MyModel = PydanticSpecAdapter.compose_structure(op, "UpperModel")
        instance = MyModel(name="alice")
        assert instance.name == "ALICE"

    def test_empty_operable_creates_empty_model(self):
        op = Operable([])
        EmptyModel = PydanticSpecAdapter.compose_structure(op, "EmptyModel")
        assert issubclass(EmptyModel, BaseModel)
        assert len(EmptyModel.model_fields) == 0


# ---------------------------------------------------------------------------
# validate_instance
# ---------------------------------------------------------------------------


class TestValidateInstance:
    def test_validate_with_plain_dict(self):
        class Simple(BaseModel):
            x: int
            y: str

        result = PydanticSpecAdapter.validate_instance(Simple, {"x": 1, "y": "hello"})
        assert result.x == 1
        assert result.y == "hello"

    def test_validate_with_basemodel_value_uses_model_construct(self):
        class Inner(BaseModel):
            val: int = 1

        class Outer(BaseModel):
            inner: Inner
            name: str

        data = {"inner": Inner(val=5), "name": "test"}
        result = PydanticSpecAdapter.validate_instance(Outer, data)
        assert result.name == "test"
        assert result.inner.val == 5

    def test_validate_returns_model_instance(self):
        class M(BaseModel):
            x: int

        result = PydanticSpecAdapter.validate_instance(M, {"x": 42})
        assert isinstance(result, M)


# ---------------------------------------------------------------------------
# dump_instance
# ---------------------------------------------------------------------------


class TestDumpInstance:
    def test_dump_returns_dict(self):
        class M(BaseModel):
            name: str
            age: int

        instance = M(name="Alice", age=30)
        result = PydanticSpecAdapter.dump_instance(instance)
        assert isinstance(result, dict)
        assert result == {"name": "Alice", "age": 30}

    def test_dump_nested_model(self):
        class Inner(BaseModel):
            val: int

        class Outer(BaseModel):
            inner: Inner
            name: str

        instance = Outer(inner=Inner(val=5), name="test")
        result = PydanticSpecAdapter.dump_instance(instance)
        assert result["name"] == "test"
        assert result["inner"] == {"val": 5}


# ---------------------------------------------------------------------------
# extract_specs
# ---------------------------------------------------------------------------


class TestExtractSpecs:
    def test_raises_type_error_for_non_model(self):
        with pytest.raises(TypeError, match="BaseModel subclass"):
            PydanticSpecAdapter.extract_specs("not_a_model")

    def test_raises_type_error_for_instance(self):
        class M(BaseModel):
            x: int

        with pytest.raises(TypeError):
            PydanticSpecAdapter.extract_specs(M(x=1))

    def test_basic_model_extracts_specs(self):
        class M(BaseModel):
            name: str
            age: int

        specs = PydanticSpecAdapter.extract_specs(M)
        assert len(specs) == 2
        names = {s.name for s in specs}
        assert "name" in names
        assert "age" in names

    def test_spec_annotations_correct(self):
        class M(BaseModel):
            x: str
            y: int

        specs = PydanticSpecAdapter.extract_specs(M)
        spec_map = {s.name: s for s in specs}
        assert spec_map["x"].annotation == str
        assert spec_map["y"].annotation == int

    def test_model_with_default_extracts_default(self):
        class M(BaseModel):
            x: int = 5

        specs = PydanticSpecAdapter.extract_specs(M)
        spec_map = {s.name: s for s in specs}
        assert spec_map["x"].get("default") == 5

    def test_returns_tuple(self):
        class M(BaseModel):
            x: str

        specs = PydanticSpecAdapter.extract_specs(M)
        assert isinstance(specs, tuple)

    def test_empty_model_returns_empty_tuple(self):
        class Empty(BaseModel):
            pass

        specs = PydanticSpecAdapter.extract_specs(Empty)
        assert specs == ()

    def test_round_trip_compose_then_extract(self):
        op = Operable([Spec(str, name="name"), Spec(int, name="age", default=0)])
        MyModel = PydanticSpecAdapter.compose_structure(op, "MyModel")
        extracted = PydanticSpecAdapter.extract_specs(MyModel)
        names = {s.name for s in extracted}
        assert "name" in names
        assert "age" in names
