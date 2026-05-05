# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for DataClassSpecAdapter (lionagi/ln/types/adapters/_dataclass.py)."""

import pytest

from lionagi.ln.types import Operable, Spec
from lionagi.ln.types.adapters._dataclass import DataClassSpecAdapter
from lionagi.ln.types.base import DataClass, Params

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_op(*specs):
    """Build an Operable from specs. Passes include=None, exclude=None to
    avoid the Unset vs None issue in get_specs when called from compose_structure."""
    return Operable(list(specs))


def compose(op, name, **kw):
    """Wrap compose_structure with explicit include=None, exclude=None."""
    kw.setdefault("include", None)
    kw.setdefault("exclude", None)
    return DataClassSpecAdapter.compose_structure(op, name, **kw)


# ---------------------------------------------------------------------------
# create_field tests
# ---------------------------------------------------------------------------


class TestCreateField:
    def test_no_default_returns_empty_dict(self):
        spec = Spec(str, name="x")
        result = DataClassSpecAdapter.create_field(spec)
        assert result == {}

    def test_with_scalar_default(self):
        spec = Spec(int, name="age").with_default(42)
        result = DataClassSpecAdapter.create_field(spec)
        assert result == {"default": 42}

    def test_with_zero_default(self):
        spec = Spec(int, name="count").with_default(0)
        result = DataClassSpecAdapter.create_field(spec)
        assert result == {"default": 0}

    def test_with_string_default(self):
        spec = Spec(str, name="label").with_default("hello")
        result = DataClassSpecAdapter.create_field(spec)
        assert result == {"default": "hello"}

    def test_with_default_factory(self):
        spec = Spec(list, name="items").with_default(list)
        result = DataClassSpecAdapter.create_field(spec)
        assert result == {"default_factory": list}
        assert callable(result["default_factory"])

    def test_with_dict_factory(self):
        spec = Spec(dict, name="meta").with_default(dict)
        result = DataClassSpecAdapter.create_field(spec)
        assert result == {"default_factory": dict}

    def test_nullable_returns_default_none(self):
        spec = Spec(str, name="note").as_nullable()
        result = DataClassSpecAdapter.create_field(spec)
        assert result == {"default": None}

    def test_non_nullable_no_default_empty(self):
        spec = Spec(str, name="required_field")
        result = DataClassSpecAdapter.create_field(spec)
        assert "default" not in result
        assert "default_factory" not in result

    def test_factory_takes_priority_over_none(self):
        # A spec with a factory should have default_factory, not default
        spec = Spec(list, name="xs").with_default(list)
        result = DataClassSpecAdapter.create_field(spec)
        assert "default_factory" in result
        assert "default" not in result


# ---------------------------------------------------------------------------
# create_field_validator tests
# ---------------------------------------------------------------------------


class TestCreateFieldValidator:
    def test_no_validator_returns_none(self):
        spec = Spec(str, name="x")
        result = DataClassSpecAdapter.create_field_validator(spec)
        assert result is None

    def test_single_callable_wrapped_in_list(self):
        def my_v(v):
            return v

        spec = Spec(str, name="code").with_validator(my_v)
        result = DataClassSpecAdapter.create_field_validator(spec)
        assert result is not None
        assert "code" in result
        assert result["code"] == [my_v]

    def test_list_of_validators_preserved(self):
        def v1(v):
            return v

        def v2(v):
            return v

        spec = Spec(str, name="val").with_validator([v1, v2])
        result = DataClassSpecAdapter.create_field_validator(spec)
        assert result is not None
        assert result["val"] == [v1, v2]

    def test_validator_key_uses_spec_name(self):
        def check(v):
            return v

        spec = Spec(int, name="score").with_validator(check)
        result = DataClassSpecAdapter.create_field_validator(spec)
        assert "score" in result


# ---------------------------------------------------------------------------
# compose_structure tests
# ---------------------------------------------------------------------------


class TestComposeStructure:
    def test_frozen_creates_params_subclass(self):
        op = make_op(Spec(str, name="x"))
        cls = compose(op, "MyParams", frozen=True)
        assert issubclass(cls, Params)
        assert cls.__name__ == "MyParams"

    def test_mutable_creates_dataclass_subclass(self):
        op = make_op(Spec(str, name="x"))
        cls = compose(op, "MyDataClass", frozen=False)
        assert issubclass(cls, DataClass)
        assert cls.__name__ == "MyDataClass"

    def test_frozen_class_is_not_dataclass_subclass_only(self):
        op = make_op(Spec(str, name="x"))
        cls = compose(op, "FrozenCls", frozen=True)
        # Params IS the frozen variant; it should be a subclass of Params
        assert issubclass(cls, Params)

    def test_compose_with_required_and_optional_fields(self):
        op = make_op(
            Spec(str, name="name"),
            Spec(int, name="age").with_default(0),
        )
        cls = compose(op, "Person")
        inst = cls(name="Alice", age=30)
        assert inst.name == "Alice"
        assert inst.age == 30

    def test_compose_optional_field_uses_default(self):
        op = make_op(
            Spec(str, name="name"),
            Spec(int, name="age").with_default(0),
        )
        cls = compose(op, "Person")
        inst = cls(name="Bob")
        assert inst.age == 0

    def test_compose_with_factory_default(self):
        op = make_op(
            Spec(str, name="key"),
            Spec(list, name="tags").with_default(list),
        )
        cls = compose(op, "Tagged")
        inst1 = cls(key="a")
        inst2 = cls(key="b")
        # Each instance gets its own list
        assert inst1.tags is not inst2.tags
        assert inst1.tags == []

    def test_compose_nullable_field_defaults_none(self):
        op = make_op(Spec(str, name="note").as_nullable())
        cls = compose(op, "WithNote")
        inst = cls()
        assert inst.note is None

    def test_compose_sets_docstring(self):
        op = make_op(Spec(str, name="x"))
        cls = compose(op, "Documented", doc="My custom doc")
        assert cls.__doc__ == "My custom doc"

    def test_compose_no_doc_does_not_set_custom_doc(self):
        op = make_op(Spec(str, name="x"))
        cls = compose(op, "NoDocs")
        # The __doc__ should not be our custom string (may be None or auto-generated)
        assert cls.__doc__ != "My custom doc"

    def test_mutable_instance_can_mutate(self):
        op = make_op(Spec(str, name="label"))
        cls = compose(op, "MutableCls", frozen=False)
        inst = cls(label="old")
        inst.label = "new"
        assert inst.label == "new"

    def test_frozen_instance_is_immutable(self):
        op = make_op(Spec(str, name="label"))
        cls = compose(op, "FrozenCls", frozen=True)
        inst = cls(label="fixed")
        with pytest.raises((AttributeError, TypeError)):
            inst.label = "changed"

    def test_compose_multiple_fields(self):
        op = make_op(
            Spec(str, name="a"),
            Spec(int, name="b").with_default(1),
            Spec(float, name="c").with_default(3.14),
        )
        cls = compose(op, "Multi")
        inst = cls(a="hello")
        assert inst.a == "hello"
        assert inst.b == 1
        assert inst.c == pytest.approx(3.14)

    def test_compose_with_validator_wires_field_validators(self):
        calls = []

        def track(v):
            calls.append(v)
            return v

        op = make_op(Spec(str, name="x").with_validator(track))
        cls = compose(op, "Validated", frozen=False)
        # Validators run during _validate which may be called at init
        # Just verify the class has __field_validators__
        assert hasattr(cls, "__field_validators__")
        assert "x" in cls.__field_validators__


# ---------------------------------------------------------------------------
# validate_instance tests
# ---------------------------------------------------------------------------


class TestValidateInstance:
    def test_returns_instance(self):
        op = make_op(Spec(str, name="x"), Spec(int, name="y").with_default(0))
        cls = compose(op, "VI")
        inst = DataClassSpecAdapter.validate_instance(cls, {"x": "hello", "y": 5})
        assert isinstance(inst, cls)

    def test_instance_has_correct_field_values(self):
        op = make_op(Spec(str, name="x"), Spec(int, name="y").with_default(0))
        cls = compose(op, "VICheck")
        inst = DataClassSpecAdapter.validate_instance(cls, {"x": "world", "y": 99})
        assert inst.x == "world"
        assert inst.y == 99

    def test_validate_with_optional_omitted(self):
        op = make_op(Spec(str, name="name"), Spec(int, name="count").with_default(0))
        cls = compose(op, "OptVI")
        inst = DataClassSpecAdapter.validate_instance(cls, {"name": "test"})
        assert inst.name == "test"
        assert inst.count == 0


# ---------------------------------------------------------------------------
# extract_specs tests
# ---------------------------------------------------------------------------


class TestExtractSpecs:
    def test_returns_tuple(self):
        op = make_op(Spec(str, name="a"), Spec(int, name="b").with_default(1))
        cls = compose(op, "ESTest")
        specs = DataClassSpecAdapter.extract_specs(cls)
        assert isinstance(specs, tuple)

    def test_extracts_correct_number_of_specs(self):
        op = make_op(Spec(str, name="a"), Spec(int, name="b").with_default(1))
        cls = compose(op, "ESCount")
        specs = DataClassSpecAdapter.extract_specs(cls)
        assert len(specs) == 2

    def test_extracted_spec_names_match(self):
        op = make_op(Spec(str, name="first"), Spec(int, name="second").with_default(0))
        cls = compose(op, "ESNames")
        specs = DataClassSpecAdapter.extract_specs(cls)
        names = {s.name for s in specs}
        assert "first" in names
        assert "second" in names

    def test_extracted_spec_preserves_default(self):
        op = make_op(Spec(int, name="val").with_default(42))
        cls = compose(op, "ESDefault")
        specs = DataClassSpecAdapter.extract_specs(cls)
        spec_by_name = {s.name: s for s in specs}
        val_spec = spec_by_name["val"]
        # The default should be preserved
        assert val_spec.get("default") == 42

    def test_raises_typeerror_for_plain_class(self):
        with pytest.raises(TypeError, match="DataClass or Params subclass"):
            DataClassSpecAdapter.extract_specs(str)

    def test_raises_typeerror_for_non_type(self):
        with pytest.raises(TypeError):
            DataClassSpecAdapter.extract_specs(42)

    def test_raises_typeerror_for_dict(self):
        with pytest.raises(TypeError):
            DataClassSpecAdapter.extract_specs(dict)

    def test_extracts_from_mutable_dataclass(self):
        op = make_op(Spec(str, name="x"), Spec(float, name="y").with_default(0.0))
        cls = compose(op, "ESMutable", frozen=False)
        specs = DataClassSpecAdapter.extract_specs(cls)
        assert len(specs) == 2
        names = {s.name for s in specs}
        assert {"x", "y"} == names


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_compose_instantiate_to_dict(self):
        op = make_op(
            Spec(str, name="name"),
            Spec(int, name="score").with_default(0),
        )
        cls = compose(op, "RTFrozen")
        inst = DataClassSpecAdapter.validate_instance(
            cls, {"name": "Alice", "score": 100}
        )
        d = inst.to_dict()
        assert d["name"] == "Alice"
        assert d["score"] == 100

    def test_mutable_round_trip(self):
        op = make_op(
            Spec(str, name="key"),
            Spec(list, name="vals").with_default(list),
        )
        cls = compose(op, "RTMutable", frozen=False)
        inst = cls(key="k", vals=[1, 2, 3])
        d = inst.to_dict()
        assert d["key"] == "k"
        assert d["vals"] == [1, 2, 3]

    def test_extract_specs_from_composed_class(self):
        op = make_op(
            Spec(str, name="label"),
            Spec(int, name="rank").with_default(1),
        )
        cls = compose(op, "RTExtract")
        specs = DataClassSpecAdapter.extract_specs(cls)
        names = {s.name for s in specs}
        assert "label" in names
        assert "rank" in names

    def test_nullable_round_trip(self):
        op = make_op(Spec(str, name="opt").as_nullable())
        cls = compose(op, "RTNullable")
        inst = cls()
        assert inst.opt is None
        d = inst.to_dict()
        assert d.get("opt") is None
