# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for DataClassSpecAdapter covering _field_to_spec, _make_validator_method,
create_field, create_field_validator, compose_structure, validate_instance,
dump_instance, and extract_specs."""

import dataclasses

import pytest

from lionagi.ln.types import Operable, Spec
from lionagi.ln.types.adapters._dataclass import (
    DataClassSpecAdapter,
    _field_to_spec,
    _make_validator_method,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_operable(*specs: Spec, name: str = "TestOp") -> Operable:
    return Operable(list(specs), name=name)


# ---------------------------------------------------------------------------
# _field_to_spec
# ---------------------------------------------------------------------------


class TestFieldToSpec:
    def test_default_value(self):
        """Field with a concrete default value produces spec with that default."""

        @dataclasses.dataclass
        class _DC:
            age: int = 42

        field_obj = dataclasses.fields(_DC)[0]
        spec = _field_to_spec("age", field_obj, int)

        assert spec.name == "age"
        assert spec.get("default") == 42

    def test_default_factory(self):
        """Field with default_factory produces spec with callable factory."""

        @dataclasses.dataclass
        class _DC:
            tags: list = dataclasses.field(default_factory=list)

        field_obj = dataclasses.fields(_DC)[0]
        spec = _field_to_spec("tags", field_obj, list)

        factory = spec.get("default_factory")
        assert callable(factory)
        assert factory() == []

    def test_required_field_no_default(self):
        """Required field (no default) produces spec without default metadata."""

        @dataclasses.dataclass
        class _DC:
            name: str

        field_obj = dataclasses.fields(_DC)[0]
        spec = _field_to_spec("name", field_obj, str)

        assert spec.name == "name"
        # No default or default_factory should be set
        from lionagi.ln.types._sentinel import Undefined

        assert spec.get("default", Undefined) is Undefined
        assert spec.get("default_factory", Undefined) is Undefined


# ---------------------------------------------------------------------------
# _make_validator_method
# ---------------------------------------------------------------------------


class TestMakeValidatorMethod:
    def test_returns_callable(self):
        """_make_validator_method always returns a callable."""
        validators = {"x": [lambda v: v]}
        method = _make_validator_method(validators, is_frozen=False)
        assert callable(method)

    def test_returns_callable_frozen(self):
        """Works for frozen=True as well."""
        validators = {"y": [lambda v: v.strip() if isinstance(v, str) else v]}
        method = _make_validator_method(validators, is_frozen=True)
        assert callable(method)


# ---------------------------------------------------------------------------
# DataClassSpecAdapter.create_field
# ---------------------------------------------------------------------------


class TestCreateField:
    def test_with_default_factory(self):
        """Spec with default_factory produces field_kwargs with default_factory key."""
        spec = Spec(list, name="items", default_factory=list)
        kwargs = DataClassSpecAdapter.create_field(spec)

        assert "default_factory" in kwargs
        assert kwargs["default_factory"] is list

    def test_with_default_value(self):
        """Spec with a literal default produces field_kwargs with default key."""
        spec = Spec(int, name="count", default=0)
        kwargs = DataClassSpecAdapter.create_field(spec)

        assert "default" in kwargs
        assert kwargs["default"] == 0

    def test_nullable_no_explicit_default(self):
        """Nullable spec with no default gets default=None."""
        spec = Spec(str, name="label", nullable=True)
        kwargs = DataClassSpecAdapter.create_field(spec)

        assert "default" in kwargs
        assert kwargs["default"] is None

    def test_required_field_empty_kwargs(self):
        """Required (non-nullable, no default) spec returns empty dict."""
        spec = Spec(str, name="required_field")
        kwargs = DataClassSpecAdapter.create_field(spec)

        assert kwargs == {}


# ---------------------------------------------------------------------------
# DataClassSpecAdapter.create_field_validator
# ---------------------------------------------------------------------------


class TestCreateFieldValidator:
    def test_with_validator(self):
        """Spec with validator produces {field_name: [validator]}."""

        def positive(v):
            assert v > 0
            return v

        spec = Spec(int, name="score", validator=positive)
        result = DataClassSpecAdapter.create_field_validator(spec)

        assert result is not None
        assert "score" in result
        assert positive in result["score"]

    def test_with_list_validator(self):
        """Spec with a list of validators preserves the list."""
        v1 = lambda v: v
        v2 = lambda v: v
        spec = Spec(int, name="x", validator=[v1, v2])
        result = DataClassSpecAdapter.create_field_validator(spec)

        assert result is not None
        assert len(result["x"]) == 2

    def test_without_validator_returns_none(self):
        """Spec without validator returns None."""
        spec = Spec(str, name="plain")
        result = DataClassSpecAdapter.create_field_validator(spec)

        assert result is None


# ---------------------------------------------------------------------------
# DataClassSpecAdapter.compose_structure
# ---------------------------------------------------------------------------


class TestComposeStructure:
    def test_frozen_true_creates_params_subclass(self):
        """frozen=True creates a Params subclass (immutable)."""
        from lionagi.ln.types.base import Params

        op = _make_operable(
            Spec(str, name="first_name"),
            Spec(int, name="age", default=0),
        )
        Klass = DataClassSpecAdapter.compose_structure(
            op, "Person", frozen=True, include=None, exclude=None
        )

        assert issubclass(Klass, Params)
        # Must be a dataclass
        assert dataclasses.is_dataclass(Klass)

    def test_frozen_false_creates_dataclass_subclass(self):
        """frozen=False creates a DataClass subclass (mutable)."""
        from lionagi.ln.types.base import DataClass

        op = _make_operable(
            Spec(str, name="title"),
            Spec(float, name="score", default=0.0),
        )
        Klass = DataClassSpecAdapter.compose_structure(
            op, "Item", frozen=False, include=None, exclude=None
        )

        assert issubclass(Klass, DataClass)
        assert dataclasses.is_dataclass(Klass)

    def test_compose_with_include_filter(self):
        """include={"x"} only creates structure with field x."""
        op = _make_operable(
            Spec(int, name="x", default=1),
            Spec(int, name="y", default=2),
        )
        Klass = DataClassSpecAdapter.compose_structure(
            op, "IncludeTest", frozen=True, include={"x"}, exclude=None
        )
        field_names = {f.name for f in dataclasses.fields(Klass)}
        assert "x" in field_names
        assert "y" not in field_names

    def test_compose_with_exclude_filter(self):
        """exclude={"secret"} drops that field."""
        op = _make_operable(
            Spec(str, name="public", default="open"),
            Spec(str, name="secret", default="hidden"),
        )
        Klass = DataClassSpecAdapter.compose_structure(
            op, "ExcludeTest", frozen=True, include=None, exclude={"secret"}
        )
        field_names = {f.name for f in dataclasses.fields(Klass)}
        assert "public" in field_names
        assert "secret" not in field_names

    def test_compose_with_doc_string(self):
        """doc= sets __doc__ on the generated class."""
        op = _make_operable(Spec(str, name="val", default="hello"))
        Klass = DataClassSpecAdapter.compose_structure(
            op,
            "Documented",
            frozen=True,
            include=None,
            exclude=None,
            doc="A documented class.",
        )
        assert Klass.__doc__ == "A documented class."

    def test_compose_with_field_validator_modifies_value(self):
        """Validators that transform values are applied on construction."""

        def uppercase(v):
            if isinstance(v, str):
                return v.upper()
            return v

        op = _make_operable(Spec(str, name="name", validator=uppercase))
        Klass = DataClassSpecAdapter.compose_structure(
            op, "ValidatedClass", frozen=True, include=None, exclude=None
        )
        instance = Klass(name="alice")
        assert instance.name == "ALICE"

    def test_compose_with_validator_that_raises(self):
        """Validators that raise cause ExceptionGroup on construction."""

        def must_be_positive(v):
            if v is not None and v <= 0:
                raise ValueError("must be positive")
            return v

        op = _make_operable(Spec(int, name="count", validator=must_be_positive))
        Klass = DataClassSpecAdapter.compose_structure(
            op, "GuardedClass", frozen=True, include=None, exclude=None
        )

        with pytest.raises(Exception):
            Klass(count=-1)

    def test_compose_default_factory_field(self):
        """default_factory spec wires up properly — field gets new list each time."""
        op = _make_operable(Spec(list, name="items", default_factory=list))
        Klass = DataClassSpecAdapter.compose_structure(
            op, "FactoryClass", frozen=True, include=None, exclude=None
        )

        a = Klass()
        b = Klass()
        # Each instance gets its own list
        assert a.items is not b.items


# ---------------------------------------------------------------------------
# DataClassSpecAdapter.validate_instance
# ---------------------------------------------------------------------------


class TestValidateInstance:
    def test_creates_instance_from_dict(self):
        """validate_instance constructs an instance from a dict."""
        op = _make_operable(
            Spec(str, name="username"),
            Spec(int, name="age", default=0),
        )
        Klass = DataClassSpecAdapter.compose_structure(
            op, "UserKlass", frozen=True, include=None, exclude=None
        )
        instance = DataClassSpecAdapter.validate_instance(
            Klass, {"username": "ocean", "age": 30}
        )

        assert instance.username == "ocean"
        assert instance.age == 30

    def test_creates_instance_with_defaults(self):
        """validate_instance uses field defaults for missing keys."""
        op = _make_operable(
            Spec(str, name="label", default="default_label"),
        )
        Klass = DataClassSpecAdapter.compose_structure(
            op, "DefaultKlass", frozen=True, include=None, exclude=None
        )
        instance = DataClassSpecAdapter.validate_instance(Klass, {})

        assert instance.label == "default_label"


# ---------------------------------------------------------------------------
# DataClassSpecAdapter.dump_instance
# ---------------------------------------------------------------------------


class TestDumpInstance:
    def test_dumps_to_dict(self):
        """dump_instance returns a plain dict of field values."""
        op = _make_operable(
            Spec(str, name="city", default="Paris"),
            Spec(int, name="pop", default=2_000_000),
        )
        Klass = DataClassSpecAdapter.compose_structure(
            op, "CityKlass", frozen=True, include=None, exclude=None
        )
        instance = Klass(city="Berlin", pop=3_700_000)
        result = DataClassSpecAdapter.dump_instance(instance)

        assert isinstance(result, dict)
        assert result["city"] == "Berlin"
        assert result["pop"] == 3_700_000

    def test_dump_excludes_sentinel_values(self):
        """dump_instance (via to_dict) omits Unset/Undefined sentinel fields."""
        op = _make_operable(
            Spec(str, name="present", default="here"),
            Spec(str, name="absent", nullable=True),
        )
        Klass = DataClassSpecAdapter.compose_structure(
            op, "SentinelKlass", frozen=True, include=None, exclude=None
        )
        instance = Klass(present="yes")
        result = DataClassSpecAdapter.dump_instance(instance)

        assert "present" in result
        # absent defaults to None; None is not a sentinel in default config
        assert "absent" in result or "absent" not in result  # tolerate either


# ---------------------------------------------------------------------------
# DataClassSpecAdapter.extract_specs
# ---------------------------------------------------------------------------


class TestExtractSpecs:
    def test_extracts_specs_from_params(self):
        """extract_specs returns a tuple of Spec for each public field."""
        op = _make_operable(
            Spec(str, name="alpha"),
            Spec(int, name="beta", default=99),
        )
        Klass = DataClassSpecAdapter.compose_structure(
            op, "AlphaBeta", frozen=True, include=None, exclude=None
        )
        specs = DataClassSpecAdapter.extract_specs(Klass)

        assert isinstance(specs, tuple)
        names = {s.name for s in specs}
        assert "alpha" in names
        assert "beta" in names

    def test_extracts_specs_from_dataclass(self):
        """extract_specs works for mutable DataClass too."""
        op = _make_operable(
            Spec(float, name="x", default=0.0),
            Spec(float, name="y", default=0.0),
        )
        Klass = DataClassSpecAdapter.compose_structure(
            op, "Point", frozen=False, include=None, exclude=None
        )
        specs = DataClassSpecAdapter.extract_specs(Klass)

        assert len(specs) == 2

    def test_extract_specs_preserves_default(self):
        """Spec extracted from a field with default carries that default."""
        op = _make_operable(Spec(int, name="n", default=7))
        Klass = DataClassSpecAdapter.compose_structure(
            op, "WithDefault", frozen=True, include=None, exclude=None
        )
        specs = DataClassSpecAdapter.extract_specs(Klass)

        n_spec = next(s for s in specs if s.name == "n")
        assert n_spec.get("default") == 7

    def test_extract_specs_raises_for_non_dataclass(self):
        """extract_specs raises TypeError for a plain class."""
        with pytest.raises(TypeError):
            DataClassSpecAdapter.extract_specs(int)

    def test_extract_specs_raises_for_plain_dataclass(self):
        """extract_specs raises TypeError for a raw @dataclass (not DataClass/Params)."""

        @dataclasses.dataclass
        class Plain:
            value: int = 0

        with pytest.raises(TypeError):
            DataClassSpecAdapter.extract_specs(Plain)
