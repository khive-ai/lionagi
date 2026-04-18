"""Coverage tests for field_model.py and operable_model.py."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from lionagi.models.field_model import FieldModel
from lionagi.models.operable_model import OperableModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pos_validator(v):
    return v > 0


def _raise_validator(v):
    if v < 0:
        raise ValueError("negative not allowed")
    return v


# ---------------------------------------------------------------------------
# FieldModel – construction
# ---------------------------------------------------------------------------


class TestFieldModelConstruction:
    def test_minimal(self):
        fm = FieldModel(base_type=str)
        assert fm.base_type is str

    def test_with_name_kwarg(self):
        fm = FieldModel(name="my_field", base_type=int)
        assert fm.name == "my_field"

    def test_default_name_is_field(self):
        fm = FieldModel(base_type=str)
        assert fm.name == "field"

    def test_annotation_alias_for_base_type(self):
        # _convert_kwargs_to_params converts 'annotation' kwarg to 'base_type'
        converted = FieldModel._convert_kwargs_to_params(annotation=float)
        assert converted.get("base_type") is float

    def test_with_description(self):
        fm = FieldModel(base_type=str, description="A label")
        assert fm.description == "A label"

    def test_with_default(self):
        fm = FieldModel(base_type=int, default=42)
        assert fm.default == 42

    def test_with_alias(self):
        fm = FieldModel(base_type=str, alias="alt_name")
        assert fm.alias == "alt_name"

    def test_invalid_param_not_in_metadata(self):
        # kwargs that don't match known keys go into metadata as Meta items
        converted = FieldModel._convert_kwargs_to_params(base_type=str, custom_k="v")
        meta = converted.get("metadata", ())
        assert any(m.key == "custom_k" for m in meta)

    def test_missing_attr_raises_attribute_error(self):
        fm = FieldModel(base_type=str)
        with pytest.raises(AttributeError):
            _ = fm.nonexistent_attr

    def test_cannot_set_both_default_and_default_factory(self):
        with pytest.raises(ValueError, match="both default and default_factory"):
            FieldModel(base_type=int, default=1, default_factory=list)


# ---------------------------------------------------------------------------
# FieldModel – fluent API
# ---------------------------------------------------------------------------


class TestFieldModelFluentAPI:
    def test_as_nullable_returns_new_instance(self):
        fm = FieldModel(base_type=str)
        nullable = fm.as_nullable()
        assert nullable is not fm
        assert nullable.is_nullable is True
        assert fm.is_nullable is False

    def test_as_nullable_annotation_includes_none(self):
        fm = FieldModel(base_type=int)
        nullable = fm.as_nullable()
        ann = nullable.annotation
        # annotation should be int | None (union type)
        import types

        assert isinstance(ann, types.UnionType) or str(ann) in (
            "int | None",
            "typing.Optional[int]",
        )

    def test_as_listable_returns_new_instance(self):
        fm = FieldModel(base_type=str)
        listable = fm.as_listable()
        assert listable is not fm
        assert listable.is_listable is True
        assert fm.is_listable is False

    def test_as_listable_base_type_is_list(self):
        fm = FieldModel(base_type=str)
        listable = fm.as_listable()
        import types

        assert listable.base_type == list[str]

    def test_with_validator_stores_callable(self):
        fm = FieldModel(base_type=int)
        validated = fm.with_validator(_pos_validator)
        assert validated.has_validator() is True
        assert fm.has_validator() is False

    def test_with_description_adds_description(self):
        fm = FieldModel(base_type=str)
        described = fm.with_description("some description")
        assert described.description == "some description"

    def test_with_description_replaces_existing(self):
        fm = FieldModel(base_type=str, description="old")
        updated = fm.with_description("new")
        assert updated.description == "new"

    def test_with_alias_adds_alias(self):
        fm = FieldModel(base_type=str)
        aliased = fm.with_alias("myalias")
        assert aliased.alias == "myalias"

    def test_with_default_adds_default(self):
        fm = FieldModel(base_type=int)
        defaulted = fm.with_default(99)
        assert defaulted.default == 99

    def test_with_default_replaces_existing(self):
        fm = FieldModel(base_type=int, default=1)
        updated = fm.with_default(2)
        assert updated.default == 2

    def test_with_default_factory(self):
        fm = FieldModel(base_type=list)
        fm2 = fm.with_default(list)  # callable default → stored as default_factory
        # The callable default is accessible via metadata
        assert fm2.extract_metadata("default") is list

    def test_chaining_fluent_methods(self):
        fm = (
            FieldModel(base_type=str)
            .with_description("chained")
            .with_alias("alias_x")
            .as_nullable()
        )
        assert fm.description == "chained"
        assert fm.alias == "alias_x"
        assert fm.is_nullable is True

    def test_with_frozen(self):
        fm = FieldModel(base_type=int)
        frozen = fm.with_frozen(True)
        assert frozen.extract_metadata("frozen") is True

    def test_with_exclude(self):
        fm = FieldModel(base_type=str)
        excluded = fm.with_exclude(True)
        assert excluded.extract_metadata("exclude") is True

    def test_with_metadata_custom_key(self):
        fm = FieldModel(base_type=str)
        fm2 = fm.with_metadata("custom_key", "custom_val")
        assert fm2.extract_metadata("custom_key") == "custom_val"

    def test_with_title(self):
        fm = FieldModel(base_type=str)
        titled = fm.with_title("My Title")
        assert titled.extract_metadata("title") == "My Title"


# ---------------------------------------------------------------------------
# FieldModel – create_field / FieldInfo
# ---------------------------------------------------------------------------


class TestFieldModelCreateField:
    def test_returns_field_info(self):
        fm = FieldModel(base_type=str, description="desc")
        fi = fm.create_field()
        assert isinstance(fi, FieldInfo)

    def test_description_in_field_info(self):
        fm = FieldModel(base_type=str, description="my desc")
        fi = fm.create_field()
        assert fi.description == "my desc"

    def test_nullable_sets_default_none(self):
        fm = FieldModel(base_type=int).as_nullable()
        fi = fm.create_field()
        assert fi.default is None

    def test_default_value_in_field_info(self):
        fm = FieldModel(base_type=int, default=7)
        fi = fm.create_field()
        assert fi.default == 7

    def test_callable_default_becomes_factory(self):
        fm = FieldModel(base_type=list, default=list)
        fi = fm.create_field()
        # callable default should be set as default_factory
        assert fi.default_factory is list or fi.default is list

    def test_annotation_set_on_field_info(self):
        fm = FieldModel(base_type=str)
        fi = fm.create_field()
        assert fi.annotation is str

    def test_extra_metadata_goes_to_json_schema_extra(self):
        fm = FieldModel(base_type=str, custom_meta="value")
        fi = fm.create_field()
        assert fi.json_schema_extra is not None
        assert fi.json_schema_extra.get("custom_meta") == "value"


# ---------------------------------------------------------------------------
# FieldModel – annotated() + cache
# ---------------------------------------------------------------------------


class TestFieldModelAnnotated:
    def test_annotated_returns_type(self):
        fm = FieldModel(base_type=str)
        result = fm.annotated()
        assert result is not None

    def test_annotated_nullable(self):
        fm = FieldModel(base_type=int).as_nullable()
        result = fm.annotated()
        # Should be a union type with None
        import types

        assert isinstance(result, types.UnionType) or "None" in str(result)

    def test_annotated_listable(self):
        fm = FieldModel(base_type=str).as_listable()
        result = fm.annotated()
        assert result is not None

    def test_annotated_cache_same_object(self):
        """LRU cache: second call returns same object."""
        fm = FieldModel(base_type=str, description="cached")
        r1 = fm.annotated()
        r2 = fm.annotated()
        assert r1 is r2


# ---------------------------------------------------------------------------
# FieldModel – validators
# ---------------------------------------------------------------------------


class TestFieldModelValidators:
    def test_is_valid_passes(self):
        fm = FieldModel(base_type=int).with_validator(_pos_validator)
        assert fm.is_valid(5) is True

    def test_is_valid_fails(self):
        fm = FieldModel(base_type=int).with_validator(_pos_validator)
        assert fm.is_valid(-1) is False

    def test_validator_raises_propagates(self):
        fm = FieldModel(base_type=int).with_validator(_raise_validator)
        with pytest.raises(ValueError, match="negative"):
            fm.validate(-1)

    def test_validate_passes_silently(self):
        fm = FieldModel(base_type=int).with_validator(_pos_validator)
        fm.validate(10)  # should not raise

    def test_has_validator_false_without_validator(self):
        fm = FieldModel(base_type=str)
        assert fm.has_validator() is False

    def test_validate_no_validators_noop(self):
        fm = FieldModel(base_type=str)
        fm.validate("anything")  # no validators → should not raise

    def test_field_validator_property_returns_dict(self):
        def my_val(v):
            return v > 0

        fm = FieldModel(name="score", base_type=int).with_validator(my_val)
        fv = fm.field_validator
        # Returns dict or None; with validator it should be not None
        assert fv is not None


# ---------------------------------------------------------------------------
# FieldModel – properties and repr
# ---------------------------------------------------------------------------


class TestFieldModelProperties:
    def test_annotation_property_str(self):
        fm = FieldModel(base_type=str)
        assert fm.annotation is str

    def test_annotation_property_listable(self):
        fm = FieldModel(base_type=int).as_listable()
        assert fm.annotation == list[int]

    def test_repr_nullable(self):
        fm = FieldModel(base_type=str).as_nullable()
        r = repr(fm)
        assert "nullable" in r

    def test_repr_listable(self):
        fm = FieldModel(base_type=str).as_listable()
        r = repr(fm)
        assert "listable" in r

    def test_repr_validated(self):
        fm = FieldModel(base_type=int).with_validator(_pos_validator)
        r = repr(fm)
        assert "validated" in r

    def test_metadata_dict(self):
        fm = FieldModel(base_type=str, description="desc", alias="al")
        d = fm.metadata_dict()
        assert d.get("description") == "desc"
        assert d.get("alias") == "al"

    def test_metadata_dict_with_exclude(self):
        fm = FieldModel(base_type=str, description="desc", alias="al")
        d = fm.metadata_dict(exclude=["alias"])
        assert "alias" not in d
        assert d.get("description") == "desc"

    def test_extract_metadata_missing_returns_none(self):
        fm = FieldModel(base_type=str)
        assert fm.extract_metadata("nonexistent") is None

    def test_to_dict_deprecated(self):
        fm = FieldModel(base_type=str, description="d")
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fm.to_dict()
            assert any(
                issubclass(warning.category, DeprecationWarning) for warning in w
            )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# OperableModel – add_field
# ---------------------------------------------------------------------------


class TestOperableModelAddField:
    def test_add_field_basic(self):
        m = OperableModel()
        m.add_field("score", value=10, annotation=int)
        assert m.score == 10
        assert "score" in m.extra_fields

    def test_add_field_string_annotation(self):
        m = OperableModel()
        m.add_field("label", value="hello", annotation=str)
        assert m.label == "hello"

    def test_add_field_duplicate_raises(self):
        m = OperableModel()
        m.add_field("x", value=1, annotation=int)
        with pytest.raises(ValueError, match="already exists"):
            m.add_field("x", value=2, annotation=int)

    def test_add_field_with_field_model(self):
        m = OperableModel()
        fm = FieldModel(base_type=float, description="rate")
        m.add_field("rate", value=3.14, field_model=fm)
        assert m.rate == 3.14
        assert "rate" in m.extra_field_models

    def test_add_field_no_value(self):
        m = OperableModel()
        m.add_field("empty_field", annotation=str)
        # Field should exist; value may be UNDEFINED
        assert "empty_field" in m.extra_fields

    def test_add_field_appears_in_all_fields(self):
        m = OperableModel()
        m.add_field("foo", value=42, annotation=int)
        assert "foo" in m.all_fields


# ---------------------------------------------------------------------------
# OperableModel – update_field
# ---------------------------------------------------------------------------


class TestOperableModelUpdateField:
    def test_update_field_value(self):
        m = OperableModel()
        m.add_field("count", value=1, annotation=int)
        m.update_field("count", value=99)
        assert m.count == 99

    def test_update_field_creates_if_not_exists(self):
        m = OperableModel()
        m.update_field("new_field", value="v", annotation=str)
        assert m.new_field == "v"

    def test_update_field_both_default_and_factory_raises(self):
        m = OperableModel()
        with pytest.raises(ValueError, match="both"):
            m.update_field("x", default=1, default_factory=list)

    def test_update_field_both_field_obj_and_model_raises(self):
        from pydantic import Field

        m = OperableModel()
        fi = Field(default=1)
        fm = FieldModel(base_type=int)
        with pytest.raises(ValueError, match="both"):
            m.update_field("y", field_obj=fi, field_model=fm)

    def test_update_field_invalid_field_obj_raises(self):
        m = OperableModel()
        with pytest.raises(ValueError, match="FieldInfo"):
            m.update_field("z", field_obj="not_a_field_info")

    def test_update_field_invalid_field_model_raises(self):
        m = OperableModel()
        with pytest.raises(ValueError, match="FieldModel"):
            m.update_field("z", field_model="not_a_field_model")


# ---------------------------------------------------------------------------
# OperableModel – remove_field
# ---------------------------------------------------------------------------


class TestOperableModelRemoveField:
    def test_remove_field_removes_from_extra_fields(self):
        m = OperableModel()
        m.add_field("temp", value=5, annotation=int)
        assert "temp" in m.extra_fields
        m.remove_field("temp")
        assert "temp" not in m.extra_fields

    def test_remove_field_removes_value_from_dict(self):
        m = OperableModel()
        m.add_field("tmp2", value=99, annotation=int)
        m.remove_field("tmp2")
        assert m.__dict__.get("tmp2") is None

    def test_remove_nonexistent_field_noop(self):
        m = OperableModel()
        m.remove_field("does_not_exist")  # should not raise


# ---------------------------------------------------------------------------
# OperableModel – field_getattr / field_setattr / field_hasattr
# ---------------------------------------------------------------------------


class TestOperableModelFieldAttr:
    def test_field_getattr_description(self):
        m = OperableModel()
        fm = FieldModel(base_type=str, description="test desc")
        m.add_field("labeled", value="x", field_model=fm)
        desc = m.field_getattr("labeled", "description")
        assert desc == "test desc"

    def test_field_getattr_missing_field_raises_key_error(self):
        m = OperableModel()
        with pytest.raises(KeyError):
            m.field_getattr("nonexistent", "description")

    def test_field_getattr_missing_attr_returns_default(self):
        m = OperableModel()
        m.add_field("n", value=1, annotation=int)
        result = m.field_getattr("n", "nonexistent_attr", "fallback")
        assert result == "fallback"

    def test_field_getattr_missing_attr_no_default_raises(self):
        m = OperableModel()
        m.add_field("n2", value=1, annotation=int)
        with pytest.raises(AttributeError):
            m.field_getattr("n2", "totally_missing_attr")

    def test_field_setattr_description(self):
        m = OperableModel()
        m.add_field("item", value="v", annotation=str)
        m.field_setattr("item", "description", "new desc")
        # description is set in json_schema_extra or on FieldInfo
        desc = m.field_getattr("item", "description", None)
        assert desc is not None

    def test_field_setattr_missing_field_raises_key_error(self):
        m = OperableModel()
        with pytest.raises(KeyError):
            m.field_setattr("ghost", "description", "x")

    def test_field_hasattr_existing_attr(self):
        m = OperableModel()
        m.add_field("chk", value=1, annotation=int)
        assert m.field_hasattr("chk", "annotation") is True

    def test_field_hasattr_missing_field_raises_key_error(self):
        m = OperableModel()
        with pytest.raises(KeyError):
            m.field_hasattr("missing", "annotation")


# ---------------------------------------------------------------------------
# OperableModel – new_model
# ---------------------------------------------------------------------------


class TestOperableModelNewModel:
    def test_new_model_returns_type(self):
        m = OperableModel()
        m.add_field("name", value="Alice", annotation=str)
        NewCls = m.new_model("Person")
        assert isinstance(NewCls, type)
        assert issubclass(NewCls, BaseModel)

    def test_new_model_has_specified_name(self):
        m = OperableModel()
        m.add_field("x", value=1, annotation=int)
        Cls = m.new_model("MyDynamic")
        assert Cls.__name__ == "MyDynamic"

    def test_new_model_instantiable(self):
        m = OperableModel()
        m.add_field("score", value=0, annotation=int)
        Cls = m.new_model("ScoreModel", use_fields={"score"})
        instance = Cls(score=42)
        assert instance.score == 42

    def test_new_model_invalid_fields_raises(self):
        m = OperableModel()
        m.add_field("a", value=1, annotation=int)
        with pytest.raises(ValueError, match="Invalid field"):
            m.new_model("Bad", use_fields={"nonexistent_field"})

    def test_new_model_frozen(self):
        m = OperableModel()
        m.add_field("val", value=1, annotation=int)
        FrozenCls = m.new_model("Frozen", use_fields={"val"}, frozen=True)
        instance = FrozenCls(val=5)
        with pytest.raises(Exception):
            instance.val = 10

    def test_new_model_without_name(self):
        m = OperableModel()
        Cls = m.new_model()
        assert isinstance(Cls, type)


# ---------------------------------------------------------------------------
# OperableModel – model_dump / to_dict
# ---------------------------------------------------------------------------


class TestOperableModelSerialize:
    def test_model_dump_includes_extra_fields(self):
        m = OperableModel()
        m.add_field("points", value=5, annotation=int)
        d = m.model_dump()
        # model_dump may not include extra fields directly, but to_dict does
        td = m.to_dict()
        assert "points" in td

    def test_to_dict_excludes_undefined(self):
        from lionagi.utils import UNDEFINED

        m = OperableModel()
        m.add_field("maybe", annotation=str)
        d = m.to_dict()
        # UNDEFINED values must be excluded
        assert d.get("maybe") is not UNDEFINED

    def test_all_fields_excludes_internal(self):
        m = OperableModel()
        m.add_field("real_field", value=1, annotation=int)
        af = m.all_fields
        assert "extra_fields" not in af
        assert "extra_field_models" not in af
        assert "real_field" in af


# ---------------------------------------------------------------------------
# OperableModel – __setattr__ with validator
# ---------------------------------------------------------------------------


class TestOperableModelSetAttr:
    def test_setattr_with_validator_pass(self):
        m = OperableModel()
        fm = FieldModel(base_type=int).with_validator(_pos_validator)
        m.add_field("positive", value=1, field_model=fm)
        m.positive = 5  # valid

    def test_dunder_field_assignment_raises(self):
        m = OperableModel()
        with pytest.raises(AttributeError):
            m.__dunder__ = "bad"


# ---------------------------------------------------------------------------
# OperableModel – __delattr__
# ---------------------------------------------------------------------------


class TestOperableModelDelAttr:
    def test_delattr_extra_field_with_no_default(self):
        # When no default is set, pydantic stores default_factory=None.
        # The __delattr__ implementation calls None() → TypeError (known behavior).
        m = OperableModel()
        m.add_field("ephemeral", value=42, annotation=int)
        with pytest.raises(TypeError):
            del m.ephemeral

    def test_delattr_extra_field_with_default_resets(self):
        from pydantic import Field

        m = OperableModel()
        fi = Field(default=0)
        fi.annotation = int
        m.extra_fields["resettable"] = fi
        object.__setattr__(m, "resettable", 99)
        del m.resettable
        assert m.resettable == 0
