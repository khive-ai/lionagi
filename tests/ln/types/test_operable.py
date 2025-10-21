"""
Comprehensive tests for the Operable class.

Tests the immutable collection of Specs including:
- Basic Operable creation with fields
- Field access and retrieval
- Immutability guarantees
- Iteration and dictionary-like interface
"""

from dataclasses import FrozenInstanceError

import pytest

from lionagi.ln.types._sentinel import Undefined, Unset
from lionagi.ln.types.operable import Operable
from lionagi.ln.types.spec import Spec


class TestOperableBasics:
    """Test basic Operable functionality."""

    def test_operable_creation_empty(self):
        """Test creating an empty Operable."""
        op = Operable()

        assert op.__op_fields__ == frozenset()
        assert op.name is None

    def test_operable_creation_with_name(self):
        """Test creating Operable with a name."""
        op = Operable(name="TestOperable")

        assert op.name == "TestOperable"
        assert op.__op_fields__ == frozenset()

    def test_operable_creation_with_fields(self):
        """Test creating Operable with fields."""
        field1 = Spec(str, name="field1", description="First field")
        field2 = Spec(int, name="field2", description="Second field")

        op = Operable(
            __op_fields__=frozenset([field1, field2]), name="TestOperable"
        )

        assert len(op.__op_fields__) == 2
        assert field1 in op.__op_fields__
        assert field2 in op.__op_fields__

    def test_operable_is_frozen(self):
        """Test that Operable is immutable (frozen)."""
        op = Operable(name="Test")

        # Cannot modify attributes
        with pytest.raises((FrozenInstanceError, AttributeError)):
            op.name = "Modified"

        with pytest.raises((FrozenInstanceError, AttributeError)):
            op.__op_fields__ = frozenset()


class TestOperableFieldAccess:
    """Test field access methods."""

    def test_operable_get_existing_field(self):
        """Test getting an existing field by name."""
        field1 = Spec(str, name="username", required=True)
        field2 = Spec(int, name="age", ge=0)

        op = Operable(__op_fields__=frozenset([field1, field2]))

        # Get field by name
        retrieved = op.get("username")
        assert retrieved is field1
        assert retrieved.get("name") == "username"
        assert retrieved.get("required") is True

    def test_operable_get_nonexistent_field(self):
        """Test getting a non-existent field returns Unset."""
        op = Operable()

        result = op.get("nonexistent")
        assert result is Unset

    def test_operable_get_with_unnamed_field(self):
        """Test getting field when field has no name."""
        field_unnamed = Spec(str, description="No name")
        field_named = Spec(int, name="named")

        op = Operable(__op_fields__=frozenset([field_unnamed, field_named]))

        # Can get named field
        assert op.get("named") is field_named

        # Cannot get unnamed field by name
        assert op.get("") is Unset

    def test_operable_fields_accessible(self):
        """Test that fields are accessible via __op_fields__."""
        field1 = Spec(str, name="field1")
        field2 = Spec(int, name="field2")
        field3 = Spec(bool, name="field3")

        op = Operable(__op_fields__=frozenset([field1, field2, field3]))

        # All fields accessible via __op_fields__
        assert len(op.__op_fields__) == 3
        assert field1 in op.__op_fields__
        assert field2 in op.__op_fields__
        assert field3 in op.__op_fields__

    def test_operable_get_multiple_fields(self):
        """Test getting multiple fields by name."""
        field1 = Spec(str, name="username")
        field2 = Spec(int, name="age")

        op = Operable(__op_fields__=frozenset([field1, field2]))

        # Can get both fields
        assert op.get("username") is field1
        assert op.get("age") is field2
        assert op.get("email") is Unset


class TestOperableEquality:
    """Test Operable equality and hashing."""

    def test_operable_equality_same_fields(self):
        """Test equality with same fields."""
        field1 = Spec(str, name="field1")
        field2 = Spec(int, name="field2")

        op1 = Operable(__op_fields__=frozenset([field1, field2]), name="Test")
        op2 = Operable(__op_fields__=frozenset([field1, field2]), name="Test")

        assert op1 == op2

    def test_operable_equality_different_fields(self):
        """Test inequality with different fields."""
        field1 = Spec(str, name="field1")
        field2 = Spec(int, name="field2")
        field3 = Spec(bool, name="field3")

        op1 = Operable(__op_fields__=frozenset([field1, field2]))
        op2 = Operable(__op_fields__=frozenset([field2, field3]))

        assert op1 != op2

    def test_operable_equality_different_names(self):
        """Test inequality with different names."""
        field1 = Spec(str, name="field1")

        op1 = Operable(__op_fields__=frozenset([field1]), name="Name1")
        op2 = Operable(__op_fields__=frozenset([field1]), name="Name2")

        assert op1 != op2

    def test_operable_hashable(self):
        """Test that Operable is hashable."""
        field1 = Spec(str, name="field1")

        op1 = Operable(__op_fields__=frozenset([field1]), name="Test")
        op2 = Operable(__op_fields__=frozenset([field1]), name="Test")

        # Can be used in sets
        op_set = {op1, op2}
        assert len(op_set) == 1  # Same operable

    def test_operable_as_dict_key(self):
        """Test using Operable as dictionary key."""
        op1 = Operable(name="Key1")
        op2 = Operable(name="Key2")

        op_dict = {op1: "value1", op2: "value2"}

        assert op_dict[op1] == "value1"
        assert op_dict[op2] == "value2"


class TestOperableWithComplexSpecs:
    """Test Operable with complex Spec configurations."""

    def test_operable_with_nullable_fields(self):
        """Test Operable with nullable fields."""
        field1 = Spec(str, name="required_field", nullable=False)
        field2 = Spec(str, name="optional_field", nullable=True)

        op = Operable(__op_fields__=frozenset([field1, field2]))

        req_field = op.get("required_field")
        opt_field = op.get("optional_field")

        assert req_field.is_nullable is False
        assert opt_field.is_nullable is True

    def test_operable_with_default_values(self):
        """Test Operable with fields having defaults."""
        field1 = Spec(str, name="name", default="Unknown")
        field2 = Spec(int, name="count", default=0)
        field3 = Spec(list, name="tags", default_factory=list)

        op = Operable(__op_fields__=frozenset([field1, field2, field3]))

        assert op.get("name").get("default") == "Unknown"
        assert op.get("count").get("default") == 0
        assert callable(op.get("tags").get("default_factory"))

    def test_operable_with_validation_constraints(self):
        """Test Operable with validation constraints."""
        field1 = Spec(int, name="age", ge=0, le=120)
        field2 = Spec(str, name="email", pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
        field3 = Spec(float, name="score", ge=0.0, le=1.0)

        op = Operable(__op_fields__=frozenset([field1, field2, field3]))

        age_field = op.get("age")
        assert age_field.get("ge") == 0
        assert age_field.get("le") == 120

        email_field = op.get("email")
        assert email_field.get("pattern") is not None


class TestOperableFieldOrdering:
    """Test field ordering and preservation."""

    def test_operable_preserves_fields_as_set(self):
        """Test that fields are stored as an unordered set."""
        fields = [Spec(str, name=f"field{i}") for i in range(10)]

        op = Operable(__op_fields__=frozenset(fields))

        # All fields are present
        assert len(op.__op_fields__) == 10
        for field in fields:
            assert field in op.__op_fields__

    def test_operable_field_uniqueness(self):
        """Test that duplicate fields are handled correctly."""
        # Same field added twice (same instance due to caching)
        field1a = Spec(str, name="field1", description="Test")
        field1b = Spec(str, name="field1", description="Test")
        field2 = Spec(int, name="field2")

        op = Operable(__op_fields__=frozenset([field1a, field1b, field2]))

        # Should only have 2 fields (field1 deduplicated)
        assert len(op.__op_fields__) == 2


class TestOperableStringRepresentation:
    """Test string representations."""

    def test_operable_str(self):
        """Test __str__ representation."""
        field1 = Spec(str, name="test")
        op = Operable(__op_fields__=frozenset([field1]), name="TestOp")

        str_repr = str(op)
        # Should have some readable representation
        assert isinstance(str_repr, str)

    def test_operable_repr(self):
        """Test __repr__ representation."""
        field1 = Spec(str, name="test")
        op = Operable(__op_fields__=frozenset([field1]), name="TestOp")

        repr_str = repr(op)
        # Should contain class name
        assert "Operable" in repr_str


class TestOperableEdgeCases:
    """Test edge cases and unusual configurations."""

    def test_operable_with_sentinel_values(self):
        """Test Operable with sentinel values in Specs."""
        field1 = Spec(str, name="field1", default=Undefined)
        field2 = Spec(str, name="field2", default=Unset)
        field3 = Spec(str, name="field3", default=None)

        op = Operable(__op_fields__=frozenset([field1, field2, field3]))

        assert op.get("field1").get("default") is Undefined
        assert op.get("field2").get("default") is Unset
        assert op.get("field3").get("default") is None

    def test_operable_empty_name_string(self):
        """Test Operable with empty name."""
        op = Operable(name="")

        assert op.name == ""
        assert len(op.__op_fields__) == 0

    def test_operable_with_mixed_named_unnamed_fields(self):
        """Test Operable with mix of named and unnamed fields."""
        named1 = Spec(str, name="named1")
        unnamed1 = Spec(int, description="Unnamed")
        named2 = Spec(bool, name="named2")
        unnamed2 = Spec(float)

        op = Operable(
            __op_fields__=frozenset([named1, unnamed1, named2, unnamed2])
        )

        # Can get named fields
        assert op.get("named1") is named1
        assert op.get("named2") is named2

        # Cannot get unnamed fields by name
        assert op.get("") is Unset

        # But they're in the collection
        assert unnamed1 in op.__op_fields__
        assert unnamed2 in op.__op_fields__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
