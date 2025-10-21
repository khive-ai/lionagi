"""
Comprehensive tests for the Spec class.

Tests the framework-agnostic field specification system including:
- Basic Spec creation and properties
- Thread-safe caching
- Spec updates and immutability
- Nullable field handling
- Comparison and hashing
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pytest

from lionagi.ln.types._sentinel import Undefined, Unset
from lionagi.ln.types.spec import Spec


class TestSpecBasics:
    """Test basic Spec functionality."""

    def test_spec_creation_with_type(self):
        """Test creating a Spec with a type."""
        spec = Spec(str)
        assert spec.base_type == str

    def test_spec_creation_with_kwargs(self):
        """Test creating a Spec with various kwargs."""
        spec = Spec(
            int, name="age", description="User age", ge=0, le=120, default=25
        )

        assert spec.base_type == int
        assert spec.get("name") == "age"
        assert spec.get("description") == "User age"
        assert spec.get("ge") == 0
        assert spec.get("le") == 120
        assert spec.get("default") == 25

    def test_spec_get_method(self):
        """Test the get method with defaults."""
        spec = Spec(str, name="test")

        assert spec.get("name") == "test"
        assert spec.get("missing") is None
        assert spec.get("missing", "default") == "default"

    def test_spec_metadata_access(self):
        """Test accessing metadata via get method."""
        spec = Spec(str, name="test", required=True, description="A test")

        assert spec.get("name") == "test"
        assert spec.get("required") is True
        assert spec.get("description") == "A test"
        assert spec.get("nonexistent") is None
        assert spec.get("nonexistent", "default") == "default"


class TestSpecImmutability:
    """Test Spec immutability and updates."""

    def test_spec_is_immutable(self):
        """Test that Spec is immutable (frozen dataclass)."""
        spec = Spec(str, name="test")

        # Spec is a frozen dataclass, so attributes cannot be modified
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            spec.base_type = int

    def test_spec_with_updates(self):
        """Test creating a new Spec with updates."""
        spec1 = Spec(str, name="test", description="Original")
        spec2 = spec1.with_updates(description="Updated", required=True)

        # Original unchanged
        assert spec1.get("description") == "Original"
        assert spec1.get("required") is None

        # New spec has updates
        assert spec2.get("description") == "Updated"
        assert spec2.get("required") is True
        assert spec2.get("name") == "test"  # Preserved
        assert spec2.base_type == str  # Preserved


class TestSpecNullable:
    """Test nullable field handling."""

    def test_spec_nullable_property_true(self):
        """Test is_nullable property when nullable=True."""
        spec = Spec(str, nullable=True)
        assert spec.is_nullable is True

    def test_spec_nullable_property_false(self):
        """Test is_nullable property when nullable=False."""
        spec = Spec(str, nullable=False)
        assert spec.is_nullable is False

    def test_spec_nullable_property_default(self):
        """Test is_nullable property when nullable not set."""
        spec = Spec(str)
        assert spec.is_nullable is False  # Default

    def test_spec_nullable_with_optional_type(self):
        """Test nullable with Optional type hint."""
        spec = Spec(Optional[str], nullable=True)
        assert spec.is_nullable is True
        assert spec.base_type == Optional[str]


class TestSpecAnnotatedCaching:
    """Test thread-safe caching for annotated() method."""

    def test_spec_annotated_caching(self):
        """Test that annotated() method returns cached results."""
        spec = Spec(str, name="test", required=True)

        # First call creates and caches
        annotated1 = spec.annotated()

        # Second call should return the same cached instance
        annotated2 = spec.annotated()

        assert annotated1 is annotated2

    def test_spec_different_specs_different_annotated(self):
        """Test that different Specs return different annotated types."""
        spec1 = Spec(str, name="test1")
        spec2 = Spec(str, name="test2")

        # Different specs should have different annotated types
        assert spec1.annotated() is not spec2.annotated()

    def test_spec_caching_thread_safety(self):
        """Test that annotated() caching is thread-safe."""
        spec = Spec(str, name="shared", value=42)
        results = []

        def get_annotated(i):
            # All threads call annotated() on the same spec
            annotated = spec.annotated()
            results.append(annotated)

        # Run in multiple threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_annotated, i) for i in range(100)]
            for future in futures:
                future.result()

        # All should be the same cached instance
        first = results[0]
        for annotated in results[1:]:
            assert annotated is first


class TestSpecComparison:
    """Test Spec comparison and hashing."""

    def test_spec_equality(self):
        """Test Spec equality comparison."""
        spec1 = Spec(str, name="test", required=True)
        spec2 = Spec(str, name="test", required=True)
        spec3 = Spec(str, name="different")

        assert spec1 == spec2  # Same due to caching
        assert spec1 != spec3
        assert spec2 != spec3

    def test_spec_hashing(self):
        """Test that Specs are hashable."""
        spec1 = Spec(str, name="test")
        spec2 = Spec(str, name="test")
        spec3 = Spec(int, name="test")

        # Can be used in sets
        spec_set = {spec1, spec2, spec3}
        assert len(spec_set) == 2  # spec1 and spec2 are same

    def test_spec_as_dict_key(self):
        """Test using Spec as dictionary key."""
        spec1 = Spec(str, name="test")
        spec2 = Spec(str, name="test")
        spec3 = Spec(int, name="other")

        spec_dict = {spec1: "value1", spec3: "value3"}

        # spec2 should access same as spec1
        assert spec_dict[spec2] == "value1"
        assert spec_dict[spec3] == "value3"


class TestSpecSentinels:
    """Test Spec with sentinel values."""

    def test_spec_with_undefined_default(self):
        """Test Spec with Undefined as default."""
        spec = Spec(str, default=Undefined)
        assert spec.get("default") is Undefined

    def test_spec_with_unset_value(self):
        """Test Spec with Unset value."""
        spec = Spec(str, name="test", value=Unset)
        assert spec.get("value") is Unset

    def test_spec_distinguishes_sentinels(self):
        """Test that sentinels are distinguished from None."""
        spec1 = Spec(str, default=None)
        spec2 = Spec(str, default=Undefined)
        spec3 = Spec(str, default=Unset)

        assert spec1.get("default") is None
        assert spec2.get("default") is Undefined
        assert spec3.get("default") is Unset

        # All different
        assert spec1 != spec2
        assert spec2 != spec3
        assert spec1 != spec3


class TestSpecComplexTypes:
    """Test Spec with complex types."""

    def test_spec_with_list_type(self):
        """Test Spec with list type."""
        spec = Spec(list[str], name="tags", default=[])
        assert spec.base_type == list[str]
        assert spec.get("default") == []

    def test_spec_with_dict_type(self):
        """Test Spec with dict type."""
        spec = Spec(dict[str, int], name="scores")
        assert spec.base_type == dict[str, int]

    def test_spec_with_union_type(self):
        """Test Spec with Union type."""
        from typing import Union

        spec = Spec(Union[str, int], name="mixed")
        assert spec.base_type == Union[str, int]

    def test_spec_with_custom_class(self):
        """Test Spec with custom class type."""

        class CustomClass:
            pass

        spec = Spec(CustomClass, name="custom")
        assert spec.base_type == CustomClass


class TestSpecValidationConstraints:
    """Test Spec with validation constraints."""

    def test_spec_numeric_constraints(self):
        """Test numeric validation constraints."""
        spec = Spec(
            int,
            name="age",
            ge=0,  # greater or equal
            le=120,  # less or equal
            gt=-1,  # greater than
            lt=121,  # less than
        )

        assert spec.get("ge") == 0
        assert spec.get("le") == 120
        assert spec.get("gt") == -1
        assert spec.get("lt") == 121

    def test_spec_string_constraints(self):
        """Test string validation constraints."""
        spec = Spec(
            str,
            name="username",
            min_length=3,
            max_length=20,
            pattern=r"^[a-zA-Z0-9_]+$",
        )

        assert spec.get("min_length") == 3
        assert spec.get("max_length") == 20
        assert spec.get("pattern") == r"^[a-zA-Z0-9_]+$"

    def test_spec_with_choices(self):
        """Test Spec with enumerated choices."""
        spec = Spec(
            str,
            name="status",
            choices=["pending", "active", "completed"],
            default="pending",
        )

        assert spec.get("choices") == ["pending", "active", "completed"]
        assert spec.get("default") == "pending"


class TestSpecStringRepresentation:
    """Test string representation of Spec."""

    def test_spec_str(self):
        """Test __str__ method."""
        spec = Spec(str, name="test", required=True)
        str_repr = str(spec)

        # Should be a readable representation
        assert "Spec" in str_repr or "{" in str_repr

    def test_spec_repr(self):
        """Test __repr__ method."""
        spec = Spec(str, name="test")
        repr_str = repr(spec)

        # Should be a valid representation
        assert "Spec" in repr_str or "{" in repr_str


class TestSpecEdgeCases:
    """Test edge cases and error conditions."""

    def test_spec_empty(self):
        """Test creating Spec with no type."""
        spec = Spec()
        assert spec.base_type is None
        assert len(spec.metadata) == 0

    def test_spec_type_only(self):
        """Test creating Spec with type only."""
        spec = Spec(str)
        assert spec.base_type == str
        # Only base_type set, no metadata
        assert len(spec.metadata) == 0

    def test_spec_with_many_fields(self):
        """Test Spec with many fields."""
        spec = Spec(
            str,
            name="field",
            description="A field",
            required=True,
            nullable=False,
            default="value",
            min_length=1,
            max_length=100,
            pattern=r".*",
            example="example",
            title="Field Title",
            deprecated=False,
        )

        # All fields should be accessible via get()
        assert spec.get("name") == "field"
        assert spec.get("required") is True
        assert spec.get("description") == "A field"
        assert (
            len(spec.metadata) == 11
        )  # All metadata fields (base_type is separate)

    def test_spec_with_callable_default(self):
        """Test Spec with callable as default factory."""

        def default_factory():
            return []

        spec = Spec(list, default_factory=default_factory)
        assert spec.get("default_factory") is default_factory


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
