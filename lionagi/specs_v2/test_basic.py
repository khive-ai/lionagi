"""Basic tests to verify LSpec v2 works.

Run with: pytest lionagi/specs_v2/test_basic.py
Or: python -m pytest lionagi/specs_v2/test_basic.py -v
"""

import pytest
from lionagi.specs_v2 import (
    FieldSpec,
    BackendRegistry,
    PydanticBackend,
)


def test_fieldspec_creation():
    """Test basic FieldSpec creation."""
    spec = FieldSpec(int, {"min": 0, "max": 100})
    assert spec.type == int
    assert spec.constraints == {"min": 0, "max": 100}


def test_fieldspec_with_constraint():
    """Test adding constraints."""
    spec = FieldSpec(int, {})
    spec = spec.with_constraint("min", 0)
    spec = spec.with_constraint("max", 100)

    assert spec.constraints == {"min": 0, "max": 100}


def test_fieldspec_constraint_replacement():
    """Test that adding same key replaces old value (fixes duplicate key issue)."""
    spec = FieldSpec(int, {"min": 0})
    spec = spec.with_constraint("min", 100)  # Should replace min=0

    assert spec.constraints == {"min": 100}
    assert len(spec.constraints) == 1


def test_fieldspec_nullable():
    """Test nullable transformation."""
    spec = FieldSpec(int, {"min": 0})
    nullable_spec = spec.as_nullable()

    assert nullable_spec.constraints["nullable"] is True
    assert nullable_spec.type == int  # Type unchanged


def test_fieldspec_listable():
    """Test listable transformation."""
    spec = FieldSpec(int, {"min": 0})
    list_spec = spec.as_listable()

    assert list_spec.type == list[int]
    assert list_spec.constraints["listable"] is True


def test_fieldspec_chaining():
    """Test chaining transformations."""
    spec = FieldSpec(str, {})
    result = spec.as_listable().as_nullable()

    assert result.type == list[str]
    assert result.constraints["listable"] is True
    assert result.constraints["nullable"] is True


def test_fieldspec_serialization():
    """Test to_dict and from_dict."""
    spec = FieldSpec(int, {"min": 0, "max": 100})
    spec_dict = spec.to_dict()

    assert spec_dict == {
        "type": "int",
        "constraints": {"min": 0, "max": 100}
    }

    # Deserialize
    restored = FieldSpec.from_dict(spec_dict)
    assert restored.type == int
    assert restored.constraints == {"min": 0, "max": 100}


def test_backend_registry_register():
    """Test registering a backend."""
    BackendRegistry.clear()  # Clean slate
    backend = PydanticBackend()

    BackendRegistry.register("test", backend)
    assert "test" in BackendRegistry.list_backends()


def test_backend_registry_default():
    """Test setting default backend."""
    BackendRegistry.clear()
    backend = PydanticBackend()

    BackendRegistry.register("pydantic", backend)
    BackendRegistry.set_default("pydantic")

    retrieved = BackendRegistry.get()
    assert retrieved is backend


def test_backend_registry_get_not_found():
    """Test getting non-existent backend raises error."""
    BackendRegistry.clear()

    with pytest.raises(ValueError, match="not registered"):
        BackendRegistry.get("nonexistent")


def test_pydantic_backend_validation():
    """Test Pydantic backend validates correctly."""
    BackendRegistry.clear()
    BackendRegistry.register("pydantic", PydanticBackend())

    spec = FieldSpec(int, {"min": 0, "max": 100})

    # Valid value
    result = BackendRegistry.validate(spec, 42)
    assert result == 42

    # Invalid value (too large)
    with pytest.raises(Exception):  # Pydantic ValidationError
        BackendRegistry.validate(spec, 150)


def test_pydantic_backend_nullable():
    """Test Pydantic backend handles nullable."""
    BackendRegistry.clear()
    BackendRegistry.register("pydantic", PydanticBackend())

    spec = FieldSpec(int, {"min": 0}).as_nullable()

    # None should be valid
    result = BackendRegistry.validate(spec, None)
    assert result is None

    # Regular int still valid
    result = BackendRegistry.validate(spec, 42)
    assert result == 42


def test_composition_reusable_specs():
    """Test creating reusable specs."""
    # Define reusable specs
    AGE_SPEC = FieldSpec(int, {"min": 0, "max": 120})
    EMAIL_SPEC = FieldSpec(str, {"pattern": r"^[\w\.-]+@[\w\.-]+$"})

    # Use them
    BackendRegistry.clear()
    BackendRegistry.register("pydantic", PydanticBackend())

    age = BackendRegistry.validate(AGE_SPEC, 25)
    assert age == 25

    # Email validation
    email = BackendRegistry.validate(EMAIL_SPEC, "user@example.com")
    assert email == "user@example.com"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
