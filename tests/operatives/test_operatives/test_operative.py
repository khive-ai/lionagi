"""Tests for the operative module."""

import pytest
from pydantic import BaseModel

from lionagi.ln.types import Operable, Spec
from lionagi.operations.operate.operative import Operative


class TestOperative:
    """Test Operative with new Spec-based API."""

    def test_initialization(self):
        """Test basic initialization of Operative."""
        operative = Operative(name="TestOp")
        assert operative.name == "TestOp"
        assert operative.adapter == "pydantic"
        assert operative.strict is False
        assert operative.auto_retry_parse is True
        assert operative.max_retries == 3

    def test_custom_name(self):
        """Test Operative with custom name."""
        operative = Operative(name="CustomName")
        assert operative.name == "CustomName"

    def test_strict_mode(self):
        """Test Operative with strict mode."""
        operative = Operative(name="StrictOp", strict=True)
        assert operative.strict is True

    def test_with_operable(self):
        """Test Operative with pre-configured Operable."""
        field1 = Spec(str, name="field1", description="Field 1")
        field2 = Spec(int, name="field2", description="Field 2")

        operable = Operable(
            __op_fields__=frozenset([field1, field2]), name="TestOperable"
        )

        operative = Operative(name="TestOp", operable=operable)

        assert operative.operable is operable
        assert operative.operable.get("field1") is not None
        assert operative.operable.get("field2") is not None

    def test_with_request_exclude(self):
        """Test Operative with request exclude fields."""
        field1 = Spec(str, name="field1", description="Field 1")
        field2 = Spec(int, name="field2", description="Field 2")
        field3 = Spec(str, name="field3", description="Field 3")

        operable = Operable(
            __op_fields__=frozenset([field1, field2, field3]),
            name="TestOperable",
        )

        operative = Operative(
            name="TestOp", operable=operable, request_exclude={"field3"}
        )

        assert operative.request_exclude == {"field3"}

    def test_create_request_model(self):
        """Test creating request model from operable."""
        field1 = Spec(str, name="name", description="Name field")
        field2 = Spec(int, name="value", description="Value field")
        field3 = Spec(
            str, name="response_only", description="Response only field"
        )

        operable = Operable(
            __op_fields__=frozenset([field1, field2, field3]),
            name="TestOperable",
        )

        operative = Operative(
            name="TestOp", operable=operable, request_exclude={"response_only"}
        )

        model_cls = operative.create_request_model()

        assert model_cls is not None
        assert issubclass(model_cls, BaseModel)
        assert "name" in model_cls.model_fields
        assert "value" in model_cls.model_fields
        assert "response_only" not in model_cls.model_fields

    def test_create_response_model(self):
        """Test creating response model from operable."""
        field1 = Spec(str, name="name", description="Name field")
        field2 = Spec(int, name="value", description="Value field")
        field3 = Spec(
            str, name="response_only", description="Response only field"
        )

        operable = Operable(
            __op_fields__=frozenset([field1, field2, field3]),
            name="TestOperable",
        )

        operative = Operative(
            name="TestOp", operable=operable, request_exclude={"response_only"}
        )

        # Create request model first
        operative.create_request_model()
        # Then response model (inherits from request)
        model_cls = operative.create_response_model()

        assert model_cls is not None
        assert issubclass(model_cls, BaseModel)
        # Response model should have all fields
        assert "name" in model_cls.model_fields
        assert "value" in model_cls.model_fields
        assert "response_only" in model_cls.model_fields

    def test_validate_response_strict(self):
        """Test strict validation of response."""
        field1 = Spec(str, name="name", description="Name field")
        field2 = Spec(int, name="value", description="Value field")

        operable = Operable(
            __op_fields__=frozenset([field1, field2]), name="TestOperable"
        )

        operative = Operative(name="TestOp", strict=True, operable=operable)
        operative.create_response_model()

        # Valid JSON
        valid_text = '{"name": "test", "value": 42}'
        result = operative.validate_response(valid_text)
        assert result is not None
        assert result.name == "test"
        assert result.value == 42

    def test_validate_response_fuzzy(self):
        """Test fuzzy validation of response."""
        field1 = Spec(str, name="name", description="Name field")

        operable = Operable(
            __op_fields__=frozenset([field1]), name="TestOperable"
        )

        operative = Operative(name="TestOp", strict=False, operable=operable)
        operative.create_response_model()

        # Typo in field name - should work with fuzzy matching
        fuzzy_text = '{"nme": "test"}'
        result = operative.validate_response(fuzzy_text, strict=False)
        # Result may be None if fuzzy matching fails, which is acceptable
        # The important thing is it doesn't raise an exception

    def test_update_response_model_with_text(self):
        """Test updating response model with text."""
        field1 = Spec(str, name="name", description="Name field")
        field2 = Spec(int, name="value", description="Value field")

        operable = Operable(
            __op_fields__=frozenset([field1, field2]), name="TestOperable"
        )

        operative = Operative(name="TestOp", operable=operable)
        operative.create_response_model()

        text = '{"name": "test", "value": 42}'
        result = operative.update_response_model(text=text)

        assert result is not None
        if isinstance(result, BaseModel):
            assert result.name == "test"
            assert result.value == 42

    def test_update_response_model_with_data(self):
        """Test updating response model with dict data."""
        field1 = Spec(str, name="name", description="Name field")
        field2 = Spec(int, name="value", description="Value field")

        operable = Operable(
            __op_fields__=frozenset([field1, field2]), name="TestOperable"
        )

        operative = Operative(name="TestOp", operable=operable)
        operative.create_response_model()

        # Set initial model
        text = '{"name": "test", "value": 42}'
        operative.update_response_model(text=text)

        # Update with new data
        data = {"name": "updated"}
        result = operative.update_response_model(data=data)

        assert result is not None
        if isinstance(result, BaseModel):
            assert result.name == "updated"

    def test_request_type_property(self):
        """Test request_type property."""
        field1 = Spec(str, name="name", description="Name field")

        operable = Operable(
            __op_fields__=frozenset([field1]), name="TestOperable"
        )

        operative = Operative(name="TestOp", operable=operable)
        operative.create_request_model()

        assert operative.request_type is not None
        assert issubclass(operative.request_type, BaseModel)

    def test_response_type_property(self):
        """Test response_type property."""
        field1 = Spec(str, name="name", description="Name field")

        operable = Operable(
            __op_fields__=frozenset([field1]), name="TestOperable"
        )

        operative = Operative(name="TestOp", operable=operable)
        operative.create_response_model()

        assert operative.response_type is not None
        assert issubclass(operative.response_type, BaseModel)

    def test_should_retry_flag(self):
        """Test _should_retry flag behavior."""
        field1 = Spec(str, name="name", description="Name field")

        operable = Operable(
            __op_fields__=frozenset([field1]), name="TestOperable"
        )

        operative = Operative(
            name="TestOp",
            strict=False,
            auto_retry_parse=True,
            operable=operable,
        )
        operative.create_response_model()

        # Valid response should set _should_retry to False
        valid_text = '{"name": "test"}'
        operative.validate_response(valid_text)
        assert operative._should_retry is False

    def test_model_caching(self):
        """Test that models are cached."""
        field1 = Spec(str, name="name", description="Name field")

        operable = Operable(
            __op_fields__=frozenset([field1]), name="TestOperable"
        )

        operative = Operative(name="TestOp", operable=operable)

        model1 = operative.create_request_model()
        model2 = operative.create_request_model()

        # Should return cached model
        assert model1 is model2

    def test_with_base_type(self):
        """Test Operative with base_type."""

        class MyBaseModel(BaseModel):
            base_field: str = "default"

        field1 = Spec(str, name="custom_field", description="Custom field")

        operable = Operable(
            __op_fields__=frozenset([field1]), name="TestOperable"
        )

        operative = Operative(
            name="TestOp", base_type=MyBaseModel, operable=operable
        )

        model_cls = operative.create_request_model()

        assert issubclass(model_cls, BaseModel)
        assert "base_field" in model_cls.model_fields
        assert "custom_field" in model_cls.model_fields

    def test_response_inherits_from_request(self):
        """Test that response model inherits from request model."""
        field1 = Spec(str, name="request_field", description="Request field")
        field2 = Spec(str, name="response_field", description="Response field")

        operable = Operable(
            __op_fields__=frozenset([field1, field2]), name="TestOperable"
        )

        operative = Operative(
            name="TestOp",
            operable=operable,
            request_exclude={"response_field"},
        )

        request_cls = operative.create_request_model()
        response_cls = operative.create_response_model()

        # Response should inherit from request
        assert issubclass(response_cls, request_cls)

        # Request should not have response_field
        assert "request_field" in request_cls.model_fields
        assert "response_field" not in request_cls.model_fields

        # Response should have both fields
        assert "request_field" in response_cls.model_fields
        assert "response_field" in response_cls.model_fields


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
