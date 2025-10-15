"""
Test suite for lionagi/fields.py - Lazy import mechanism

Tests cover:
- All 14 lazy import paths
- Import caching behavior
- __all__ exports
- Error handling for invalid imports
- TYPE_CHECKING guard behavior
"""

import importlib
import sys
from unittest.mock import patch

import pytest


class TestLazyImports:
    """Test lazy import mechanism and caching."""

    def setup_method(self):
        """Reset lazy import cache before each test."""
        # Clear the module from sys.modules to force reimport
        if "lionagi.fields" in sys.modules:
            # Save the _lazy_imports state
            module = sys.modules["lionagi.fields"]
            if hasattr(module, "_lazy_imports"):
                module._lazy_imports.clear()

    def test_action_request_model_import(self):
        """Test lazy import of ActionRequestModel."""
        from lionagi.fields import ActionRequestModel

        assert ActionRequestModel is not None
        assert hasattr(ActionRequestModel, "__name__")

    def test_action_response_model_import(self):
        """Test lazy import of ActionResponseModel."""
        from lionagi.fields import ActionResponseModel

        assert ActionResponseModel is not None
        assert hasattr(ActionResponseModel, "__name__")

    def test_instruct_import(self):
        """Test lazy import of Instruct."""
        from lionagi.fields import Instruct

        assert Instruct is not None
        assert hasattr(Instruct, "__name__")

    def test_reason_import(self):
        """Test lazy import of Reason."""
        from lionagi.fields import Reason

        assert Reason is not None
        assert hasattr(Reason, "__name__")

    def test_get_default_field_import(self):
        """Test lazy import of get_default_field function."""
        from lionagi.fields import get_default_field

        assert get_default_field is not None
        assert callable(get_default_field)

    def test_action_requests_field_import(self):
        """Test lazy import of ACTION_REQUESTS_FIELD constant."""
        from lionagi.fields import ACTION_REQUESTS_FIELD

        assert ACTION_REQUESTS_FIELD is not None

    def test_action_responses_field_import(self):
        """Test lazy import of ACTION_RESPONSES_FIELD constant."""
        from lionagi.fields import ACTION_RESPONSES_FIELD

        assert ACTION_RESPONSES_FIELD is not None

    def test_action_required_field_import(self):
        """Test lazy import of ACTION_REQUIRED_FIELD constant."""
        from lionagi.fields import ACTION_REQUIRED_FIELD

        assert ACTION_REQUIRED_FIELD is not None

    def test_instruct_field_import(self):
        """Test lazy import of INSTRUCT_FIELD constant."""
        from lionagi.fields import INSTRUCT_FIELD

        assert INSTRUCT_FIELD is not None

    def test_list_instruct_field_model_import(self):
        """Test lazy import of LIST_INSTRUCT_FIELD_MODEL constant."""
        from lionagi.fields import LIST_INSTRUCT_FIELD_MODEL

        assert LIST_INSTRUCT_FIELD_MODEL is not None

    def test_reason_field_import(self):
        """Test lazy import of REASON_FIELD constant."""
        from lionagi.fields import REASON_FIELD

        assert REASON_FIELD is not None

    def test_import_caching_classes(self):
        """Test that second import of class uses cache."""
        import lionagi.fields as fields_module

        # First import
        first = fields_module.ActionRequestModel

        # Verify it's in the cache
        assert "ActionRequestModel" in fields_module._lazy_imports

        # Second import should use cache
        second = fields_module.ActionRequestModel
        assert first is second

    def test_import_caching_function(self):
        """Test that second import of function uses cache."""
        import lionagi.fields as fields_module

        # First import
        first = fields_module.get_default_field

        # Verify it's in the cache
        assert "get_default_field" in fields_module._lazy_imports

        # Second import should use cache
        second = fields_module.get_default_field
        assert first is second

    def test_field_constants_not_cached(self):
        """Test that FIELD constants are dynamically generated, not cached."""
        import lionagi.fields as fields_module

        # FIELD constants call get_default_field() each time
        # They shouldn't be in the cache
        _ = fields_module.ACTION_REQUESTS_FIELD

        # The function get_default_field should be cached after first use
        # but the FIELD constant itself should not be
        assert "ACTION_REQUESTS_FIELD" not in fields_module._lazy_imports


class TestExports:
    """Test __all__ exports."""

    def test_all_exports_present(self):
        """Test that all items in __all__ are importable."""
        import lionagi.fields as fields_module

        for name in fields_module.__all__:
            assert hasattr(fields_module, name), f"Missing export: {name}"
            attr = getattr(fields_module, name)
            assert attr is not None

    def test_all_exports_count(self):
        """Test that __all__ contains expected number of exports."""
        from lionagi.fields import __all__

        # 4 classes + 1 function + 6 FIELD constants = 11 total
        assert len(__all__) == 11

    def test_all_exports_list(self):
        """Test that __all__ contains exactly the expected items."""
        from lionagi.fields import __all__

        expected = {
            "ACTION_REQUESTS_FIELD",
            "ACTION_RESPONSES_FIELD",
            "ACTION_REQUIRED_FIELD",
            "INSTRUCT_FIELD",
            "LIST_INSTRUCT_FIELD_MODEL",
            "REASON_FIELD",
            "ActionRequestModel",
            "ActionResponseModel",
            "Instruct",
            "Reason",
            "get_default_field",
        }

        assert set(__all__) == expected


class TestErrorHandling:
    """Test error handling for invalid imports."""

    def test_invalid_attribute_raises_attribute_error(self):
        """Test that accessing invalid attribute raises AttributeError."""
        import lionagi.fields as fields_module

        with pytest.raises(AttributeError) as exc_info:
            _ = fields_module.NonExistentAttribute

        assert "has no attribute" in str(exc_info.value)
        assert "NonExistentAttribute" in str(exc_info.value)

    def test_invalid_attribute_error_message(self):
        """Test that error message contains module name and attribute."""
        import lionagi.fields as fields_module

        with pytest.raises(AttributeError) as exc_info:
            _ = fields_module.InvalidField

        error_msg = str(exc_info.value)
        assert "lionagi.fields" in error_msg
        assert "InvalidField" in error_msg


class TestFieldGeneration:
    """Test dynamic field generation using get_default_field."""

    def test_action_requests_field_uses_correct_params(self):
        """Test that ACTION_REQUESTS_FIELD calls get_default_field correctly."""
        from lionagi.fields import ACTION_REQUESTS_FIELD, get_default_field

        # Generate the same field manually
        manual_field = get_default_field("action_requests")

        # Compare key attributes
        assert type(ACTION_REQUESTS_FIELD) == type(manual_field)

    def test_list_instruct_field_uses_listable_param(self):
        """Test that LIST_INSTRUCT_FIELD_MODEL uses listable=True."""
        from lionagi.fields import LIST_INSTRUCT_FIELD_MODEL, get_default_field

        # Generate the same field manually
        manual_field = get_default_field("instruct", listable=True)

        # Compare types
        assert type(LIST_INSTRUCT_FIELD_MODEL) == type(manual_field)


class TestTypeChecking:
    """Test TYPE_CHECKING guard behavior."""

    def test_type_checking_imports_available(self):
        """Test that TYPE_CHECKING imports don't cause runtime issues."""
        # This test ensures that the TYPE_CHECKING block doesn't interfere
        # with normal operation and that lazy imports work correctly
        from lionagi.fields import (
            ACTION_REQUESTS_FIELD,
            ActionRequestModel,
            Instruct,
            Reason,
            get_default_field,
        )

        # All imports should work without issues
        assert ActionRequestModel is not None
        assert Instruct is not None
        assert Reason is not None
        assert get_default_field is not None
        assert ACTION_REQUESTS_FIELD is not None

    def test_type_checking_does_not_affect_lazy_imports(self):
        """Test that TYPE_CHECKING guard doesn't affect lazy loading."""
        import lionagi.fields as fields_module

        # Clear cache
        fields_module._lazy_imports.clear()

        # Import should still work via lazy loading
        action_request = fields_module.ActionRequestModel

        # Verify it was lazy loaded
        assert "ActionRequestModel" in fields_module._lazy_imports
        assert action_request is not None


class TestModuleReload:
    """Test module behavior on reload."""

    def test_lazy_imports_cache_persists(self):
        """Test that cache persists within same module instance."""
        import lionagi.fields as fields_module

        # Import something
        first = fields_module.ActionRequestModel

        # Check cache
        assert "ActionRequestModel" in fields_module._lazy_imports

        # Access again
        second = fields_module.ActionRequestModel

        # Should be same object from cache
        assert first is second

    def test_multiple_imports_use_cache(self):
        """Test that multiple different imports build up cache."""
        import lionagi.fields as fields_module

        # Clear cache
        fields_module._lazy_imports.clear()

        # Import multiple items
        _ = fields_module.ActionRequestModel
        _ = fields_module.ActionResponseModel
        _ = fields_module.Instruct
        _ = fields_module.get_default_field

        # All should be cached
        assert "ActionRequestModel" in fields_module._lazy_imports
        assert "ActionResponseModel" in fields_module._lazy_imports
        assert "Instruct" in fields_module._lazy_imports
        assert "get_default_field" in fields_module._lazy_imports


class TestImportIntegration:
    """Test integration with rest of lionagi package."""

    def test_import_from_package_root(self):
        """Test that fields can be imported from lionagi root if exposed."""
        # This tests the integration with the package structure
        try:
            from lionagi import fields

            assert fields is not None
            assert hasattr(fields, "ActionRequestModel")
        except ImportError:
            # If fields is not exposed at package root, that's fine
            # Just test that the module works standalone
            from lionagi.fields import ActionRequestModel

            assert ActionRequestModel is not None

    def test_all_field_constants_callable(self):
        """Test that all FIELD constants are properly created."""
        from lionagi.fields import (
            ACTION_REQUESTS_FIELD,
            ACTION_REQUIRED_FIELD,
            ACTION_RESPONSES_FIELD,
            INSTRUCT_FIELD,
            LIST_INSTRUCT_FIELD_MODEL,
            REASON_FIELD,
        )

        # All should be created without errors
        field_constants = [
            ACTION_REQUESTS_FIELD,
            ACTION_RESPONSES_FIELD,
            ACTION_REQUIRED_FIELD,
            INSTRUCT_FIELD,
            LIST_INSTRUCT_FIELD_MODEL,
            REASON_FIELD,
        ]

        for field in field_constants:
            assert field is not None

    def test_classes_and_functions_are_correct_types(self):
        """Test that imported classes and functions have correct types."""
        from lionagi.fields import (
            ActionRequestModel,
            ActionResponseModel,
            Instruct,
            Reason,
            get_default_field,
        )

        # Classes should be classes
        assert isinstance(ActionRequestModel, type)
        assert isinstance(ActionResponseModel, type)
        assert isinstance(Instruct, type)
        assert isinstance(Reason, type)

        # Function should be callable
        assert callable(get_default_field)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
