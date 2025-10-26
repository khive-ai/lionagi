# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for main __init__.py module imports."""

import pytest

import lionagi
from lionagi.ln import import_module


class TestMainImports:
    """Tests for main lionagi package imports."""

    # All exports from lionagi.__all__ (alphabetically sorted)
    EXPECTED_EXPORTS = (
        "__version__",
        "BaseModel",
        "Branch",
        "Broadcaster",
        "Builder",
        "DataClass",
        "Edge",
        "Element",
        "Event",
        "Field",
        "FieldModel",
        "Graph",
        "HookRegistry",
        "HookedEvent",
        "Node",
        "Operable",
        "OperableModel",
        "Operation",
        "Params",
        "Pile",
        "Progression",
        "Session",
        "Spec",
        "Undefined",
        "Unset",
        "iModel",
        "ln",
        "load_mcp_tools",
        "logger",
        "types",
    )

    def test_all_exports_defined(self):
        """Test that __all__ is defined and contains expected exports."""
        assert hasattr(lionagi, "__all__")
        assert lionagi.__all__ == self.EXPECTED_EXPORTS

    def test_all_exports_alphabetically_sorted(self):
        """Test that __all__ exports are alphabetically sorted.

        Note: Dunder names (like __version__) come first by convention,
        then regular names are sorted alphabetically.
        """
        # Separate dunder names from regular names
        dunder_names = [
            name for name in lionagi.__all__ if name.startswith("__")
        ]
        regular_names = [
            name for name in lionagi.__all__ if not name.startswith("__")
        ]

        # Check dunder names are sorted
        assert tuple(dunder_names) == tuple(sorted(dunder_names))

        # Check regular names are sorted
        assert tuple(regular_names) == tuple(sorted(regular_names))

        # Check dunder names come before regular names
        expected_exports = tuple(sorted(dunder_names)) + tuple(
            sorted(regular_names)
        )
        assert lionagi.__all__ == expected_exports

    @pytest.mark.parametrize("export_name", EXPECTED_EXPORTS)
    def test_import_all_exports(self, export_name):
        """Test that each export in __all__ can be imported."""
        obj = import_module("lionagi", import_name=export_name)
        assert obj is not None

    @pytest.mark.parametrize("export_name", EXPECTED_EXPORTS)
    def test_getattr_all_exports(self, export_name):
        """Test that each export can be accessed via getattr."""
        obj = getattr(lionagi, export_name)
        assert obj is not None

    def test_lazy_import_caching(self):
        """Test that lazy imports are cached after first access."""
        # First access
        session1 = lionagi.Session
        # Second access should return cached version
        session2 = lionagi.Session
        assert session1 is session2

    def test_invalid_import_raises_attribute_error(self):
        """Test that importing non-existent attribute raises AttributeError."""
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = lionagi.NonExistentAttribute

    def test_pydantic_imports(self):
        """Test that pydantic re-exports work correctly."""
        from pydantic import BaseModel, Field

        assert lionagi.BaseModel is BaseModel
        assert lionagi.Field is Field

    def test_ln_import(self):
        """Test that ln submodule is directly importable."""
        from lionagi import ln

        assert hasattr(ln, "import_module")
        assert hasattr(ln, "types")

    def test_types_module_import(self):
        """Test that types module is importable."""
        from lionagi import types

        assert types is not None
        # Should be the _types module
        assert hasattr(types, "__name__")

    def test_version_import(self):
        """Test that __version__ is importable and is a string."""
        from lionagi import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_logger_import(self):
        """Test that logger is importable."""
        from lionagi import logger

        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")

    def test_data_classes_import(self):
        """Test that ln.types data classes are importable."""
        from lionagi import DataClass, Params, Undefined, Unset

        assert DataClass is not None
        assert Params is not None
        assert Undefined is not None
        assert Unset is not None


class TestLazyLoadingBehavior:
    """Tests for lazy loading mechanism."""

    def test_lazy_loading_on_first_access(self):
        """Test that objects are loaded on first access."""
        # Access a lazy-loaded object
        branch = lionagi.Branch
        assert branch is not None
        # Should now be in lazy imports cache
        assert "Branch" in lionagi._lazy_imports

    def test_multiple_imports_same_object(self):
        """Test that multiple imports return same cached object."""
        obj1 = lionagi.iModel
        obj2 = lionagi.iModel
        obj3 = getattr(lionagi, "iModel")
        assert obj1 is obj2 is obj3

    def test_all_protocol_types_importable(self):
        """Test that all protocol types are importable."""
        from lionagi import (
            Edge,
            Element,
            Event,
            Graph,
            Node,
            Pile,
            Progression,
        )

        assert Element is not None
        assert Pile is not None
        assert Progression is not None
        assert Node is not None
        assert Edge is not None
        assert Graph is not None
        assert Event is not None

    def test_all_models_importable(self):
        """Test that all model types are importable."""
        from lionagi import FieldModel, OperableModel

        assert FieldModel is not None
        assert OperableModel is not None

    def test_all_service_types_importable(self):
        """Test that all service types are importable."""
        from lionagi import Broadcaster, HookedEvent, HookRegistry, iModel

        assert iModel is not None
        assert HookRegistry is not None
        assert HookedEvent is not None
        assert Broadcaster is not None

    def test_all_operation_types_importable(self):
        """Test that all operation types are importable."""
        from lionagi import Builder, Operation, load_mcp_tools

        assert Builder is not None
        assert Operation is not None
        assert load_mcp_tools is not None

    def test_all_session_types_importable(self):
        """Test that all session types are importable."""
        from lionagi import Branch, Session

        assert Session is not None
        assert Branch is not None
