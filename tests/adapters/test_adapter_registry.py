import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lionagi._errors import MissingAdapterError
from lionagi.adapters.adapter import Adapter, AdapterRegistry


def test_get_raises_missingadaptererror_for_unknown_key():
    """
    Test that AdapterRegistry.get() raises MissingAdapterError for an unknown key.
    """
    # Clear the registry to ensure a clean test environment
    AdapterRegistry._adapters = {}
    AdapterRegistry._adapter_map = {}
    AdapterRegistry._initialized = False

    with pytest.raises(MissingAdapterError) as excinfo:
        AdapterRegistry.get("bogus_key")

    # Assert that the correct exception type is raised
    assert excinfo.type is MissingAdapterError
    # Assert the content of the exception message
    assert "Adapter for key 'bogus_key' not found" in str(excinfo.value)


def test_get_returns_registered_adapter():
    """
    Test that AdapterRegistry.get() returns the correct adapter for a known key.
    """
    # Clear the registry to ensure a clean test environment
    AdapterRegistry._adapters = {}
    AdapterRegistry._adapter_map = {}
    AdapterRegistry._initialized = False

    # Create a mock adapter
    mock_adapter = Mock(spec=Adapter)
    mock_adapter.obj_key = "test_adapter"

    # Register the adapter
    AdapterRegistry.register(mock_adapter)

    # Act
    retrieved_adapter = AdapterRegistry.get("test_adapter")

    # Assert
    assert retrieved_adapter is mock_adapter


def test_load_adapter_from_map():
    """
    Test that AdapterRegistry can load adapters from the pre-computed map.
    """
    # Clear the registry to ensure a clean test environment
    AdapterRegistry._adapters = {}
    AdapterRegistry._adapter_map = {}
    AdapterRegistry._initialized = False

    # Create a temporary adapter map file
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", delete=False
    ) as temp_file:
        adapter_map = {"test_key": "lionagi.adapters.json_adapter.JsonAdapter"}
        json.dump(adapter_map, temp_file)
        temp_file_path = temp_file.name

    try:
        # Mock the package directory to point to the temporary file
        package_dir = os.path.dirname(temp_file_path)
        adapter_map_path = temp_file_path

        with (
            patch("os.path.dirname", return_value=package_dir),
            patch("os.path.join", return_value=adapter_map_path),
        ):
            # Initialize the registry
            AdapterRegistry._initialize()

            # Verify the adapter map was loaded
            assert "test_key" in AdapterRegistry._adapter_map
            assert (
                AdapterRegistry._adapter_map["test_key"]
                == "lionagi.adapters.json_adapter.JsonAdapter"
            )
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


def test_import_adapter_from_map():
    """
    Test that AdapterRegistry can import adapters from the pre-computed map.
    """
    # Clear the registry to ensure a clean test environment
    AdapterRegistry._adapters = {}
    AdapterRegistry._adapter_map = {
        "json": "lionagi.adapters.json_adapter.JsonAdapter"
    }
    AdapterRegistry._initialized = True

    # Create a mock module with a mock adapter class
    mock_module = Mock()
    mock_adapter_class = Mock(spec=Adapter)
    mock_adapter_class.obj_key = "json"
    mock_adapter_instance = Mock(spec=Adapter)
    mock_adapter_class.return_value = mock_adapter_instance

    # Set the mock adapter class as an attribute of the mock module
    mock_module.JsonAdapter = mock_adapter_class

    # Mock importlib.import_module to return our mock module
    with patch("importlib.import_module", return_value=mock_module):
        # Get the adapter
        adapter = AdapterRegistry._import_adapter("json")

        # Verify the adapter was imported and registered
        assert adapter is not None
        assert "json" in AdapterRegistry._adapters
