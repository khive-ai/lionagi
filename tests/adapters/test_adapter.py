from unittest.mock import Mock

import pytest

from lionagi._errors import MissingAdapterError
from lionagi.adapters.adapter import Adapter, AdapterRegistry


def test_get_raises_missingadaptererror_for_unknown_key():
    """
    Test that AdapterRegistry.get() raises MissingAdapterError for an unknown key.
    """
    # Clear the registry to ensure a clean test environment
    AdapterRegistry._adapters = {}

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

    # Create a mock adapter
    mock_adapter = Mock(spec=Adapter)
    mock_adapter.obj_key = "test_adapter"

    # Register the adapter
    AdapterRegistry.register(mock_adapter)

    # Act
    retrieved_adapter = AdapterRegistry.get("test_adapter")

    # Assert
    assert retrieved_adapter is mock_adapter
