import pytest
from unittest.mock import Mock
from lionagi._class_registry import AdapterRegistry
from lionagi._errors import MissingAdapterError

def test_get_raises_missingadaptererror_for_unknown_key():
    """
    Test that AdapterRegistry.get() raises MissingAdapterError for an unknown key.
    """
    registry = AdapterRegistry()
    with pytest.raises(MissingAdapterError) as excinfo:
        registry.get("bogus_key")
    # Assert that the correct exception type is raised
    assert excinfo.type is MissingAdapterError
    # Optionally, assert the content of the exception message
    assert "Adapter for key 'bogus_key' not found" in str(excinfo.value)

def test_get_returns_registered_adapter():
    """
    Test that AdapterRegistry.get() returns the correct adapter for a known key.
    """
    registry = AdapterRegistry()
    mock_adapter = Mock()
    registry.register("test_adapter", mock_adapter)

    # Act
    retrieved_adapter = registry.get("test_adapter")

    # Assert
    assert retrieved_adapter is mock_adapter