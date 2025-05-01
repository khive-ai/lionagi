import os
import tempfile
from unittest.mock import patch

import pytest

from lionagi.scripts.build_hooks import AdapterRegistryBuildHook


def test_initialize():
    """Test the initialize hook."""
    version = "1.0.0"
    build_data = {"key": "value"}

    hook = AdapterRegistryBuildHook()
    result = hook.initialize(version, build_data)

    assert result == build_data


def test_clean():
    """Test the clean hook."""
    versions = ["1.0.0", "1.1.0"]
    build_data = {"key": "value"}

    hook = AdapterRegistryBuildHook()
    result = hook.clean(versions, build_data)

    assert result == build_data


def test_build_sdist():
    """Test the build_sdist hook."""
    with tempfile.TemporaryDirectory() as temp_dir:
        build_data = {"key": "value"}

        hook = AdapterRegistryBuildHook()
        with patch.object(hook, "_generate_adapter_registry") as mock_generate:
            result = hook.build_sdist(temp_dir, build_data)

            assert result == build_data
            assert mock_generate.called


def test_build_wheel():
    """Test the build_wheel hook."""
    with tempfile.TemporaryDirectory() as temp_dir:
        build_data = {"key": "value"}

        hook = AdapterRegistryBuildHook()
        with patch.object(hook, "_generate_adapter_registry") as mock_generate:
            result = hook.build_wheel(temp_dir, build_data)

            assert result == build_data
            assert mock_generate.called


def test_finalize():
    """Test the finalize hook."""
    build_data = {"key": "value"}

    hook = AdapterRegistryBuildHook()
    result = hook.finalize(build_data)

    assert result == build_data


def test_generate_adapter_registry():
    """Test the _generate_adapter_registry method."""
    hook = AdapterRegistryBuildHook()
    with patch("lionagi.scripts.build_adapter_registry.main") as mock_main:
        hook._generate_adapter_registry()

        assert mock_main.called


def test_generate_adapter_registry_error_handling():
    """Test that _generate_adapter_registry handles errors gracefully."""
    hook = AdapterRegistryBuildHook()
    with patch(
        "lionagi.scripts.build_adapter_registry.main",
        side_effect=Exception("Test error"),
    ):
        # Should not raise an exception
        hook._generate_adapter_registry()
