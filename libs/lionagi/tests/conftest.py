"""Configuration for all tests - asyncio backend only."""

import os

import pytest

# Ensure AnyIO's pytest plugin is loaded explicitly (even if autoload is disabled)
pytest_plugins = ("anyio",)


@pytest.fixture(scope="session")
def anyio_backend():
    """Force tests to run only on asyncio backend."""
    return "asyncio"
