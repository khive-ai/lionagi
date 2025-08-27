"""Configuration for concurrency tests - asyncio backend only."""

import os

import pytest

# Ensure AnyIO's pytest plugin is loaded explicitly (even if autoload is disabled)
pytest_plugins = ("anyio",)


@pytest.fixture(scope="session")
def anyio_backend():
    """Force tests to run only on asyncio backend."""
    return "asyncio"


# (Optional but recommended for CI) Avoid 3rd-party plugin interference that can cause hangs:
# In CI, set:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1  and run:  pytest -p anyio -q
# You can also enforce here:
# os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
