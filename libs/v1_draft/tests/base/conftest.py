# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Test configuration for base module tests.

Forces asyncio backend to avoid Trio/pytest interaction edge cases.
Keeps sleeps tiny and bounded; prefers fail_after/move_on_after for intentional blocking.
"""

import pytest


@pytest.fixture
def anyio_backend():
    """Force asyncio backend for all async tests to avoid Trio edge cases."""
    return "asyncio"
