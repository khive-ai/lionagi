import os

import pytest

# Default backends: asyncio and trio
BACKENDS = ("asyncio", "trio")


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="append",
        default=list(BACKENDS),
        help="AnyIO backend(s) to run: asyncio, trio",
    )


def pytest_generate_tests(metafunc):
    if "anyio_backend" in metafunc.fixturenames:
        requested = metafunc.config.getoption("--backend")
        metafunc.parametrize("anyio_backend", requested)
