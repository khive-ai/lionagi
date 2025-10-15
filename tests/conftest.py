# tests/conftest.py
import json
import sys
import types

import pytest


@pytest.fixture
def ensure_fake_lionagi(monkeypatch):
    """
    If lionagi is not installed in the environment, install minimal stubs so the
    tests exercising chunk_content/chunk() can run.
    """
    if "lionagi" in sys.modules:
        # Real lionagi present; do nothing.
        yield
        return

    pkg = types.ModuleType("lionagi")

    # ln: provide lcall (with optional flatten) and json_dumps
    ln_ns = types.SimpleNamespace()

    def lcall(
        items, func, *args, flatten=False, output_flatten=False, **kwargs
    ):
        results = []
        for x in items:
            r = func(x, *args, **kwargs)
            if (flatten or output_flatten) and isinstance(r, list):
                results.extend(r)
            else:
                results.append(r)
        return results

    ln_ns.lcall = lcall
    ln_ns.json_dumps = staticmethod(lambda d: json.dumps(d))
    pkg.ln = ln_ns

    # utils: is_import_installed
    utils_mod = types.ModuleType("lionagi.utils")

    def is_import_installed(name: str) -> bool:
        try:
            __import__(name)
            return True
        except ImportError:
            return False

    utils_mod.is_import_installed = is_import_installed

    # protocols.graph.node: Node
    protocols_mod = types.ModuleType("lionagi.protocols")
    graph_mod = types.ModuleType("lionagi.protocols.graph")
    node_mod = types.ModuleType("lionagi.protocols.graph.node")

    class Node:
        def __init__(self, content, metadata):
            self.content = content
            self.metadata = metadata

        def __repr__(self):
            return (
                f"Node(content={self.content!r}, metadata={self.metadata!r})"
            )

    node_mod.Node = Node

    sys.modules["lionagi"] = pkg
    sys.modules["lionagi.utils"] = utils_mod
    sys.modules["lionagi.protocols"] = protocols_mod
    sys.modules["lionagi.protocols.graph"] = graph_mod
    sys.modules["lionagi.protocols.graph.node"] = node_mod
    yield


@pytest.fixture(scope="session")
def mod_paths():
    """
    Resolve module paths for the code under test from env vars with sensible defaults.
    Adjust env vars to match your layout:
      UUT_CHUNK_MOD  (chunk_by_chars/tokens, chunk_content)
      UUT_API_MOD    (dir_to_files, chunk)
      UUT_SCHEMA_MOD (load_pydantic_model_from_schema)
    """
    import os

    return {
        "chunk_mod": os.getenv("UUT_CHUNK_MOD", "lionagi.libs.file.chunk"),
        "api_mod": os.getenv("UUT_API_MOD", "lionagi.libs.file.process"),
        "schema_mod": os.getenv(
            "UUT_SCHEMA_MOD",
            "lionagi.libs.schema.load_pydantic_model_from_schema",
        ),
    }


# =============================================================================
# Shared Service Layer Fixtures (Phase 2 Consolidation)
# =============================================================================


@pytest.fixture
def openai_endpoint_config():
    """Standard OpenAI endpoint configuration for testing."""
    from lionagi.service.connections.endpoint_config import EndpointConfig

    return EndpointConfig(
        name="test_endpoint",
        provider="openai",
        endpoint="chat",
        base_url="https://api.openai.com/v1",
        endpoint_params=["chat", "completions"],
        openai_compatible=True,
        api_key="test-key",
    )


@pytest.fixture
def anthropic_endpoint_config():
    """Standard Anthropic endpoint configuration for testing."""
    from lionagi.service.connections.endpoint_config import EndpointConfig

    return EndpointConfig(
        name="anthropic_chat",
        provider="anthropic",
        endpoint="messages",
        base_url="https://api.anthropic.com/v1",
        endpoint_params=["messages"],
        openai_compatible=False,
        api_key="test-key",
    )


@pytest.fixture
def base_imodel():
    """Basic OpenAI iModel instance for testing."""
    from lionagi.service.imodel import iModel

    return iModel(provider="openai", model="gpt-4.1-mini", api_key="test-key")


@pytest.fixture
def anthropic_imodel():
    """Anthropic iModel instance for testing."""
    from lionagi.service.imodel import iModel

    return iModel(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        api_key="test-key",
    )


@pytest.fixture
def mock_branch_factory():
    """Factory for creating mock Branch instances with customizable behavior."""
    from unittest.mock import AsyncMock, MagicMock

    from lionagi.session.branch import Branch

    def _factory(selected_items=None, operate_response=None):
        """
        Create a mock branch with customizable operate response.

        Args:
            selected_items: List of items to return in SelectionModel
            operate_response: Custom response object to return from operate
        """
        from lionagi.operations.select.utils import SelectionModel

        branch = MagicMock(spec=Branch)

        if operate_response is not None:
            # Use custom response
            async def mock_operate(**kwargs):
                return operate_response

        elif selected_items is not None:
            # Use SelectionModel with provided items
            async def mock_operate(**kwargs):
                return SelectionModel(selected=selected_items)

        else:
            # Default: return empty SelectionModel
            async def mock_operate(**kwargs):
                return SelectionModel(selected=[])

        branch.operate = AsyncMock(side_effect=mock_operate)
        return branch

    return _factory


@pytest.fixture
def mock_response():
    """Standard mock API response for testing."""
    from unittest.mock import MagicMock

    response = MagicMock()
    response.json.return_value = {
        "choices": [
            {"message": {"content": "Test response", "role": "assistant"}}
        ],
        "model": "gpt-4.1-mini",
        "usage": {
            "total_tokens": 50,
            "prompt_tokens": 20,
            "completion_tokens": 30,
        },
    }
    return response


@pytest.fixture
def mock_streaming_response():
    """Mock streaming response for testing streaming operations."""

    class MockStreamingResponse:
        def __init__(self):
            self.chunks = [
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": " world"}}]},
                {"choices": [{"delta": {}}]},  # End marker
            ]

        async def __aiter__(self):
            for chunk in self.chunks:
                yield chunk

    return MockStreamingResponse()
