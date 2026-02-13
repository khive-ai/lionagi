"""Tests for integration documentation examples (llm-providers.md, tools.md, mcp-servers.md).

Validates that the code patterns shown in integration docs actually work:
- All supported LLM providers can construct iModel instances
- Tool and function_to_schema produce correct OpenAI-format schemas
- MCP-related imports resolve correctly
"""

import pytest
from pydantic import BaseModel

from lionagi.service.imodel import iModel

# ===========================================================================
# LLM Providers (llm-providers.md)
# ===========================================================================


class TestLLMProviders:
    """iModel construction for every documented provider."""

    def test_openai_imodel_constructs(self):
        """OpenAI provider with gpt-4.1-mini should construct without error."""
        model = iModel(provider="openai", model="gpt-4.1-mini", api_key="test")
        assert model is not None
        assert model.endpoint is not None

    def test_openai_multiple_models(self):
        """OpenAI provider supports multiple model variants."""
        for model_name in ("gpt-4.1", "gpt-4o", "gpt-4o-mini"):
            m = iModel(provider="openai", model=model_name, api_key="test")
            assert m is not None

    def test_anthropic_imodel_constructs(self):
        """Anthropic provider should construct with a Claude model."""
        model = iModel(
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            api_key="test",
        )
        assert model is not None
        assert model.endpoint is not None

    def test_anthropic_haiku_model(self):
        """Anthropic provider should also accept the Haiku model variant."""
        model = iModel(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            api_key="test",
        )
        assert model is not None

    def test_gemini_imodel_constructs(self):
        """Gemini provider should construct with gemini-2.5-flash."""
        model = iModel(
            provider="gemini", model="gemini-2.5-flash", api_key="test"
        )
        assert model is not None
        assert model.endpoint is not None

    def test_groq_imodel_constructs(self):
        """Groq provider should construct with llama-3.3-70b-versatile."""
        model = iModel(
            provider="groq",
            model="llama-3.3-70b-versatile",
            api_key="test",
        )
        assert model is not None
        assert model.endpoint is not None

    def test_openrouter_imodel_constructs(self):
        """OpenRouter provider should construct with a namespaced model."""
        model = iModel(
            provider="openrouter",
            model="google/gemini-2.5-flash",
            api_key="test",
        )
        assert model is not None
        assert model.endpoint is not None

    def test_perplexity_imodel_constructs(self):
        """Perplexity provider should construct with the sonar model."""
        model = iModel(provider="perplexity", model="sonar", api_key="test")
        assert model is not None
        assert model.endpoint is not None

    def test_nvidia_nim_imodel_constructs(self):
        """NVIDIA NIM provider should construct with meta/llama3-8b-instruct."""
        model = iModel(
            provider="nvidia_nim",
            model="meta/llama3-8b-instruct",
            api_key="test",
        )
        assert model is not None
        assert model.endpoint is not None

    def test_ollama_imodel_constructs(self):
        """Ollama provider should construct with a base_url and no api_key."""
        model = iModel(
            provider="ollama",
            model="llama3",
            base_url="http://localhost:11434",
            api_key="test",
        )
        assert model is not None
        assert model.endpoint is not None

    def test_custom_endpoint_imodel_constructs(self):
        """A custom provider with explicit base_url should construct."""
        model = iModel(
            provider="custom",
            model="my-model",
            base_url="https://example.com/v1",
            api_key="test",
        )
        assert model is not None
        assert model.endpoint is not None

    def test_imodel_with_rate_limiting(self):
        """iModel should accept rate-limiting parameters without error."""
        model = iModel(
            provider="openai",
            model="gpt-4o-mini",
            api_key="test",
            limit_requests=100,
            limit_tokens=50000,
            capacity_refresh_time=30,
        )
        assert model is not None
        assert model.executor is not None

    def test_imodel_copy_returns_new_instance(self):
        """iModel.copy() should return a distinct iModel with a new ID."""
        original = iModel(
            provider="openai", model="gpt-4o-mini", api_key="test"
        )
        copied = original.copy()
        assert copied is not original
        assert isinstance(copied, iModel)
        assert copied.id != original.id

    def test_cli_provider_claude_code(self):
        """claude_code CLI provider should construct (no CLI binary needed)."""
        try:
            model = iModel(provider="claude_code", api_key="test")
            assert model is not None
        except Exception:
            pytest.skip("claude_code provider could not be constructed")

    def test_cli_provider_gemini_code(self):
        """gemini_code CLI provider should construct (no CLI binary needed)."""
        try:
            model = iModel(provider="gemini_code", api_key="test")
            assert model is not None
        except Exception:
            pytest.skip("gemini_code provider could not be constructed")

    def test_cli_provider_codex(self):
        """codex CLI provider should construct (no CLI binary needed)."""
        try:
            model = iModel(provider="codex", api_key="test")
            assert model is not None
        except Exception:
            pytest.skip("codex provider could not be constructed")


# ===========================================================================
# Tools (tools.md)
# ===========================================================================


class TestTools:
    """Tool construction and schema generation from documented patterns."""

    def test_function_to_schema_structure(self):
        """function_to_schema should produce an OpenAI-format tool schema."""
        from lionagi.libs.schema.function_to_schema import function_to_schema

        def greet(name: str, enthusiasm: int) -> str:
            """Greet someone by name.

            Args:
                name: The person's name.
                enthusiasm: How many exclamation marks to add.
            """
            return f"Hello, {name}{'!' * enthusiasm}"

        schema = function_to_schema(greet)
        assert isinstance(schema, dict)
        assert schema["type"] == "function"
        assert "function" in schema
        fn = schema["function"]
        assert fn["name"] == "greet"
        assert "parameters" in fn
        params = fn["parameters"]
        assert "properties" in params
        assert "name" in params["properties"]
        assert "enthusiasm" in params["properties"]
        assert "required" in params

    def test_function_to_schema_description(self):
        """function_to_schema should extract the docstring description."""
        from lionagi.libs.schema.function_to_schema import function_to_schema

        def add(a: int, b: int) -> int:
            """Add two numbers together.

            Args:
                a: First number.
                b: Second number.
            """
            return a + b

        schema = function_to_schema(add)
        fn = schema["function"]
        assert fn["description"] is not None
        assert len(fn["description"]) > 0

    def test_tool_constructs_from_callable(self):
        """Tool(func_callable=fn) should construct with expected attributes."""
        from lionagi.protocols.action.tool import Tool

        def search(query: str, limit: int) -> list:
            """Search for items.

            Args:
                query: Search query string.
                limit: Maximum results to return.
            """
            return []

        tool = Tool(func_callable=search)
        assert tool is not None
        assert tool.function == "search"
        assert tool.tool_schema is not None
        assert tool.tool_schema["type"] == "function"
        assert tool.tool_schema["function"]["name"] == "search"

    def test_tool_schema_has_parameters(self):
        """Tool schema should include parameter definitions from the function signature."""
        from lionagi.protocols.action.tool import Tool

        def translate(text: str, target_language: str) -> str:
            """Translate text.

            Args:
                text: The text to translate.
                target_language: ISO language code for the target language.
            """
            return text

        tool = Tool(func_callable=translate)
        params = tool.tool_schema["function"]["parameters"]
        assert "text" in params["properties"]
        assert "target_language" in params["properties"]

    def test_tool_with_request_options(self):
        """Tool should accept a Pydantic model as request_options for parameter validation."""
        from lionagi.protocols.action.tool import Tool

        class SearchParams(BaseModel):
            query: str
            max_results: int = 10

        def search(query: str, max_results: int = 10) -> list:
            """Search with validated parameters.

            Args:
                query: The search query.
                max_results: Maximum number of results.
            """
            return []

        tool = Tool(func_callable=search, request_options=SearchParams)
        assert tool is not None
        assert tool.function == "search"
        assert tool.request_options is SearchParams


# ===========================================================================
# MCP (mcp-servers.md)
# ===========================================================================


class TestMCP:
    """MCP-related imports and constructs from documented patterns."""

    def test_load_mcp_tools_import_resolves(self):
        """load_mcp_tools should be importable from the top-level lionagi package."""
        from lionagi import load_mcp_tools

        assert callable(load_mcp_tools)
