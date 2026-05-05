# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for provider endpoint constructors and create_payload.

All tests are pure construction/payload tests — no real network calls.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------


class TestProviderConfigConstants:
    def test_default_api_timeout_value(self):
        from lionagi.providers.config import DEFAULT_API_TIMEOUT

        assert DEFAULT_API_TIMEOUT == 600

    def test_default_agentic_timeout_value(self):
        from lionagi.providers.config import DEFAULT_AGENTIC_TIMEOUT

        assert DEFAULT_AGENTIC_TIMEOUT == 3600

    def test_agentic_timeout_greater_than_api_timeout(self):
        from lionagi.providers.config import (
            DEFAULT_AGENTIC_TIMEOUT,
            DEFAULT_API_TIMEOUT,
        )

        assert DEFAULT_AGENTIC_TIMEOUT > DEFAULT_API_TIMEOUT


# ---------------------------------------------------------------------------
# AnthropicMessagesEndpoint — constructor and create_payload
# ---------------------------------------------------------------------------


class TestAnthropicMessagesEndpoint:
    def test_endpoint_instantiation_with_dummy_key(self):
        from lionagi.providers.anthropic.messages.endpoint import (
            AnthropicMessagesEndpoint,
        )

        ep = AnthropicMessagesEndpoint(api_key="dummy-key-test")
        assert ep is not None
        assert ep.config.provider == "anthropic"

    def test_endpoint_config_has_anthropic_version_header(self):
        from lionagi.providers.anthropic.messages.endpoint import (
            AnthropicMessagesEndpoint,
        )

        ep = AnthropicMessagesEndpoint(api_key="dummy-key-test")
        assert "anthropic-version" in ep.config.default_headers
        assert ep.config.default_headers["anthropic-version"] == "2023-06-01"

    def test_endpoint_path_is_messages(self):
        from lionagi.providers.anthropic.messages.endpoint import (
            AnthropicMessagesEndpoint,
        )

        ep = AnthropicMessagesEndpoint(api_key="dummy-key-test")
        assert ep.config.endpoint == "messages"

    def test_create_payload_removes_api_key(self):
        from lionagi.providers.anthropic.messages.endpoint import (
            AnthropicMessagesEndpoint,
        )

        ep = AnthropicMessagesEndpoint(api_key="dummy-key-test")
        payload, headers = ep.create_payload(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "claude-3-5-haiku",
                "max_tokens": 100,
                "api_key": "should-be-stripped",
            }
        )
        assert "api_key" not in payload

    def test_create_payload_extracts_system_message(self):
        from lionagi.providers.anthropic.messages.endpoint import (
            AnthropicMessagesEndpoint,
        )

        ep = AnthropicMessagesEndpoint(api_key="dummy-key-test")
        payload, _headers = ep.create_payload(
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                ],
                "model": "claude-3-5-haiku",
                "max_tokens": 50,
            }
        )
        # System message extracted to top-level key
        assert "system" in payload
        # The system message list contains a text block
        assert payload["system"][0]["text"] == "You are helpful."
        # Only the user message remains in messages
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

    def test_create_payload_includes_anthropic_version_header(self):
        from lionagi.providers.anthropic.messages.endpoint import (
            AnthropicMessagesEndpoint,
        )

        ep = AnthropicMessagesEndpoint(api_key="dummy-key-test")
        _payload, headers = ep.create_payload(
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "model": "claude-3-5-haiku",
                "max_tokens": 10,
            }
        )
        assert headers.get("anthropic-version") == "2023-06-01"

    def test_create_payload_no_system_message_passthrough(self):
        from lionagi.providers.anthropic.messages.endpoint import (
            AnthropicMessagesEndpoint,
        )

        ep = AnthropicMessagesEndpoint(api_key="dummy-key-test")
        payload, _headers = ep.create_payload(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "claude-3-5-haiku",
                "max_tokens": 20,
            }
        )
        # No system key when there is no system message
        assert "system" not in payload
        assert len(payload["messages"]) == 1


# ---------------------------------------------------------------------------
# ExaSearchEndpoint — constructor and config
# ---------------------------------------------------------------------------


class TestExaSearchEndpoint:
    def test_endpoint_instantiation_with_dummy_key(self):
        from lionagi.providers.exa.search.endpoint import ExaSearchEndpoint

        ep = ExaSearchEndpoint(api_key="dummy-key-test")
        assert ep is not None
        assert ep.config.provider == "exa"

    def test_endpoint_path_is_search(self):
        from lionagi.providers.exa.search.endpoint import ExaSearchEndpoint

        ep = ExaSearchEndpoint(api_key="dummy-key-test")
        assert ep.config.endpoint == "search"

    def test_endpoint_base_url(self):
        from lionagi.providers.exa.search.endpoint import ExaSearchEndpoint

        ep = ExaSearchEndpoint(api_key="dummy-key-test")
        assert "exa.ai" in ep.config.base_url

    def test_endpoint_has_request_options(self):
        from lionagi.providers.exa.search.endpoint import ExaSearchEndpoint

        ep = ExaSearchEndpoint(api_key="dummy-key-test")
        # ExaSearchEndpoint has request_options set from the config
        assert ep.config.request_options is not None

    def test_create_payload_with_query(self):
        from lionagi.providers.exa.search.endpoint import ExaSearchEndpoint

        ep = ExaSearchEndpoint(api_key="dummy-key-test")
        payload, headers = ep.create_payload({"query": "lionagi framework"})
        assert "query" in payload
        assert payload["query"] == "lionagi framework"

    def test_create_payload_headers_include_auth(self):
        from lionagi.providers.exa.search.endpoint import ExaSearchEndpoint

        ep = ExaSearchEndpoint(api_key="dummy-key-test")
        _payload, headers = ep.create_payload({"query": "test"})
        # Exa uses x-api-key auth
        assert "x-api-key" in headers


# ---------------------------------------------------------------------------
# TavilySearchEndpoint — constructor and config
# ---------------------------------------------------------------------------


class TestTavilySearchEndpoint:
    def test_endpoint_instantiation_with_dummy_key(self):
        from lionagi.providers.tavily.search.endpoint import TavilySearchEndpoint

        ep = TavilySearchEndpoint(api_key="dummy-key-test")
        assert ep is not None
        assert ep.config.provider == "tavily"

    def test_endpoint_path_is_search(self):
        from lionagi.providers.tavily.search.endpoint import TavilySearchEndpoint

        ep = TavilySearchEndpoint(api_key="dummy-key-test")
        assert ep.config.endpoint == "search"

    def test_endpoint_base_url(self):
        from lionagi.providers.tavily.search.endpoint import TavilySearchEndpoint

        ep = TavilySearchEndpoint(api_key="dummy-key-test")
        assert "tavily.com" in ep.config.base_url

    def test_endpoint_method_is_post(self):
        from lionagi.providers.tavily.search.endpoint import TavilySearchEndpoint

        ep = TavilySearchEndpoint(api_key="dummy-key-test")
        assert ep.config.method == "POST"

    def test_endpoint_timeout_set(self):
        from lionagi.providers.tavily.search.endpoint import TavilySearchEndpoint

        ep = TavilySearchEndpoint(api_key="dummy-key-test")
        assert ep.config.timeout == 120

    def test_create_payload_with_query(self):
        from lionagi.providers.tavily.search.endpoint import TavilySearchEndpoint

        ep = TavilySearchEndpoint(api_key="dummy-key-test")
        payload, headers = ep.create_payload({"query": "latest AI news"})
        assert "query" in payload
        assert payload["query"] == "latest AI news"


# ---------------------------------------------------------------------------
# OllamaGenerateEndpoint — skipped when ollama is not installed
# ---------------------------------------------------------------------------


class TestOllamaGenerateEndpoint:
    @pytest.fixture(autouse=True)
    def mock_ollama_installed(self):
        """Inject a mock ollama module so the endpoint can be instantiated."""
        mock_ollama = MagicMock()
        mock_ollama.__spec__ = MagicMock()
        sys.modules.setdefault("ollama", mock_ollama)
        # Patch _HAS_OLLAMA on the generate module if already imported
        try:
            import lionagi.providers.ollama.generate.endpoint as gen_mod

            original = gen_mod._HAS_OLLAMA
            gen_mod._HAS_OLLAMA = True
            yield
            gen_mod._HAS_OLLAMA = original
        except ImportError:
            yield

    def test_endpoint_raises_without_ollama(self):
        import lionagi.providers.ollama.generate.endpoint as gen_mod
        from lionagi.providers.ollama.generate.endpoint import OllamaGenerateEndpoint

        original = gen_mod._HAS_OLLAMA
        gen_mod._HAS_OLLAMA = False
        try:
            with pytest.raises(ModuleNotFoundError, match="ollama is not installed"):
                OllamaGenerateEndpoint()
        finally:
            gen_mod._HAS_OLLAMA = original

    def test_endpoint_instantiation_with_mock_ollama(self):
        from lionagi.providers.ollama.generate.endpoint import OllamaGenerateEndpoint

        ep = OllamaGenerateEndpoint()
        assert ep is not None
        assert ep.config.provider == "ollama"

    def test_endpoint_path_is_generate(self):
        from lionagi.providers.ollama.generate.endpoint import OllamaGenerateEndpoint

        ep = OllamaGenerateEndpoint()
        assert ep.config.endpoint == "generate"

    def test_endpoint_base_url_is_local(self):
        from lionagi.providers.ollama.generate.endpoint import OllamaGenerateEndpoint

        ep = OllamaGenerateEndpoint()
        assert "localhost" in ep.config.base_url or "11434" in ep.config.base_url

    def test_create_payload_removes_unsupported_params(self):
        from lionagi.providers.ollama.generate.endpoint import OllamaGenerateEndpoint

        ep = OllamaGenerateEndpoint()
        request = {
            "model": "llama3.2",
            "prompt": "Why is the sky blue?",
            "reasoning_effort": "high",
            "stream_options": {"include_usage": True},
        }
        payload, _headers = ep.create_payload(request)
        assert "reasoning_effort" not in payload
        assert "stream_options" not in payload

    def test_create_payload_preserves_model_and_prompt(self):
        from lionagi.providers.ollama.generate.endpoint import OllamaGenerateEndpoint

        ep = OllamaGenerateEndpoint()
        payload, _headers = ep.create_payload(
            {"model": "llama3.2", "prompt": "Hello world"}
        )
        assert payload.get("model") == "llama3.2"
        assert payload.get("prompt") == "Hello world"


# ---------------------------------------------------------------------------
# ExaContentsEndpoint — constructor and create_payload
# ---------------------------------------------------------------------------


class TestExaContentsEndpoint:
    def test_endpoint_instantiation_with_dummy_key(self):
        from lionagi.providers.exa.contents.endpoint import ExaContentsEndpoint

        ep = ExaContentsEndpoint(api_key="dummy-key-test")
        assert ep is not None
        assert ep.config.provider == "exa"

    def test_endpoint_path_is_contents(self):
        from lionagi.providers.exa.contents.endpoint import ExaContentsEndpoint

        ep = ExaContentsEndpoint(api_key="dummy-key-test")
        assert ep.config.endpoint == "contents"

    def test_endpoint_auth_type_is_x_api_key(self):
        from lionagi.providers.exa.contents.endpoint import ExaContentsEndpoint

        ep = ExaContentsEndpoint(api_key="dummy-key-test")
        assert ep.config.auth_type == "x-api-key"

    def test_endpoint_base_url_contains_exa(self):
        from lionagi.providers.exa.contents.endpoint import ExaContentsEndpoint

        ep = ExaContentsEndpoint(api_key="dummy-key-test")
        assert "exa.ai" in ep.config.base_url

    def test_create_payload_with_ids(self):
        from lionagi.providers.exa.contents.endpoint import ExaContentsEndpoint

        ep = ExaContentsEndpoint(api_key="dummy-key-test")
        payload, headers = ep.create_payload({"ids": ["https://example.com"]})
        assert "ids" in payload
        assert payload["ids"] == ["https://example.com"]

    def test_create_payload_headers_include_auth(self):
        from lionagi.providers.exa.contents.endpoint import ExaContentsEndpoint

        ep = ExaContentsEndpoint(api_key="dummy-key-test")
        _payload, headers = ep.create_payload({"ids": ["https://example.com"]})
        assert "x-api-key" in headers

    def test_create_payload_with_extra_headers(self):
        from lionagi.providers.exa.contents.endpoint import ExaContentsEndpoint

        ep = ExaContentsEndpoint(api_key="dummy-key-test")
        _payload, headers = ep.create_payload(
            {"ids": ["https://example.com"]},
            extra_headers={"X-Custom": "my-value"},
        )
        assert headers.get("X-Custom") == "my-value"


# ---------------------------------------------------------------------------
# ExaFindSimilarEndpoint — constructor and create_payload
# ---------------------------------------------------------------------------


class TestExaFindSimilarEndpoint:
    def test_endpoint_instantiation_with_dummy_key(self):
        from lionagi.providers.exa.find_similar.endpoint import ExaFindSimilarEndpoint

        ep = ExaFindSimilarEndpoint(api_key="dummy-key-test")
        assert ep is not None
        assert ep.config.provider == "exa"

    def test_endpoint_path_is_find_similar(self):
        from lionagi.providers.exa.find_similar.endpoint import ExaFindSimilarEndpoint

        ep = ExaFindSimilarEndpoint(api_key="dummy-key-test")
        assert ep.config.endpoint == "findSimilar"

    def test_endpoint_auth_type_is_x_api_key(self):
        from lionagi.providers.exa.find_similar.endpoint import ExaFindSimilarEndpoint

        ep = ExaFindSimilarEndpoint(api_key="dummy-key-test")
        assert ep.config.auth_type == "x-api-key"

    def test_create_payload_with_url(self):
        from lionagi.providers.exa.find_similar.endpoint import ExaFindSimilarEndpoint

        ep = ExaFindSimilarEndpoint(api_key="dummy-key-test")
        payload, headers = ep.create_payload(
            {"url": "https://arxiv.org/abs/2303.08774"}
        )
        assert "url" in payload
        assert "arxiv.org" in payload["url"]

    def test_create_payload_headers_include_auth(self):
        from lionagi.providers.exa.find_similar.endpoint import ExaFindSimilarEndpoint

        ep = ExaFindSimilarEndpoint(api_key="dummy-key-test")
        _payload, headers = ep.create_payload({"url": "https://example.com"})
        assert "x-api-key" in headers

    def test_create_payload_with_extra_headers(self):
        from lionagi.providers.exa.find_similar.endpoint import ExaFindSimilarEndpoint

        ep = ExaFindSimilarEndpoint(api_key="dummy-key-test")
        _payload, headers = ep.create_payload(
            {"url": "https://example.com"},
            extra_headers={"X-Trace-Id": "abc123"},
        )
        assert headers.get("X-Trace-Id") == "abc123"


# ---------------------------------------------------------------------------
# ExaSearchEndpoint — extra_headers path
# ---------------------------------------------------------------------------


class TestExaSearchEndpointExtraHeaders:
    def test_create_payload_with_extra_headers_merged(self):
        from lionagi.providers.exa.search.endpoint import ExaSearchEndpoint

        ep = ExaSearchEndpoint(api_key="dummy-key-test")
        _payload, headers = ep.create_payload(
            {"query": "test query"},
            extra_headers={"X-Request-Id": "req-001"},
        )
        assert headers.get("X-Request-Id") == "req-001"
        # Original auth key still present
        assert "x-api-key" in headers


# ---------------------------------------------------------------------------
# FirecrawlScrapeEndpoint — constructor and create_payload
# ---------------------------------------------------------------------------


class TestFirecrawlScrapeEndpoint:
    def test_endpoint_instantiation_with_dummy_key(self):
        from lionagi.providers.firecrawl.scrape.endpoint import FirecrawlScrapeEndpoint

        ep = FirecrawlScrapeEndpoint(api_key="dummy-key-test")
        assert ep is not None
        assert ep.config.provider == "firecrawl"

    def test_endpoint_path_is_scrape(self):
        from lionagi.providers.firecrawl.scrape.endpoint import FirecrawlScrapeEndpoint

        ep = FirecrawlScrapeEndpoint(api_key="dummy-key-test")
        assert "scrape" in ep.config.endpoint

    def test_endpoint_auth_type_is_bearer(self):
        from lionagi.providers.firecrawl.scrape.endpoint import FirecrawlScrapeEndpoint

        ep = FirecrawlScrapeEndpoint(api_key="dummy-key-test")
        assert ep.config.auth_type == "bearer"

    def test_endpoint_base_url_contains_firecrawl(self):
        from lionagi.providers.firecrawl.scrape.endpoint import FirecrawlScrapeEndpoint

        ep = FirecrawlScrapeEndpoint(api_key="dummy-key-test")
        assert "firecrawl.dev" in ep.config.base_url

    def test_create_payload_with_url(self):
        from lionagi.providers.firecrawl.scrape.endpoint import FirecrawlScrapeEndpoint

        ep = FirecrawlScrapeEndpoint(api_key="dummy-key-test")
        payload, headers = ep.create_payload({"url": "https://docs.example.com"})
        assert payload.get("url") == "https://docs.example.com"

    def test_create_payload_headers_include_bearer(self):
        from lionagi.providers.firecrawl.scrape.endpoint import FirecrawlScrapeEndpoint

        ep = FirecrawlScrapeEndpoint(api_key="dummy-key-test")
        _payload, headers = ep.create_payload({"url": "https://docs.example.com"})
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")

    def test_create_payload_with_extra_headers(self):
        from lionagi.providers.firecrawl.scrape.endpoint import FirecrawlScrapeEndpoint

        ep = FirecrawlScrapeEndpoint(api_key="dummy-key-test")
        _payload, headers = ep.create_payload(
            {"url": "https://docs.example.com"},
            extra_headers={"X-Custom": "value"},
        )
        assert headers.get("X-Custom") == "value"


# ---------------------------------------------------------------------------
# FirecrawlCrawlEndpoint — constructor and create_payload
# ---------------------------------------------------------------------------


class TestFirecrawlCrawlEndpoint:
    def test_endpoint_instantiation_with_dummy_key(self):
        from lionagi.providers.firecrawl.crawl.endpoint import FirecrawlCrawlEndpoint

        ep = FirecrawlCrawlEndpoint(api_key="dummy-key-test")
        assert ep is not None
        assert ep.config.provider == "firecrawl"

    def test_endpoint_path_is_crawl(self):
        from lionagi.providers.firecrawl.crawl.endpoint import FirecrawlCrawlEndpoint

        ep = FirecrawlCrawlEndpoint(api_key="dummy-key-test")
        assert "crawl" in ep.config.endpoint

    def test_endpoint_auth_type_is_bearer(self):
        from lionagi.providers.firecrawl.crawl.endpoint import FirecrawlCrawlEndpoint

        ep = FirecrawlCrawlEndpoint(api_key="dummy-key-test")
        assert ep.config.auth_type == "bearer"

    def test_create_payload_with_url(self):
        from lionagi.providers.firecrawl.crawl.endpoint import FirecrawlCrawlEndpoint

        ep = FirecrawlCrawlEndpoint(api_key="dummy-key-test")
        payload, headers = ep.create_payload({"url": "https://docs.example.com"})
        assert payload.get("url") == "https://docs.example.com"

    def test_create_payload_headers_include_bearer(self):
        from lionagi.providers.firecrawl.crawl.endpoint import FirecrawlCrawlEndpoint

        ep = FirecrawlCrawlEndpoint(api_key="dummy-key-test")
        _payload, headers = ep.create_payload({"url": "https://docs.example.com"})
        assert "Authorization" in headers


# ---------------------------------------------------------------------------
# FirecrawlMapEndpoint — constructor and create_payload
# ---------------------------------------------------------------------------


class TestFirecrawlMapEndpoint:
    def test_endpoint_instantiation_with_dummy_key(self):
        from lionagi.providers.firecrawl.map.endpoint import FirecrawlMapEndpoint

        ep = FirecrawlMapEndpoint(api_key="dummy-key-test")
        assert ep is not None
        assert ep.config.provider == "firecrawl"

    def test_endpoint_path_is_map(self):
        from lionagi.providers.firecrawl.map.endpoint import FirecrawlMapEndpoint

        ep = FirecrawlMapEndpoint(api_key="dummy-key-test")
        assert "map" in ep.config.endpoint

    def test_endpoint_auth_type_is_bearer(self):
        from lionagi.providers.firecrawl.map.endpoint import FirecrawlMapEndpoint

        ep = FirecrawlMapEndpoint(api_key="dummy-key-test")
        assert ep.config.auth_type == "bearer"

    def test_create_payload_with_url(self):
        from lionagi.providers.firecrawl.map.endpoint import FirecrawlMapEndpoint

        ep = FirecrawlMapEndpoint(api_key="dummy-key-test")
        payload, headers = ep.create_payload({"url": "https://docs.example.com"})
        assert payload.get("url") == "https://docs.example.com"

    def test_create_payload_headers_include_bearer(self):
        from lionagi.providers.firecrawl.map.endpoint import FirecrawlMapEndpoint

        ep = FirecrawlMapEndpoint(api_key="dummy-key-test")
        _payload, headers = ep.create_payload({"url": "https://docs.example.com"})
        assert "Authorization" in headers


# ---------------------------------------------------------------------------
# OpenaiImageGenerationEndpoint — constructor and create_payload
# ---------------------------------------------------------------------------


class TestOpenaiImageGenerationEndpoint:
    def test_endpoint_instantiation_with_dummy_key(self):
        from lionagi.providers.openai.images.endpoint import (
            OpenaiImageGenerationEndpoint,
        )

        ep = OpenaiImageGenerationEndpoint(api_key="dummy-key-test")
        assert ep is not None
        assert ep.config.provider == "openai"

    def test_endpoint_full_url_contains_openai(self):
        from lionagi.providers.openai.images.endpoint import (
            OpenaiImageGenerationEndpoint,
        )

        ep = OpenaiImageGenerationEndpoint(api_key="dummy-key-test")
        assert "openai" in ep.config.full_url

    def test_endpoint_path_is_images_generations(self):
        from lionagi.providers.openai.images.endpoint import (
            OpenaiImageGenerationEndpoint,
        )

        ep = OpenaiImageGenerationEndpoint(api_key="dummy-key-test")
        assert "images/generations" in ep.config.endpoint

    def test_create_payload_with_prompt(self):
        from lionagi.providers.openai.images.endpoint import (
            OpenaiImageGenerationEndpoint,
        )

        ep = OpenaiImageGenerationEndpoint(api_key="dummy-key-test")
        payload, headers = ep.create_payload({"prompt": "A sunset over the ocean"})
        assert payload.get("prompt") == "A sunset over the ocean"
        assert "Authorization" in headers


# ---------------------------------------------------------------------------
# OpenaiImageEditEndpoint — constructor, transport_arg_keys, mock _call
# ---------------------------------------------------------------------------


class TestOpenaiImageEditEndpoint:
    def test_endpoint_instantiation_with_dummy_key(self):
        from lionagi.providers.openai.images.endpoint import OpenaiImageEditEndpoint

        ep = OpenaiImageEditEndpoint(api_key="dummy-key-test")
        assert ep is not None
        assert ep.config.provider == "openai"

    def test_transport_arg_keys_attribute(self):
        from lionagi.providers.openai.images.endpoint import OpenaiImageEditEndpoint

        ep = OpenaiImageEditEndpoint(api_key="dummy-key-test")
        assert "image" in ep.transport_arg_keys
        assert "image_filename" in ep.transport_arg_keys
        assert "mask" in ep.transport_arg_keys
        assert "mask_filename" in ep.transport_arg_keys

    def test_endpoint_full_url_contains_openai(self):
        from lionagi.providers.openai.images.endpoint import OpenaiImageEditEndpoint

        ep = OpenaiImageEditEndpoint(api_key="dummy-key-test")
        assert "openai" in ep.config.full_url

    @pytest.mark.asyncio
    async def test_call_returns_json_on_200(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from lionagi.providers.openai.images.endpoint import OpenaiImageEditEndpoint

        ep = OpenaiImageEditEndpoint(api_key="dummy-key-test")

        fake_response = AsyncMock()
        fake_response.status = 200
        fake_response.json = AsyncMock(
            return_value={"data": [{"url": "https://fake.img"}]}
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=fake_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(ep, "_create_http_session", return_value=mock_session_ctx):
            result = await ep._call(
                {"prompt": "Add a rainbow"},
                {"Authorization": "Bearer dummy"},
                image=b"PNG bytes",
                image_filename="image.png",
            )
        assert result == {"data": [{"url": "https://fake.img"}]}

    @pytest.mark.asyncio
    async def test_call_raises_on_non_200(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        import aiohttp

        from lionagi.providers.openai.images.endpoint import OpenaiImageEditEndpoint

        ep = OpenaiImageEditEndpoint(api_key="dummy-key-test")

        fake_response = AsyncMock()
        fake_response.status = 400
        fake_response.text = AsyncMock(return_value="bad request")
        fake_response.request_info = MagicMock()
        fake_response.history = ()
        fake_response.headers = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=fake_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(ep, "_create_http_session", return_value=mock_session_ctx):
            with pytest.raises(aiohttp.ClientResponseError):
                await ep._call(
                    {"prompt": "Add a rainbow"},
                    {"Authorization": "Bearer dummy"},
                    image=b"PNG bytes",
                )


# ---------------------------------------------------------------------------
# OpenaiAudioSpeechEndpoint — constructor, create_payload, mock _call
# ---------------------------------------------------------------------------


class TestOpenaiAudioSpeechEndpoint:
    def test_endpoint_instantiation_with_dummy_key(self):
        from lionagi.providers.openai.audio.endpoint import OpenaiAudioSpeechEndpoint

        ep = OpenaiAudioSpeechEndpoint(api_key="dummy-key-test")
        assert ep is not None
        assert ep.config.provider == "openai"

    def test_endpoint_path_is_audio_speech(self):
        from lionagi.providers.openai.audio.endpoint import OpenaiAudioSpeechEndpoint

        ep = OpenaiAudioSpeechEndpoint(api_key="dummy-key-test")
        assert "speech" in ep.config.endpoint

    def test_create_payload_returns_tuple(self):
        from lionagi.providers.openai.audio.endpoint import OpenaiAudioSpeechEndpoint

        ep = OpenaiAudioSpeechEndpoint(api_key="dummy-key-test")
        payload, headers = ep.create_payload(
            {"input": "Hello world", "voice": "nova", "model": "tts-1"}
        )
        assert isinstance(payload, dict)
        assert isinstance(headers, dict)

    def test_create_payload_includes_bearer_auth(self):
        from lionagi.providers.openai.audio.endpoint import OpenaiAudioSpeechEndpoint

        ep = OpenaiAudioSpeechEndpoint(api_key="dummy-key-test")
        _payload, headers = ep.create_payload(
            {"input": "Hello!", "voice": "nova", "model": "tts-1"}
        )
        assert "Authorization" in headers

    @pytest.mark.asyncio
    async def test_call_returns_bytes_on_200(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from lionagi.providers.openai.audio.endpoint import OpenaiAudioSpeechEndpoint

        ep = OpenaiAudioSpeechEndpoint(api_key="dummy-key-test")
        audio_bytes = b"\xff\xfb\x90\x00"

        fake_response = AsyncMock()
        fake_response.status = 200
        fake_response.read = AsyncMock(return_value=audio_bytes)

        mock_session = MagicMock()
        mock_session.request = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=fake_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(ep, "_create_http_session", return_value=mock_session_ctx):
            result = await ep._call(
                {"input": "Hello!", "voice": "nova", "model": "tts-1"},
                {"Authorization": "Bearer dummy"},
            )
        assert result == audio_bytes

    @pytest.mark.asyncio
    async def test_call_raises_on_non_200(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        import aiohttp

        from lionagi.providers.openai.audio.endpoint import OpenaiAudioSpeechEndpoint

        ep = OpenaiAudioSpeechEndpoint(api_key="dummy-key-test")

        fake_response = AsyncMock()
        fake_response.status = 503
        fake_response.text = AsyncMock(return_value="service unavailable")
        fake_response.request_info = MagicMock()
        fake_response.history = ()
        fake_response.headers = {}

        mock_session = MagicMock()
        mock_session.request = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=fake_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(ep, "_create_http_session", return_value=mock_session_ctx):
            with pytest.raises(aiohttp.ClientResponseError):
                await ep._call(
                    {"input": "Hello!", "voice": "nova", "model": "tts-1"},
                    {"Authorization": "Bearer dummy"},
                )


# ---------------------------------------------------------------------------
# OpenaiAudioTranscriptionEndpoint — constructor, transport_arg_keys, mock _call
# ---------------------------------------------------------------------------


class TestOpenaiAudioTranscriptionEndpoint:
    def test_endpoint_instantiation_with_dummy_key(self):
        from lionagi.providers.openai.audio.endpoint import (
            OpenaiAudioTranscriptionEndpoint,
        )

        ep = OpenaiAudioTranscriptionEndpoint(api_key="dummy-key-test")
        assert ep is not None
        assert ep.config.provider == "openai"

    def test_transport_arg_keys_attribute(self):
        from lionagi.providers.openai.audio.endpoint import (
            OpenaiAudioTranscriptionEndpoint,
        )

        ep = OpenaiAudioTranscriptionEndpoint(api_key="dummy-key-test")
        assert "file" in ep.transport_arg_keys
        assert "filename" in ep.transport_arg_keys

    def test_endpoint_path_is_audio_transcriptions(self):
        from lionagi.providers.openai.audio.endpoint import (
            OpenaiAudioTranscriptionEndpoint,
        )

        ep = OpenaiAudioTranscriptionEndpoint(api_key="dummy-key-test")
        assert "transcription" in ep.config.endpoint

    def test_create_payload_returns_tuple(self):
        from lionagi.providers.openai.audio.endpoint import (
            OpenaiAudioTranscriptionEndpoint,
        )

        ep = OpenaiAudioTranscriptionEndpoint(api_key="dummy-key-test")
        payload, headers = ep.create_payload({"model": "whisper-1"})
        assert isinstance(payload, dict)
        assert isinstance(headers, dict)

    @pytest.mark.asyncio
    async def test_call_returns_json_on_200(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from lionagi.providers.openai.audio.endpoint import (
            OpenaiAudioTranscriptionEndpoint,
        )

        ep = OpenaiAudioTranscriptionEndpoint(api_key="dummy-key-test")
        transcription = {"text": "Hello world"}

        fake_response = AsyncMock()
        fake_response.status = 200
        fake_response.json = AsyncMock(return_value=transcription)

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=fake_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(ep, "_create_http_session", return_value=mock_session_ctx):
            result = await ep._call(
                {"model": "whisper-1"},
                {"Authorization": "Bearer dummy"},
                file=b"audio data",
                filename="audio.mp3",
            )
        assert result == transcription

    @pytest.mark.asyncio
    async def test_call_raises_on_non_200(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        import aiohttp

        from lionagi.providers.openai.audio.endpoint import (
            OpenaiAudioTranscriptionEndpoint,
        )

        ep = OpenaiAudioTranscriptionEndpoint(api_key="dummy-key-test")

        fake_response = AsyncMock()
        fake_response.status = 422
        fake_response.text = AsyncMock(return_value="unprocessable entity")
        fake_response.request_info = MagicMock()
        fake_response.history = ()
        fake_response.headers = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=fake_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(ep, "_create_http_session", return_value=mock_session_ctx):
            with pytest.raises(aiohttp.ClientResponseError):
                await ep._call(
                    {"model": "whisper-1"},
                    {"Authorization": "Bearer dummy"},
                    file=b"audio data",
                )


# ---------------------------------------------------------------------------
# GroqAudioTranscriptionEndpoint — constructor, transport_arg_keys, mock _call
# ---------------------------------------------------------------------------


class TestGroqAudioTranscriptionEndpoint:
    def test_endpoint_instantiation_with_dummy_key(self):
        from lionagi.providers.groq.audio_transcription.endpoint import (
            GroqAudioTranscriptionEndpoint,
        )

        ep = GroqAudioTranscriptionEndpoint(api_key="dummy-key-test")
        assert ep is not None
        assert ep.config.provider == "groq"

    def test_transport_arg_keys_attribute(self):
        from lionagi.providers.groq.audio_transcription.endpoint import (
            GroqAudioTranscriptionEndpoint,
        )

        ep = GroqAudioTranscriptionEndpoint(api_key="dummy-key-test")
        assert "file" in ep.transport_arg_keys
        assert "filename" in ep.transport_arg_keys

    def test_endpoint_path_is_audio_transcriptions(self):
        from lionagi.providers.groq.audio_transcription.endpoint import (
            GroqAudioTranscriptionEndpoint,
        )

        ep = GroqAudioTranscriptionEndpoint(api_key="dummy-key-test")
        assert "transcription" in ep.config.endpoint

    def test_endpoint_base_url_contains_groq(self):
        from lionagi.providers.groq.audio_transcription.endpoint import (
            GroqAudioTranscriptionEndpoint,
        )

        ep = GroqAudioTranscriptionEndpoint(api_key="dummy-key-test")
        assert "groq.com" in ep.config.base_url

    @pytest.mark.asyncio
    async def test_call_returns_json_on_200(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from lionagi.providers.groq.audio_transcription.endpoint import (
            GroqAudioTranscriptionEndpoint,
        )

        ep = GroqAudioTranscriptionEndpoint(api_key="dummy-key-test")
        transcription = {"text": "Hello from Groq"}

        fake_response = AsyncMock()
        fake_response.status = 200
        fake_response.json = AsyncMock(return_value=transcription)

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=fake_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(ep, "_create_http_session", return_value=mock_session_ctx):
            result = await ep._call(
                {"model": "whisper-large-v3"},
                {"Authorization": "Bearer dummy"},
                file=b"audio data",
                filename="audio.mp3",
            )
        assert result == transcription

    @pytest.mark.asyncio
    async def test_call_raises_on_non_200(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        import aiohttp

        from lionagi.providers.groq.audio_transcription.endpoint import (
            GroqAudioTranscriptionEndpoint,
        )

        ep = GroqAudioTranscriptionEndpoint(api_key="dummy-key-test")

        fake_response = AsyncMock()
        fake_response.status = 429
        fake_response.text = AsyncMock(return_value="rate limited")
        fake_response.request_info = MagicMock()
        fake_response.history = ()
        fake_response.headers = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=fake_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(ep, "_create_http_session", return_value=mock_session_ctx):
            with pytest.raises(aiohttp.ClientResponseError):
                await ep._call(
                    {"model": "whisper-large-v3"},
                    {"Authorization": "Bearer dummy"},
                    file=b"audio data",
                )


# ---------------------------------------------------------------------------
# Tavily endpoints — lines 17-25, 31-39 (config=None path)
# ---------------------------------------------------------------------------


class TestTavilySearchEndpoint:
    def test_instantiation_with_api_key(self):
        from lionagi.providers.tavily.search.endpoint import TavilySearchEndpoint

        ep = TavilySearchEndpoint(api_key="dummy-tavily-key")
        assert ep is not None

    def test_instantiation_sets_defaults(self):
        from lionagi.providers.tavily.search.endpoint import TavilySearchEndpoint

        ep = TavilySearchEndpoint(api_key="dummy-tavily-key")
        assert ep.config is not None

    def test_instantiation_with_no_env_key_uses_fallback(self):
        from unittest.mock import patch

        with patch("lionagi.config.settings") as mock_settings:
            mock_settings.TAVILY_API_KEY = None
            from lionagi.providers.tavily.search.endpoint import TavilySearchEndpoint

            ep = TavilySearchEndpoint(api_key="fallback-key")
            assert ep is not None

    def test_create_payload_contains_query(self):
        from lionagi.providers.tavily.search.endpoint import TavilySearchEndpoint

        ep = TavilySearchEndpoint(api_key="dummy-tavily-key")
        payload, headers = ep.create_payload({"query": "test search"})
        assert "query" in payload

    def test_config_none_path_sets_timeout(self):
        from lionagi.providers.tavily.search.endpoint import TavilySearchEndpoint

        ep = TavilySearchEndpoint(api_key="test-key")
        # timeout defaults to 120 via the config=None path
        assert ep.config.timeout == 120

    def test_config_none_path_sets_max_retries(self):
        from lionagi.providers.tavily.search.endpoint import TavilySearchEndpoint

        ep = TavilySearchEndpoint(api_key="test-key")
        assert ep.config.max_retries == 3


class TestTavilyExtractEndpoint:
    def test_instantiation_with_api_key(self):
        from lionagi.providers.tavily.search.endpoint import TavilyExtractEndpoint

        ep = TavilyExtractEndpoint(api_key="dummy-tavily-key")
        assert ep is not None

    def test_instantiation_sets_defaults(self):
        from lionagi.providers.tavily.search.endpoint import TavilyExtractEndpoint

        ep = TavilyExtractEndpoint(api_key="dummy-tavily-key")
        assert ep.config is not None

    def test_config_none_path_sets_timeout(self):
        from lionagi.providers.tavily.search.endpoint import TavilyExtractEndpoint

        ep = TavilyExtractEndpoint(api_key="test-key")
        assert ep.config.timeout == 120

    def test_config_none_path_sets_max_retries(self):
        from lionagi.providers.tavily.search.endpoint import TavilyExtractEndpoint

        ep = TavilyExtractEndpoint(api_key="test-key")
        assert ep.config.max_retries == 3

    def test_create_payload_contains_urls(self):
        from lionagi.providers.tavily.search.endpoint import TavilyExtractEndpoint

        ep = TavilyExtractEndpoint(api_key="dummy-tavily-key")
        payload, headers = ep.create_payload({"urls": ["https://example.com"]})
        assert "urls" in payload
