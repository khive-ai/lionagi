# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Claude Code v1 integration."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from lionagi.services.adapters.claude_code_adapter import ClaudeCodeAdapter
from lionagi.services.core import CallContext
from lionagi.services.imodel import iModel
from lionagi.services.providers.claude_code import (
    ClaudeCodeCLIService,
    ClaudeCodeRequestModel,
    create_claude_code_service,
)
from lionagi.services.providers.provider_registry import get_provider_registry, register_builtin_adapters


class TestClaudeCodeRequestModel:
    """Test Claude Code request model."""

    def test_basic_creation(self):
        """Test basic request model creation."""
        req = ClaudeCodeRequestModel(prompt="Hello Claude")
        assert req.prompt == "Hello Claude"
        assert req.get_prompt() == "Hello Claude"

    def test_messages_to_prompt_conversion(self):
        """Test conversion from messages to prompt."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        req = ClaudeCodeRequestModel(messages=messages)
        prompt = req.get_prompt()

        # Should combine non-system messages
        assert "Hello" in prompt
        assert "Hi there!" in prompt
        assert "How are you?" in prompt
        assert "You are a helpful assistant" not in prompt

    def test_continuation_mode(self):
        """Test continuation mode uses last message."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        req = ClaudeCodeRequestModel(messages=messages, continue_conversation=True)
        assert req.get_prompt() == "How are you?"

    def test_system_prompt_extraction(self):
        """Test system prompt extraction."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]

        req = ClaudeCodeRequestModel(messages=messages)
        assert req.get_system_prompt() == "You are a helpful assistant"

    def test_workspace_path_security(self):
        """Test workspace path security validation."""
        base_repo = Path("/tmp/test-repo")

        # Valid relative path
        req = ClaudeCodeRequestModel(ws="subdir", repo=str(base_repo))
        workspace = req.get_workspace_path(base_repo)
        # Handle macOS symlink resolution (/tmp -> /private/tmp)
        expected = (base_repo / "subdir").resolve()
        assert workspace.resolve() == expected

        # Invalid absolute path
        req = ClaudeCodeRequestModel(ws="/etc/passwd", repo=str(base_repo))
        with pytest.raises(ValueError, match="relative"):
            req.get_workspace_path(base_repo)

        # Invalid directory traversal
        req = ClaudeCodeRequestModel(ws="../../../etc", repo=str(base_repo))
        with pytest.raises(ValueError, match="traversal"):
            req.get_workspace_path(base_repo)

    def test_cli_args_generation(self):
        """Test CLI arguments generation."""
        req = ClaudeCodeRequestModel(
            prompt="Test prompt",
            model="sonnet",
            allowed_tools=["Read", "Write"],
            max_turns=3,
            permission_mode="bypassPermissions",
        )

        args = req.as_cli_args()

        assert "-p" in args
        assert "Test prompt" in args
        assert "--output-format" in args
        assert "stream-json" in args
        assert "--allowedTools" in args
        assert '"Read"' in args
        assert '"Write"' in args
        assert "--max-turns" in args
        assert "4" in args  # max_turns + 1
        assert "--dangerously-skip-permissions" in args
        assert "--model" in args
        assert "sonnet" in args


class TestClaudeCodeAdapter:
    """Test Claude Code provider adapter."""

    def test_adapter_properties(self):
        """Test adapter basic properties."""
        adapter = ClaudeCodeAdapter()

        assert adapter.name == "claude_code"
        assert adapter.default_base_url == "claude_code://."
        assert adapter.request_model == ClaudeCodeRequestModel
        assert "exec:claude" in adapter.requires
        assert "fs.read" in adapter.requires
        assert "fs.write" in adapter.requires

    def test_supports_detection(self):
        """Test adapter supports various configurations."""
        adapter = ClaudeCodeAdapter()

        # Direct provider match
        assert adapter.supports(provider="claude_code", model=None, base_url=None)
        assert adapter.supports(provider="CLAUDE_CODE", model=None, base_url=None)

        # Model prefix match
        assert adapter.supports(provider=None, model="claude_code/sonnet", base_url=None)
        assert adapter.supports(provider=None, model="claude_code/opus", base_url=None)

        # Base URL match
        assert adapter.supports(provider=None, model=None, base_url="claude_code://.")
        assert adapter.supports(provider=None, model=None, base_url="claude_code:///tmp/repo")

        # No match
        assert not adapter.supports(provider="openai", model="gpt-4", base_url=None)
        assert not adapter.supports(provider=None, model="openai/gpt-4", base_url=None)

    def test_service_creation(self):
        """Test service creation with different configurations."""
        adapter = ClaudeCodeAdapter()

        # Default configuration
        service = adapter.create_service(base_url=None)
        assert isinstance(service, ClaudeCodeCLIService)
        assert service.base_repo == Path.cwd()

        # Custom repo path from base_url
        service = adapter.create_service(base_url="claude_code:///tmp/my-repo")
        assert service.base_repo == Path("/tmp/my-repo")

        # Custom repo path from kwargs
        service = adapter.create_service(base_url=None, base_repo="/tmp/custom-repo")
        assert service.base_repo == Path("/tmp/custom-repo")

    def test_required_rights(self):
        """Test required rights calculation."""
        adapter = ClaudeCodeAdapter()

        # Default rights
        rights = adapter.required_rights(base_url=None)
        assert "exec:claude" in rights
        assert "fs.read" in rights
        assert "fs.write" in rights

        # Custom repo rights
        rights = adapter.required_rights(base_url="claude_code:///tmp/custom")
        assert "exec:claude" in rights
        # Path gets resolved, so /tmp/custom becomes /private/tmp/custom on macOS
        resolved_custom = str(Path("/tmp/custom").resolve())
        assert f"fs.read:{resolved_custom}" in rights or "fs.read" in rights
        assert f"fs.write:{resolved_custom}" in rights or "fs.write" in rights


@pytest.mark.skipif(not Path("/tmp").exists(), reason="Test requires /tmp directory")
class TestClaudeCodeService:
    """Test Claude Code service implementation."""

    def test_service_creation(self):
        """Test service creation and properties."""
        service = create_claude_code_service("/tmp")

        assert isinstance(service, ClaudeCodeCLIService)
        assert service.name == "claude_code_cli"
        assert service.base_repo == Path("/tmp")
        assert "exec:claude" in service.requires

    @pytest.mark.asyncio
    async def test_stream_method_structure(self):
        """Test stream method with mocked transport."""
        service = create_claude_code_service("/tmp")
        request = ClaudeCodeRequestModel(prompt="Hello")
        context = CallContext(call_id=uuid4(), branch_id=uuid4())

        # Mock the transport stream method
        async def mock_stream(*args, **kwargs):
            """Mock streaming lines from Claude CLI."""
            # Yield JSON lines that would come from Claude CLI
            yield '{"type": "system", "session_id": "test-123"}'
            yield '{"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello!"}]}}'
            yield '{"type": "result", "result": "Response complete"}'

        # Patch the transport's stream method
        with patch.object(service._transport, 'stream', side_effect=mock_stream):
            chunks = []
            async for chunk in service.stream(request, ctx=context):
                chunks.append(chunk)

        # Verify we got expected chunks
        assert len(chunks) >= 2
        assert any(chunk.get("type") == "system" for chunk in chunks if isinstance(chunk, dict))


class TestClaudeCodeProviderRegistry:
    """Test Claude Code integration with provider registry."""

    def test_registry_integration(self):
        """Test Claude Code adapter is registered."""
        register_builtin_adapters()  # Ensure adapters are registered
        registry = get_provider_registry()

        # Should be registered after importing
        adapters = registry.known_adapters()
        assert "claude_code" in adapters

    def test_provider_resolution(self):
        """Test provider resolution for Claude Code."""
        register_builtin_adapters()  # Ensure adapters are registered
        registry = get_provider_registry()

        # Test direct provider resolution
        resolution, adapter = registry.resolve(provider="claude_code", model=None, base_url=None)

        assert resolution.provider == "claude_code"
        assert resolution.adapter_name == "claude_code"
        assert adapter.name == "claude_code"

    def test_service_creation_through_registry(self):
        """Test service creation through provider registry."""
        register_builtin_adapters()  # Ensure adapters are registered
        registry = get_provider_registry()

        service, resolution, rights = registry.create_service(
            provider="claude_code", model=None, base_url=None, base_repo="/tmp"
        )

        assert isinstance(service, ClaudeCodeCLIService)
        assert resolution.provider == "claude_code"
        assert "exec:claude" in rights


@pytest.mark.skipif(not Path("/tmp").exists(), reason="Test requires /tmp directory")
class TestClaudeCodeiModelIntegration:
    """Test Claude Code integration with iModel."""

    @pytest.mark.asyncio
    async def test_imodel_creation(self):
        """Test creating iModel with Claude Code provider."""
        # Mock the async executor to avoid context issues
        with patch("lionagi.services.imodel.RateLimitedExecutor"):
            # This will use the adapter to create the service
            model = iModel(provider="claude_code", model="sonnet", base_repo="/tmp")

            assert model.provider == "claude_code"
            assert model.model == "sonnet"
            assert isinstance(model.service, ClaudeCodeCLIService)

    @pytest.mark.asyncio
    async def test_imodel_invoke_structure(self):
        """Test iModel invoke method structure with mocked execution."""
        mock_executor = AsyncMock()
        mock_executor.start = AsyncMock()
        with patch("lionagi.services.imodel.RateLimitedExecutor", return_value=mock_executor):
            async with iModel(provider="claude_code", model="sonnet", base_repo="/tmp") as model:
                # Mock the executor and service
                mock_call = AsyncMock()
                mock_call.wait_completion.return_value = {
                    "session_id": "test-session-123",
                    "result": "Task completed successfully",
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                    "total_cost_usd": 0.01,
                }

                with patch.object(model.executor, "submit_call", return_value=mock_call):
                    result = await model.invoke(
                        messages=[{"role": "user", "content": "Hello Claude Code"}]
                    )

                # Verify result structure
                assert "result" in result
                assert "session_id" in result
                assert result["session_id"] == "test-session-123"

    @pytest.mark.asyncio
    async def test_session_management_in_metadata(self):
        """Test that session management works through provider metadata."""
        # Mock the async components to avoid executor creation in sync context
        with patch("lionagi.services.imodel.RateLimitedExecutor"):
            model = iModel(provider="claude_code", model="sonnet", base_repo="/tmp")

            # Initially no session
            assert model.provider_metadata.session_id is None

            # Simulate session update (would happen in _post_process_result)
            model.provider_metadata.session_id = "test-session-456"
            assert model.provider_metadata.session_id == "test-session-456"
