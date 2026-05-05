"""Surgical gap-fill tests for mcp_wrapper.py missing branches.

Targets missing statements in lionagi/service/connections/mcp_wrapper.py:
Lines: 90-101 (_filter_env), 115-131 (_validate_command), 182 (set_security_config),
       299-353 (_create_client branches), 402-404 (create_mcp_tool dict content item)
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lionagi.service.connections.mcp_wrapper import (
    MCPConnectionPool,
    MCPSecurityConfig,
    _filter_env,
    _validate_command,
    create_mcp_tool,
)

# ---------------------------------------------------------------------------
# _filter_env (lines 90-101)
# ---------------------------------------------------------------------------


def test_filter_env_returns_all_when_filtering_disabled():
    """Line 90-91: filter_sensitive_env=False returns env unchanged."""
    config = MCPSecurityConfig(filter_sensitive_env=False)
    env = {"MY_API_KEY": "secret", "NORMAL_VAR": "value"}
    result = _filter_env(env, config)
    assert result is env  # exact same dict returned


def test_filter_env_removes_sensitive_keys():
    """Lines 93-101: sensitive keys are filtered out."""
    config = MCPSecurityConfig(filter_sensitive_env=True)
    env = {
        "MY_API_KEY": "should_be_removed",
        "DB_PASSWORD": "also_removed",
        "SAFE_VAR": "kept",
        "ANOTHER_SAFE": "also_kept",
    }
    result = _filter_env(env, config)
    assert "SAFE_VAR" in result
    assert "ANOTHER_SAFE" in result
    assert "MY_API_KEY" not in result
    assert "DB_PASSWORD" not in result


def test_filter_env_case_insensitive_matching():
    """Lines 96-99: matching is case-insensitive via key.upper()."""
    config = MCPSecurityConfig(filter_sensitive_env=True)
    env = {
        "my_api_key": "lowercase_key_should_match",
        "normal_var": "safe",
    }
    result = _filter_env(env, config)
    assert "my_api_key" not in result
    assert "normal_var" in result


def test_filter_env_custom_denylist():
    """Lines 93-101: custom denylist patterns are applied."""
    config = MCPSecurityConfig(
        filter_sensitive_env=True,
        env_denylist_patterns=frozenset({"CUSTOM_PATTERN"}),
    )
    env = {
        "HAS_CUSTOM_PATTERN_VAR": "removed",
        "SAFE_VAR": "kept",
    }
    result = _filter_env(env, config)
    assert "HAS_CUSTOM_PATTERN_VAR" not in result
    assert "SAFE_VAR" in result


def test_filter_env_empty_dict():
    """Edge case: empty env returns empty dict."""
    config = MCPSecurityConfig(filter_sensitive_env=True)
    result = _filter_env({}, config)
    assert result == {}


# ---------------------------------------------------------------------------
# _validate_command (lines 115-133)
# ---------------------------------------------------------------------------


def test_validate_command_no_allowlist_returns_none():
    """Line 115-116: None allowlist skips validation."""
    config = MCPSecurityConfig(command_allowlist=None)
    # Should not raise
    _validate_command("any_command", config)
    _validate_command("/path/to/binary", config)


def test_validate_command_in_allowlist():
    """Line 130: command in allowlist passes."""
    config = MCPSecurityConfig(command_allowlist=frozenset({"python", "node"}))
    # Should not raise
    _validate_command("python", config)
    _validate_command("node", config)


def test_validate_command_not_in_allowlist_raises():
    """Line 130-133: command not in allowlist raises ValueError."""
    config = MCPSecurityConfig(command_allowlist=frozenset({"python"}))
    with pytest.raises(ValueError, match="not in allowlist"):
        _validate_command("ruby", config)


def test_validate_command_path_separator_bare_in_allowlist_raises():
    """Lines 119-125: path with separator where bare name IS in allowlist raises."""
    config = MCPSecurityConfig(command_allowlist=frozenset({"python"}))
    with pytest.raises(ValueError, match="path separator"):
        _validate_command("/usr/bin/python", config)


def test_validate_command_path_separator_bare_not_in_allowlist_raises():
    """Lines 119-128: path with separator where bare name is NOT in allowlist."""
    config = MCPSecurityConfig(command_allowlist=frozenset({"node"}))
    with pytest.raises(ValueError, match="not in allowlist"):
        _validate_command("/usr/bin/python", config)


def test_validate_command_backslash_separator():
    """Line 119: backslash separator also triggers path check."""
    config = MCPSecurityConfig(command_allowlist=frozenset({"python"}))
    with pytest.raises(ValueError):
        _validate_command("C:\\Python\\python.exe", config)


# ---------------------------------------------------------------------------
# MCPConnectionPool.set_security_config (line 182)
# ---------------------------------------------------------------------------


def test_set_security_config():
    """Line 182: set_security_config stores the config."""
    original = MCPConnectionPool._security
    try:
        config = MCPSecurityConfig(command_allowlist=frozenset({"python"}))
        MCPConnectionPool.set_security_config(config)
        assert MCPConnectionPool._security is config
    finally:
        MCPConnectionPool._security = original


# ---------------------------------------------------------------------------
# _create_client with security config (lines 307-325)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_pool_security():
    """Reset security config after each test."""
    original = MCPConnectionPool._security
    yield
    MCPConnectionPool._security = original


def _make_fastmcp_mock():
    """Build a sys.modules-compatible mock for fastmcp."""
    import sys
    import types

    fastmcp_mod = types.ModuleType("fastmcp")
    mock_client_cls = MagicMock()
    fastmcp_mod.Client = mock_client_cls

    fastmcp_client_mod = types.ModuleType("fastmcp.client")
    fastmcp_transports_mod = types.ModuleType("fastmcp.client.transports")
    mock_transport_cls = MagicMock()
    fastmcp_transports_mod.StdioTransport = mock_transport_cls
    fastmcp_client_mod.transports = fastmcp_transports_mod

    return (
        fastmcp_mod,
        fastmcp_client_mod,
        fastmcp_transports_mod,
        mock_client_cls,
        mock_transport_cls,
    )


@pytest.mark.asyncio
async def test_create_client_with_security_config_validates_command():
    """Lines 307-308: when security config set, command is validated."""
    import sys

    MCPConnectionPool._security = MCPSecurityConfig(
        command_allowlist=frozenset({"python"})
    )
    config = {"command": "ruby", "args": []}

    fm, fc, ft, _c, _t = _make_fastmcp_mock()
    with patch.dict(
        sys.modules,
        {
            "fastmcp": fm,
            "fastmcp.client": fc,
            "fastmcp.client.transports": ft,
        },
    ):
        with pytest.raises(ValueError, match="not in allowlist"):
            await MCPConnectionPool._create_client(config)


@pytest.mark.asyncio
async def test_create_client_with_security_config_filters_env():
    """Lines 322-323: when security config set, env is filtered."""
    import sys

    custom_config = MCPSecurityConfig(
        filter_sensitive_env=True,
        env_denylist_patterns=frozenset({"CUSTOM_PATTERN"}),
    )
    MCPConnectionPool._security = custom_config
    config = {
        "command": "python",
        "args": [],
        "env": {"SAFE_VAR": "value"},
    }

    fm, fc, ft, mock_client_cls, mock_transport_cls = _make_fastmcp_mock()
    mock_client_instance = AsyncMock()
    mock_client_cls.return_value = mock_client_instance
    mock_transport_instance = MagicMock()
    mock_transport_cls.return_value = mock_transport_instance

    with patch.dict(os.environ, {"HAS_CUSTOM_PATTERN_VAR": "secret"}, clear=False):
        with patch.dict(
            sys.modules,
            {
                "fastmcp": fm,
                "fastmcp.client": fc,
                "fastmcp.client.transports": ft,
            },
        ):
            await MCPConnectionPool._create_client(config)
            call_kwargs = mock_transport_cls.call_args.kwargs
            env = call_kwargs["env"]
            assert "HAS_CUSTOM_PATTERN_VAR" not in env
            assert "SAFE_VAR" in env


@pytest.mark.asyncio
async def test_create_client_no_security_uses_default_filter():
    """Lines 324-325: no security config → default filter applied."""
    import sys

    MCPConnectionPool._security = None
    config = {"command": "python", "args": []}

    fm, fc, ft, mock_client_cls, mock_transport_cls = _make_fastmcp_mock()
    mock_client_instance = AsyncMock()
    mock_client_cls.return_value = mock_client_instance
    mock_transport_instance = MagicMock()
    mock_transport_cls.return_value = mock_transport_instance

    with patch.dict(os.environ, {"MY_API_KEY": "secret"}, clear=False):
        with patch.dict(
            sys.modules,
            {
                "fastmcp": fm,
                "fastmcp.client": fc,
                "fastmcp.client.transports": ft,
            },
        ):
            await MCPConnectionPool._create_client(config)
            call_kwargs = mock_transport_cls.call_args.kwargs
            env = call_kwargs["env"]
            # Default sensitive patterns should filter API_KEY
            assert "MY_API_KEY" not in env


@pytest.mark.asyncio
async def test_create_client_mcp_debug_env_suppresses_log_override():
    """Line 330: MCP_DEBUG env var prevents LOG_LEVEL=ERROR override."""
    import sys

    MCPConnectionPool._security = None
    config = {"command": "python", "args": []}

    fm, fc, ft, mock_client_cls, mock_transport_cls = _make_fastmcp_mock()
    mock_client_instance = AsyncMock()
    mock_client_cls.return_value = mock_client_instance
    mock_transport_instance = MagicMock()
    mock_transport_cls.return_value = mock_transport_instance

    with patch.dict(os.environ, {"MCP_DEBUG": "true"}, clear=False):
        with patch.dict(
            sys.modules,
            {
                "fastmcp": fm,
                "fastmcp.client": fc,
                "fastmcp.client.transports": ft,
            },
        ):
            await MCPConnectionPool._create_client(config)
            call_kwargs = mock_transport_cls.call_args.kwargs
            env = call_kwargs["env"]
            # MCP_DEBUG=true means we should NOT force LOG_LEVEL=ERROR
            assert env.get("LOG_LEVEL") != "ERROR"


# ---------------------------------------------------------------------------
# create_mcp_tool — content dict item (lines 402-404)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_tool_result_content_is_list_with_dict_text_item():
    """Lines 402-404: result.content is a list with a dict item of type 'text'."""
    mcp_config = {"url": "http://localhost:8080"}
    tool_name = "test_tool"

    mock_client = AsyncMock()
    mock_result = MagicMock()
    # result.content is a list with a dict item (not an object with .text)
    mock_result.content = [{"type": "text", "text": "dict text content"}]
    mock_client.call_tool.return_value = mock_result

    with patch.object(MCPConnectionPool, "get_client", return_value=mock_client):
        tool = create_mcp_tool(mcp_config, tool_name)
        result = await tool()
    assert result == "dict text content"


@pytest.mark.asyncio
async def test_mcp_tool_result_content_is_list_with_non_text_item():
    """Line 404: content list with item that has no .text and isn't text type → return content."""
    mcp_config = {"url": "http://localhost:8080"}
    tool_name = "test_tool"

    mock_client = AsyncMock()
    mock_result = MagicMock()
    # content is a list with a single non-text dict item
    mock_result.content = [{"type": "image", "data": "base64..."}]
    # the item has no .text attribute and type != "text", so content is returned
    del mock_result.content[
        0
    ]  # make it a MagicMock with no .text but also trigger list path

    # Actually let's use a proper mock: content is a list with 1 item that has no .text
    item = MagicMock(spec=[])  # spec=[] means no attributes
    mock_result.content = [item]
    mock_client.call_tool.return_value = mock_result

    with patch.object(MCPConnectionPool, "get_client", return_value=mock_client):
        tool = create_mcp_tool(mcp_config, tool_name)
        result = await tool()
    # Falls through to return content
    assert result == [item]


@pytest.mark.asyncio
async def test_mcp_tool_result_content_multiple_items_returned_as_is():
    """Line 404: content with >1 items returns the whole list."""
    mcp_config = {"url": "http://localhost:8080"}
    tool_name = "test_tool"

    mock_client = AsyncMock()
    mock_result = MagicMock()
    mock_result.content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    mock_client.call_tool.return_value = mock_result

    with patch.object(MCPConnectionPool, "get_client", return_value=mock_client):
        tool = create_mcp_tool(mcp_config, tool_name)
        result = await tool()
    assert result == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]


# ---------------------------------------------------------------------------
# _create_client URL, invalid-args, and missing-key branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_client_url_config():
    """Line 301: URL config creates FastMCPClient with the URL."""
    import sys

    MCPConnectionPool._security = None
    config = {"url": "http://localhost:9000"}

    fm, fc, ft, mock_client_cls, _t = _make_fastmcp_mock()
    mock_client_instance = AsyncMock()
    mock_client_cls.return_value = mock_client_instance

    with patch.dict(
        sys.modules,
        {
            "fastmcp": fm,
            "fastmcp.client": fc,
            "fastmcp.client.transports": ft,
        },
    ):
        client = await MCPConnectionPool._create_client(config)
        # FastMCPClient should have been called with the URL
        mock_client_cls.assert_called_once_with("http://localhost:9000")
        assert client is mock_client_instance


@pytest.mark.asyncio
async def test_create_client_invalid_args_type():
    """Line 313: args not a list raises ValueError."""
    import sys

    MCPConnectionPool._security = None
    config = {"command": "python", "args": "not-a-list"}

    fm, fc, ft, mock_client_cls, _t = _make_fastmcp_mock()

    with patch.dict(
        sys.modules,
        {
            "fastmcp": fm,
            "fastmcp.client": fc,
            "fastmcp.client.transports": ft,
        },
    ):
        with pytest.raises(ValueError, match="Config 'args' must be a list"):
            await MCPConnectionPool._create_client(config)


@pytest.mark.asyncio
async def test_create_client_early_validation_neither_url_nor_command():
    """Lines 288-289: early check rejects config without url or command."""
    # This hits the early guard at line 288-289 before fastmcp is imported.
    # Line 349 is effectively unreachable (same guard already fired).
    MCPConnectionPool._security = None
    config = {"server_type": "custom"}

    with pytest.raises(ValueError, match="Config must have either 'url' or 'command'"):
        await MCPConnectionPool._create_client(config)


# ---------------------------------------------------------------------------
# MCPSecurityConfig defaults
# ---------------------------------------------------------------------------


def test_mcp_security_config_defaults():
    """Verify MCPSecurityConfig defaults."""
    config = MCPSecurityConfig()
    assert config.command_allowlist is None
    assert config.filter_sensitive_env is True
    assert config.max_connections_per_server == 5
    assert isinstance(config.env_denylist_patterns, frozenset)
    assert len(config.env_denylist_patterns) > 0


def test_mcp_security_config_custom():
    """Verify MCPSecurityConfig custom values."""
    config = MCPSecurityConfig(
        command_allowlist=frozenset({"python"}),
        filter_sensitive_env=False,
        max_connections_per_server=3,
    )
    assert config.command_allowlist == frozenset({"python"})
    assert config.filter_sensitive_env is False
    assert config.max_connections_per_server == 3
