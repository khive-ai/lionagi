"""Tests for MCP security configuration."""

import os

import pytest

from lionagi.service.connections.mcp.wrapper import (
    MCPSecurityConfig,
    _filter_env,
    _validate_command,
)


class TestMCPSecurityConfig:
    """Test MCPSecurityConfig dataclass."""

    def test_default_config(self):
        """Default config allows all commands, filters sensitive env."""
        config = MCPSecurityConfig()
        assert config.command_allowlist is None
        assert config.filter_sensitive_env is True
        assert config.max_connections_per_server == 5
        assert len(config.env_denylist_patterns) > 0

    def test_custom_allowlist(self):
        """Custom allowlist restricts commands."""
        config = MCPSecurityConfig(
            command_allowlist=frozenset({"node", "python"})
        )
        assert "node" in config.command_allowlist
        assert "python" in config.command_allowlist

    def test_frozen(self):
        """Config is immutable."""
        config = MCPSecurityConfig()
        with pytest.raises(AttributeError):
            config.filter_sensitive_env = False


class TestFilterEnv:
    """Test environment variable filtering."""

    def test_filters_sensitive_keys(self):
        """Known sensitive patterns are filtered."""
        config = MCPSecurityConfig()
        env = {
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            "OPENAI_API_KEY": "sk-secret",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "DATABASE_URL": "postgres://...",
            "SAFE_VAR": "safe",
        }
        filtered = _filter_env(env, config)

        assert "PATH" in filtered
        assert "HOME" in filtered
        assert "SAFE_VAR" in filtered
        assert "OPENAI_API_KEY" not in filtered
        assert "AWS_SECRET_ACCESS_KEY" not in filtered
        assert "DATABASE_URL" not in filtered

    def test_no_filter_when_disabled(self):
        """All env vars pass when filtering is disabled."""
        config = MCPSecurityConfig(filter_sensitive_env=False)
        env = {"OPENAI_API_KEY": "sk-secret", "PATH": "/usr/bin"}
        filtered = _filter_env(env, config)

        assert "OPENAI_API_KEY" in filtered
        assert "PATH" in filtered

    def test_custom_deny_patterns(self):
        """Custom deny patterns are respected."""
        config = MCPSecurityConfig(
            env_denylist_patterns=frozenset({"CUSTOM_SECRET"})
        )
        env = {
            "CUSTOM_SECRET_KEY": "hidden",
            "PATH": "/usr/bin",
        }
        filtered = _filter_env(env, config)

        assert "CUSTOM_SECRET_KEY" not in filtered
        assert "PATH" in filtered

    def test_case_insensitive_matching(self):
        """Filtering is case-insensitive."""
        config = MCPSecurityConfig()
        env = {"openai_api_key": "sk-secret"}
        filtered = _filter_env(env, config)
        # Pattern is OPENAI_API_KEY, key is openai_api_key
        # Both get uppercased for comparison
        assert "openai_api_key" not in filtered


class TestValidateCommand:
    """Test command validation."""

    def test_no_allowlist_allows_all(self):
        """No allowlist means all commands pass."""
        config = MCPSecurityConfig(command_allowlist=None)
        # Should not raise
        _validate_command("anything", config)
        _validate_command("node", config)

    def test_allowlist_blocks_unlisted(self):
        """Commands not in allowlist are blocked."""
        config = MCPSecurityConfig(
            command_allowlist=frozenset({"node", "python"})
        )
        with pytest.raises(ValueError, match="not in allowlist"):
            _validate_command("bash", config)

    def test_allowlist_permits_listed(self):
        """Commands in allowlist are permitted."""
        config = MCPSecurityConfig(
            command_allowlist=frozenset({"node", "python"})
        )
        _validate_command("node", config)
        _validate_command("python", config)

    def test_path_separator_rejected_bare_in_allowlist(self):
        """Path commands rejected even when bare name is in allowlist."""
        config = MCPSecurityConfig(
            command_allowlist=frozenset({"node"})
        )
        with pytest.raises(ValueError, match="path separator"):
            _validate_command("/usr/bin/node", config)

    def test_path_separator_rejected_bare_not_in_allowlist(self):
        """Path commands rejected when bare name not in allowlist either."""
        config = MCPSecurityConfig(
            command_allowlist=frozenset({"python"})
        )
        with pytest.raises(ValueError, match="not in allowlist"):
            _validate_command("/usr/bin/node", config)

    def test_no_allowlist_allows_paths(self):
        """No allowlist means path separators are fine."""
        config = MCPSecurityConfig(command_allowlist=None)
        # Should not raise
        _validate_command("/usr/bin/node", config)
