# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for create_agent: wiring tools, permissions, hooks."""

import pytest

from lionagi.agent.config import AgentConfig
from lionagi.agent.factory import create_agent
from lionagi.session.branch import Branch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make(config: AgentConfig) -> Branch:
    return await create_agent(config, load_settings=False)


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------


async def test_create_agent_default_config_returns_branch():
    config = AgentConfig()
    branch = await _make(config)
    assert isinstance(branch, Branch)


async def test_create_agent_default_no_coding_tools():
    """Default config with no tools= list should not register coding tools."""
    config = AgentConfig()
    branch = await _make(config)
    coding_tools = {"reader", "editor", "bash", "search", "context", "sandbox", "subagent"}
    assert not coding_tools.intersection(branch.acts.registry.keys())


# ---------------------------------------------------------------------------
# Coding preset
# ---------------------------------------------------------------------------


_CODING_TOOLS = {"reader", "editor", "bash", "search", "context", "sandbox", "subagent"}


async def test_create_agent_coding_preset_registers_7_tools():
    config = AgentConfig.coding()
    branch = await _make(config)
    registered_coding = _CODING_TOOLS.intersection(branch.acts.registry.keys())
    assert registered_coding == _CODING_TOOLS


async def test_create_agent_coding_preset_tool_names():
    config = AgentConfig.coding()
    branch = await _make(config)
    assert _CODING_TOOLS.issubset(branch.acts.registry.keys())


async def test_create_agent_coding_all_tools_async():
    """Every registered tool's callable must be a coroutine function."""
    import asyncio

    config = AgentConfig.coding()
    branch = await _make(config)
    for name, tool in branch.acts.registry.items():
        assert asyncio.iscoroutinefunction(
            tool.func_callable
        ), f"Tool '{name}' is not async"


# ---------------------------------------------------------------------------
# Permissions wired as preprocessor
# ---------------------------------------------------------------------------


async def test_create_agent_with_permissions_sets_preprocessor():
    from lionagi.agent.permissions import PermissionPolicy

    config = AgentConfig.coding()
    config.permissions = PermissionPolicy.read_only()
    branch = await _make(config)

    # Only coding tools get permission preprocessors (MCP tools from ambient env are unaffected)
    for name in _CODING_TOOLS:
        tool = branch.acts.registry.get(name)
        assert tool is not None, f"Coding tool '{name}' not registered"
        assert tool.preprocessor is not None, f"Tool '{name}' missing preprocessor"


async def test_create_agent_permission_deny_all_preprocessor_raises():
    """If deny_all policy is set, preprocessor on any tool should raise PermissionError."""
    from lionagi.agent.permissions import PermissionPolicy

    config = AgentConfig.coding()
    config.permissions = PermissionPolicy.deny_all()
    branch = await _make(config)

    reader_tool = branch.acts.registry["reader"]
    assert reader_tool.preprocessor is not None
    with pytest.raises(PermissionError):
        await reader_tool.preprocessor(
            {"action": "read", "path": "/tmp/x.py"},
        )


# ---------------------------------------------------------------------------
# load_settings=False — no side effects
# ---------------------------------------------------------------------------


async def test_create_agent_load_settings_false_no_side_effects(tmp_path, monkeypatch):
    """load_settings=False must not read .lionagi/settings.yaml."""
    import lionagi.agent.factory as factory_mod

    called = []

    def fake_load(project_dir, include_project):
        called.append(True)
        return {}

    monkeypatch.setattr(
        "lionagi.agent.settings.load_settings", fake_load, raising=False
    )

    config = AgentConfig()
    await create_agent(config, load_settings=False)
    assert called == [], "load_settings was called despite load_settings=False"


# ---------------------------------------------------------------------------
# Hooks wired into tools
# ---------------------------------------------------------------------------


async def test_pre_hook_registered_on_tool():
    config = AgentConfig.coding()
    calls = []

    async def my_hook(tool_name, action, args):
        calls.append(tool_name)
        return None  # pass through

    config.pre("bash", my_hook)
    branch = await _make(config)

    bash_tool = branch.acts.registry["bash"]
    assert bash_tool.preprocessor is not None
    # Invoke the preprocessor to verify our hook is wired
    await bash_tool.preprocessor({"action": "run", "command": "echo hi"})
    assert "bash" in calls


async def test_post_hook_registered_on_tool():
    config = AgentConfig.coding()
    calls = []

    async def my_post(tool_name, action, args, result):
        calls.append(tool_name)
        return result

    config.post("reader", my_post)
    branch = await _make(config)

    reader_tool = branch.acts.registry["reader"]
    assert reader_tool.postprocessor is not None
    result = {"success": True}
    await reader_tool.postprocessor(result)
    assert "reader" in calls
