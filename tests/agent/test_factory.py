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
    coding_tools = {
        "reader",
        "editor",
        "bash",
        "search",
        "context",
        "sandbox",
        "subagent",
    }
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


async def test_create_agent_coding_permissions_recheck_user_mutated_args(tmp_path):
    """User pre-hooks must not be able to rewrite safe args after permission checks."""
    from lionagi.agent.permissions import PermissionPolicy

    config = AgentConfig.coding(cwd=str(tmp_path))
    config.permissions = PermissionPolicy(
        mode="rules",
        allow={"bash": ["echo *"]},
        deny={"bash": ["rm *"]},
    )

    async def rewrite_to_denied(tool_name, action, args):
        return {**args, "command": "rm /tmp/important"}

    config.pre("bash", rewrite_to_denied)
    branch = await _make(config)

    bash_tool = branch.acts.registry["bash"]
    with pytest.raises(PermissionError, match="denied by rule"):
        await bash_tool.preprocessor({"action": "run", "command": "echo ok"})


async def test_create_agent_standalone_permissions_recheck_user_mutated_args():
    """Standalone tools get the same post-mutation permission validation."""
    from lionagi.agent.permissions import PermissionPolicy

    config = AgentConfig(tools=["bash"])
    config.permissions = PermissionPolicy(
        mode="rules",
        allow={"bash": ["echo *"]},
        deny={"bash": ["rm *"]},
    )

    async def rewrite_to_denied(tool_name, action, args):
        return {**args, "command": "rm /tmp/important"}

    config.pre("bash", rewrite_to_denied)
    branch = await _make(config)

    bash_tool = branch.acts.registry["bash_tool"]
    with pytest.raises(PermissionError, match="denied by rule"):
        await bash_tool.preprocessor({"action": "run", "command": "echo ok"})


# ---------------------------------------------------------------------------
# load_settings=False — no side effects
# ---------------------------------------------------------------------------


async def test_create_agent_load_settings_false_no_side_effects(monkeypatch):
    """load_settings=False must not read .lionagi/settings.yaml."""
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


async def test_create_agent_does_not_autoload_project_mcp_without_trust(
    tmp_path, monkeypatch
):
    from lionagi.protocols.action.manager import ActionManager

    project = tmp_path / "project"
    project.mkdir()
    (project / ".mcp.json").write_text('{"mcpServers": {"demo": {"command": "true"}}}')
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    calls = []

    async def fake_load_mcp_config(self, config_path, server_names=None, update=False):
        calls.append((config_path, server_names, update))
        return {}

    monkeypatch.setattr(ActionManager, "load_mcp_config", fake_load_mcp_config)

    await create_agent(
        AgentConfig(cwd=str(project)),
        load_settings=False,
        trust_project_settings=False,
    )

    assert calls == []


async def test_create_agent_autoloads_project_mcp_when_trusted(tmp_path, monkeypatch):
    from lionagi.protocols.action.manager import ActionManager

    project = tmp_path / "project"
    project.mkdir()
    mcp_path = project / ".mcp.json"
    mcp_path.write_text('{"mcpServers": {"demo": {"command": "true"}}}')
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    calls = []

    async def fake_load_mcp_config(self, config_path, server_names=None, update=False):
        calls.append((config_path, server_names, update))
        return {}

    monkeypatch.setattr(ActionManager, "load_mcp_config", fake_load_mcp_config)

    await create_agent(
        AgentConfig(cwd=str(project)),
        load_settings=False,
        trust_project_settings=True,
    )

    assert calls == [(str(mcp_path), None, False)]


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


# ---------------------------------------------------------------------------
# A5: model string parsed into provider / model / effort / yolo kwargs
# ---------------------------------------------------------------------------


async def test_create_agent_parses_model_provider_effort_and_yolo_kwargs(monkeypatch):
    import lionagi.cli._providers as providers_mod
    import lionagi.service.imodel as imodel_mod

    monkeypatch.setitem(providers_mod.PROVIDER_EFFORT_KWARG, "openai", "reasoning_effort")
    monkeypatch.setitem(providers_mod.PROVIDER_YOLO_KWARGS, "openai", {"stream": True})

    real_init = imodel_mod.iModel.__init__
    captured = {}

    def spy_init(self, *args, **kwargs):
        captured.update(kwargs)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(imodel_mod.iModel, "__init__", spy_init)

    config = AgentConfig(model="openai/gpt-4.1-mini", effort="high", yolo=True)
    branch = await create_agent(config, load_settings=False)

    assert isinstance(branch, Branch)
    assert captured.get("provider") == "openai"
    assert captured.get("model") == "gpt-4.1-mini"
    assert captured.get("reasoning_effort") == "high"
    assert captured.get("stream") is True


# ---------------------------------------------------------------------------
# A6: trust_project_settings=False prevents project settings from loading
# ---------------------------------------------------------------------------


async def test_create_agent_does_not_load_project_settings_without_trust(
    tmp_path, monkeypatch
):
    import lionagi.agent.settings as settings_mod

    (tmp_path / ".lionagi").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))

    calls = []
    real_load = settings_mod.load_settings

    def spy_load(project_dir=None, *, include_project=True):
        calls.append(include_project)
        return real_load(project_dir, include_project=include_project)

    monkeypatch.setattr(settings_mod, "load_settings", spy_load)

    config = AgentConfig()
    await create_agent(config, load_settings=True, trust_project_settings=False)

    assert calls == [False], f"load_settings called with include_project={calls}"


# ---------------------------------------------------------------------------
# C9: _chain_post_hooks ignores non-dict hook returns; dict returns update result
# ---------------------------------------------------------------------------


async def test_agent_post_hooks_ignore_non_dict_results_and_keep_previous_result():
    """Non-dict hook return is ignored; a subsequent dict return is applied."""
    from lionagi.agent.factory import _chain_post_hooks

    async def hook_returns_string(tool_name, op, kwargs, result):
        return "not a dict — should be ignored"

    async def hook_returns_dict(tool_name, op, kwargs, result):
        return {"ok": 2}

    chained = _chain_post_hooks("mytool", [hook_returns_string, hook_returns_dict])
    assert chained is not None

    final = await chained({"ok": 1})
    assert final == {"ok": 2}
