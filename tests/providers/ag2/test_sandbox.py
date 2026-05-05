"""Tests for lionagi.providers.ag2.sandbox module.

Tests SandboxAgent and SandboxManager without requiring daytona.
Uses monkeypatching for daytona-dependent code paths.

Covers ~41 missing statements from lines 41-183 of sandbox.py.
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lionagi.providers.ag2.sandbox import SandboxAgent, SandboxManager

# ---------------------------------------------------------------------------
# SandboxAgent — dataclass creation
# ---------------------------------------------------------------------------


def test_sandbox_agent_creation():
    """SandboxAgent is a dataclass with expected fields."""
    mock_sandbox = MagicMock()
    agent = SandboxAgent(
        name="TestAgent",
        sandbox=mock_sandbox,
        url="https://sandbox-123.example.com",
        sandbox_id="sandbox-abc-123",
    )
    assert agent.name == "TestAgent"
    assert agent.sandbox is mock_sandbox
    assert agent.url == "https://sandbox-123.example.com"
    assert agent.sandbox_id == "sandbox-abc-123"


# ---------------------------------------------------------------------------
# SandboxManager — dataclass creation and defaults
# ---------------------------------------------------------------------------


def test_sandbox_manager_defaults():
    """SandboxManager has expected default field values."""
    manager = SandboxManager()
    assert manager.api_key is None
    assert manager.target == "us"
    assert manager.model is None
    assert manager.env_vars == {}
    assert manager._sandboxes == []
    assert manager._daytona is None


def test_sandbox_manager_custom_values():
    """SandboxManager accepts custom field values."""
    manager = SandboxManager(
        api_key="my-key",
        target="eu",
        model="gpt-4",
        env_vars={"KEY": "val"},
    )
    assert manager.api_key == "my-key"
    assert manager.target == "eu"
    assert manager.model == "gpt-4"
    assert manager.env_vars == {"KEY": "val"}


# ---------------------------------------------------------------------------
# SandboxManager._get_daytona
# ---------------------------------------------------------------------------


def test_get_daytona_returns_existing():
    """_get_daytona returns _daytona if already set."""
    mock_daytona = MagicMock()
    manager = SandboxManager()
    manager._daytona = mock_daytona
    result = manager._get_daytona()
    assert result is mock_daytona


def test_get_daytona_creates_with_api_key():
    """_get_daytona instantiates Daytona with api_key when provided."""
    manager = SandboxManager(api_key="test-api-key", target="us")

    mock_daytona_cls = MagicMock()
    mock_daytona_instance = MagicMock()
    mock_daytona_cls.return_value = mock_daytona_instance

    daytona_mod = types.ModuleType("daytona")
    daytona_mod.Daytona = mock_daytona_cls

    with patch.dict(sys.modules, {"daytona": daytona_mod}):
        result = manager._get_daytona()

    assert result is mock_daytona_instance
    assert manager._daytona is mock_daytona_instance
    mock_daytona_cls.assert_called_once_with(api_key="test-api-key", target="us")


def test_get_daytona_creates_without_api_key():
    """_get_daytona instantiates Daytona without api_key when not provided."""
    manager = SandboxManager(target="eu")

    mock_daytona_cls = MagicMock()
    mock_daytona_instance = MagicMock()
    mock_daytona_cls.return_value = mock_daytona_instance

    daytona_mod = types.ModuleType("daytona")
    daytona_mod.Daytona = mock_daytona_cls

    with patch.dict(sys.modules, {"daytona": daytona_mod}):
        result = manager._get_daytona()

    # api_key not passed since it's None
    call_kwargs = mock_daytona_cls.call_args.kwargs
    assert "api_key" not in call_kwargs
    assert call_kwargs.get("target") == "eu"


def test_get_daytona_without_target():
    """_get_daytona handles empty target string."""
    manager = SandboxManager(target="")

    mock_daytona_cls = MagicMock()
    mock_daytona_instance = MagicMock()
    mock_daytona_cls.return_value = mock_daytona_instance

    daytona_mod = types.ModuleType("daytona")
    daytona_mod.Daytona = mock_daytona_cls

    with patch.dict(sys.modules, {"daytona": daytona_mod}):
        result = manager._get_daytona()

    # target="" is falsy, so not passed
    call_kwargs = mock_daytona_cls.call_args.kwargs
    assert "target" not in call_kwargs


# ---------------------------------------------------------------------------
# SandboxManager.create_agent_sandbox
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_agent_sandbox_raises_without_model():
    """create_agent_sandbox raises ValueError when no model is set."""
    manager = SandboxManager()  # no model

    with pytest.raises(ValueError, match="requires an explicit model"):
        await manager.create_agent_sandbox(
            name="Agent",
            system_message="You are an assistant.",
        )


@pytest.mark.asyncio
async def test_create_agent_sandbox_uses_manager_model():
    """create_agent_sandbox uses manager.model when none passed explicitly."""
    manager = SandboxManager(model="gpt-4")

    mock_sandbox = MagicMock()
    mock_sandbox.id = "sandbox-001"
    mock_link = MagicMock()
    mock_link.url = "https://sandbox-001.example.com"
    mock_sandbox.get_preview_link.return_value = mock_link
    mock_sandbox.process = MagicMock()

    mock_daytona = MagicMock()
    mock_daytona.create.return_value = mock_sandbox
    manager._daytona = mock_daytona

    # Mock the daytona imports used inside create_agent_sandbox
    mock_image_cls = MagicMock()
    mock_image_instance = MagicMock()
    mock_image_cls.base.return_value.pip_install.return_value = mock_image_instance

    mock_params_cls = MagicMock()
    mock_resources_cls = MagicMock()

    daytona_mod = types.ModuleType("daytona")
    daytona_mod.Daytona = MagicMock()
    daytona_mod.CreateSandboxFromImageParams = mock_params_cls
    daytona_mod.Image = mock_image_cls
    daytona_mod.Resources = mock_resources_cls

    with patch.dict(sys.modules, {"daytona": daytona_mod}):
        agent = await manager.create_agent_sandbox(
            name="ExpertAgent",
            system_message="You are an expert.",
        )

    assert agent.name == "ExpertAgent"
    assert agent.url == "https://sandbox-001.example.com"
    assert agent.sandbox_id == "sandbox-001"
    assert agent in manager._sandboxes


@pytest.mark.asyncio
async def test_create_agent_sandbox_with_explicit_model():
    """create_agent_sandbox uses explicit model over manager.model."""
    manager = SandboxManager(model="gpt-3.5-turbo")

    mock_sandbox = MagicMock()
    mock_sandbox.id = "sb-002"
    mock_link = MagicMock()
    mock_link.url = "https://sandbox-002.example.com"
    mock_sandbox.get_preview_link.return_value = mock_link
    mock_sandbox.process = MagicMock()

    mock_daytona = MagicMock()
    mock_daytona.create.return_value = mock_sandbox
    manager._daytona = mock_daytona

    daytona_mod = types.ModuleType("daytona")
    daytona_mod.Daytona = MagicMock()
    daytona_mod.CreateSandboxFromImageParams = MagicMock()
    daytona_mod.Image = MagicMock()
    daytona_mod.Image.base.return_value.pip_install.return_value = MagicMock()
    daytona_mod.Resources = MagicMock()

    with patch.dict(sys.modules, {"daytona": daytona_mod}):
        agent = await manager.create_agent_sandbox(
            name="SpecialAgent",
            system_message="You are special.",
            model="gpt-4o",
            extra_packages=["numpy"],
        )

    assert agent.name == "SpecialAgent"


@pytest.mark.asyncio
async def test_create_agent_sandbox_with_extra_env():
    """create_agent_sandbox merges extra_env into env_vars."""
    manager = SandboxManager(model="gpt-4", env_vars={"BASE_KEY": "base_value"})

    mock_sandbox = MagicMock()
    mock_sandbox.id = "sb-003"
    mock_link = MagicMock()
    mock_link.url = "https://sb-003.example.com"
    mock_sandbox.get_preview_link.return_value = mock_link
    mock_sandbox.process = MagicMock()

    mock_daytona = MagicMock()
    mock_daytona.create.return_value = mock_sandbox
    manager._daytona = mock_daytona

    captured_params = {}

    def capture_create(params, timeout=None):
        captured_params["env_vars"] = (
            params.env_vars if hasattr(params, "env_vars") else None
        )
        return mock_sandbox

    mock_daytona.create.side_effect = capture_create

    mock_params_cls = MagicMock(side_effect=lambda **kw: MagicMock(**kw))

    daytona_mod = types.ModuleType("daytona")
    daytona_mod.Daytona = MagicMock()
    daytona_mod.CreateSandboxFromImageParams = mock_params_cls
    daytona_mod.Image = MagicMock()
    daytona_mod.Image.base.return_value.pip_install.return_value = MagicMock()
    daytona_mod.Resources = MagicMock()

    with patch.dict(sys.modules, {"daytona": daytona_mod}):
        agent = await manager.create_agent_sandbox(
            name="EnvAgent",
            system_message="Test agent.",
            extra_env={"EXTRA_KEY": "extra_value"},
        )

    assert agent.name == "EnvAgent"
    # Check that extra_env was merged — verify create params had env_vars
    mock_params_call_kwargs = mock_params_cls.call_args.kwargs
    env = mock_params_call_kwargs.get("env_vars", {})
    assert env.get("BASE_KEY") == "base_value"
    assert env.get("EXTRA_KEY") == "extra_value"


# ---------------------------------------------------------------------------
# SandboxManager.create_agent_configs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_agent_configs_calls_create_agent_sandbox():
    """create_agent_configs calls create_agent_sandbox for each spec."""
    manager = SandboxManager(model="gpt-4")

    mock_agent1 = SandboxAgent(
        name="Agent1",
        sandbox=MagicMock(),
        url="https://a1.example.com",
        sandbox_id="sb-1",
    )
    mock_agent2 = SandboxAgent(
        name="Agent2",
        sandbox=MagicMock(),
        url="https://a2.example.com",
        sandbox_id="sb-2",
    )

    call_count = 0

    async def mock_create_sandbox(name, system_message, model=None):
        nonlocal call_count
        call_count += 1
        return mock_agent1 if call_count == 1 else mock_agent2

    with patch.object(manager, "create_agent_sandbox", side_effect=mock_create_sandbox):
        specs = [
            {
                "name": "Agent1",
                "system_message": "You are agent 1.",
                "role": "assistant",
            },
            {"name": "Agent2", "system_message": "You are agent 2."},
        ]
        configs = await manager.create_agent_configs(specs)

    assert len(configs) == 2
    assert configs[0]["name"] == "Agent1"
    assert configs[0]["nlip_url"] == "https://a1.example.com"
    assert configs[0]["role"] == "assistant"
    assert configs[1]["name"] == "Agent2"
    assert configs[1]["nlip_url"] == "https://a2.example.com"
    assert configs[1]["role"] == "remote agent"  # default role


@pytest.mark.asyncio
async def test_create_agent_configs_default_system_message():
    """create_agent_configs uses default system_message when not provided."""
    manager = SandboxManager(model="gpt-4")

    captured_args = {}

    async def mock_create_sandbox(name, system_message, model=None):
        captured_args["system_message"] = system_message
        return SandboxAgent(
            name=name,
            sandbox=MagicMock(),
            url="https://sb.example.com",
            sandbox_id="sb-x",
        )

    with patch.object(manager, "create_agent_sandbox", side_effect=mock_create_sandbox):
        specs = [{"name": "MyAgent"}]  # no system_message
        await manager.create_agent_configs(specs)

    # Default: "You are {name}."
    assert captured_args["system_message"] == "You are MyAgent."


# ---------------------------------------------------------------------------
# SandboxManager.cleanup
# ---------------------------------------------------------------------------


def test_cleanup_calls_daytona_delete():
    """cleanup calls daytona.delete for each sandbox."""
    mock_daytona = MagicMock()
    manager = SandboxManager()
    manager._daytona = mock_daytona

    mock_sandbox1 = MagicMock()
    mock_sandbox2 = MagicMock()
    agent1 = SandboxAgent(name="a1", sandbox=mock_sandbox1, url="u1", sandbox_id="s1")
    agent2 = SandboxAgent(name="a2", sandbox=mock_sandbox2, url="u2", sandbox_id="s2")
    manager._sandboxes = [agent1, agent2]

    manager.cleanup()

    mock_daytona.delete.assert_any_call(mock_sandbox1)
    mock_daytona.delete.assert_any_call(mock_sandbox2)
    assert manager._sandboxes == []


def test_cleanup_continues_on_delete_error():
    """cleanup logs warning but continues when daytona.delete raises."""
    mock_daytona = MagicMock()
    mock_daytona.delete.side_effect = Exception("Delete failed")
    manager = SandboxManager()
    manager._daytona = mock_daytona

    mock_sandbox = MagicMock()
    agent = SandboxAgent(name="a", sandbox=mock_sandbox, url="u", sandbox_id="s")
    manager._sandboxes = [agent]

    # Should not raise
    manager.cleanup()
    assert manager._sandboxes == []


def test_cleanup_empty_sandboxes():
    """cleanup with no sandboxes is a no-op."""
    mock_daytona = MagicMock()
    manager = SandboxManager()
    manager._daytona = mock_daytona
    manager._sandboxes = []

    manager.cleanup()
    mock_daytona.delete.assert_not_called()


# ---------------------------------------------------------------------------
# SandboxManager context manager protocol
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aenter_returns_self():
    """__aenter__ returns the manager itself."""
    manager = SandboxManager()
    result = await manager.__aenter__()
    assert result is manager


@pytest.mark.asyncio
async def test_aexit_calls_cleanup():
    """__aexit__ calls cleanup."""
    manager = SandboxManager()
    mock_daytona = MagicMock()
    manager._daytona = mock_daytona

    agent = SandboxAgent(
        name="test",
        sandbox=MagicMock(),
        url="http://test",
        sandbox_id="test-id",
    )
    manager._sandboxes = [agent]

    await manager.__aexit__(None, None, None)
    # cleanup was called — sandboxes should be cleared
    assert manager._sandboxes == []


@pytest.mark.asyncio
async def test_context_manager_protocol():
    """Full async context manager usage."""
    manager = SandboxManager()
    mock_cleanup = MagicMock()
    manager.cleanup = mock_cleanup

    async with manager as m:
        assert m is manager

    mock_cleanup.assert_called_once()
