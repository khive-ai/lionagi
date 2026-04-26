# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for agent settings loading and hook resolution security."""

import pytest

from lionagi.agent.config import AgentConfig
from lionagi.agent.settings import apply_hooks_from_settings, load_settings


def test_load_settings_skips_project_settings_when_untrusted(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    (project / ".lionagi").mkdir(parents=True)
    (project / ".lionagi" / "settings.yaml").write_text(
        "hooks:\n  pre:\n    bash:\n      - python: lionagi.agent.hooks:guard_destructive\n"
    )
    monkeypatch.setenv("HOME", str(home))

    assert load_settings(project, include_project=False) == {}


def test_load_settings_includes_project_settings_when_trusted(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    (project / ".lionagi").mkdir(parents=True)
    (project / ".lionagi" / "settings.yaml").write_text(
        "hooks:\n  pre:\n    bash:\n      - python: lionagi.agent.hooks:guard_destructive\n"
    )
    monkeypatch.setenv("HOME", str(home))

    settings = load_settings(project, include_project=True)

    assert settings["hooks"]["pre"]["bash"][0]["python"] == (
        "lionagi.agent.hooks:guard_destructive"
    )


def test_apply_hooks_rejects_untrusted_python_modules():
    settings = {"hooks": {"pre": {"bash": [{"python": "os:path"}]}}}

    with pytest.raises(PermissionError, match="Untrusted hook module"):
        apply_hooks_from_settings(AgentConfig(), settings)


def test_apply_hooks_rejects_shell_string_commands():
    settings = {"hooks": {"pre": {"bash": [{"command": "echo unsafe"}]}}}

    with pytest.raises(ValueError, match="argv list"):
        apply_hooks_from_settings(AgentConfig(), settings)
