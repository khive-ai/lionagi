# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.cli._providers — parse_model_spec and build_imodel_from_spec."""

import pytest

from lionagi.cli._providers import build_imodel_from_spec, parse_model_spec


def test_parse_model_spec_rejects_effort_for_gemini_provider():
    """Gemini providers do not support effort levels — ValueError is raised."""
    with pytest.raises(ValueError, match="does not support effort"):
        parse_model_spec("gemini-code/gemini-3.1-pro-high")


def test_build_imodel_from_spec_maps_effort_and_yolo_without_network(monkeypatch):
    """build_imodel_from_spec passes correct kwargs to iModel for codex+xhigh+yolo."""
    calls = []

    class FakeIModel:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    import lionagi.cli._providers as pmod

    monkeypatch.setattr(pmod, "iModel", FakeIModel)

    build_imodel_from_spec("codex/gpt-5.4-xhigh", yolo=True, theme="dark")

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["model"] == "codex/gpt-5.4"
    assert kwargs.get("reasoning_effort") == "xhigh"
    assert kwargs.get("full_auto") is True
    assert kwargs.get("skip_git_repo_check") is True
    assert kwargs.get("cli_display_theme") == "dark"
