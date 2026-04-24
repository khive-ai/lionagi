# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for #919: li o flow -f spec.yaml file-based flow specification."""

import argparse
import json
from unittest.mock import AsyncMock, patch

import yaml

from lionagi.cli.orchestrate import (
    _load_flow_spec,
    add_orchestrate_subparser,
    run_orchestrate,
)


def _parse_flow_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="li")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_orchestrate_subparser(subparsers)
    return parser.parse_args(["o", "flow", *argv])


class TestLoadFlowSpec:
    def test_yaml_spec(self, tmp_path):
        spec = {
            "agent": "orchestrator",
            "team_mode": "ws-terminal",
            "workers": 8,
            "effort": "xhigh",
            "prompt": "Implement the terminal component",
        }
        p = tmp_path / "spec.yaml"
        p.write_text(yaml.dump(spec))
        result = _load_flow_spec(str(p))
        assert result["agent"] == "orchestrator"
        assert result["workers"] == 8
        assert result["effort"] == "xhigh"
        assert result["prompt"] == "Implement the terminal component"

    def test_json_spec(self, tmp_path):
        spec = {"model": "claude-code/opus-4-7", "prompt": "test task", "bare": True}
        p = tmp_path / "spec.json"
        p.write_text(json.dumps(spec))
        result = _load_flow_spec(str(p))
        assert result["model"] == "claude-code/opus-4-7"
        assert result["prompt"] == "test task"
        assert result["bare"] is True

    def test_yml_extension(self, tmp_path):
        spec = {"prompt": "hello"}
        p = tmp_path / "spec.yml"
        p.write_text(yaml.dump(spec))
        result = _load_flow_spec(str(p))
        assert result["prompt"] == "hello"

    def test_missing_file(self):
        result = _load_flow_spec("/nonexistent/path/spec.yaml")
        assert result is None

    def test_invalid_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(": : : invalid")
        result = _load_flow_spec(str(p))
        assert result is None

    def test_empty_yaml(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        result = _load_flow_spec(str(p))
        assert result == {}

    def test_unknown_extension_tries_yaml_first(self, tmp_path):
        spec = {"prompt": "detect format"}
        p = tmp_path / "spec.txt"
        p.write_text(yaml.dump(spec))
        result = _load_flow_spec(str(p))
        assert result["prompt"] == "detect format"

    def test_json_content_with_yaml_extension(self, tmp_path):
        spec = {"model": "codex/gpt-5.5", "prompt": "json in yaml"}
        p = tmp_path / "spec.yaml"
        p.write_text(json.dumps(spec))
        result = _load_flow_spec(str(p))
        assert result == spec

    def test_scalar_spec_returns_none(self, tmp_path, caplog):
        p = tmp_path / "scalar.yaml"
        p.write_text("2\n")
        result = _load_flow_spec(str(p))
        assert result is None
        assert "spec file must contain a YAML/JSON object" in caplog.text

    def test_full_spec_fields(self, tmp_path):
        spec = {
            "agent": "orchestrator",
            "model": "claude-code/opus-4-7",
            "team_mode": "ws-terminal",
            "workers": 8,
            "critic_model": "claude-code/opus-4-7",
            "effort": "xhigh",
            "max_agents": 12,
            "bare": False,
            "dry_run": False,
            "save": "/tmp/flow-out",
            "prompt": "Build a CLI tool",
        }
        p = tmp_path / "full.yaml"
        p.write_text(yaml.dump(spec))
        result = _load_flow_spec(str(p))
        assert result == spec

    def test_lone_positional_overrides_prompt_when_spec_supplies_model(
        self, tmp_path, capsys
    ):
        p = tmp_path / "spec.yaml"
        p.write_text(yaml.dump({"model": "claude-code/opus-4-7"}))
        args = _parse_flow_args(["-f", str(p), "Write the thing"])

        with patch(
            "lionagi.cli.orchestrate._run_flow",
            AsyncMock(return_value="flow output"),
        ) as run_flow:
            code = run_orchestrate(args)

        assert code == 0
        run_flow.assert_called_once()
        assert run_flow.call_args.kwargs["model_spec"] == "claude-code/opus-4-7"
        assert run_flow.call_args.kwargs["prompt"] == "Write the thing"
        assert capsys.readouterr().out.strip() == "flow output"

    def test_non_object_spec_fails_with_cli_error(self, tmp_path, caplog):
        p = tmp_path / "list.yaml"
        p.write_text("- item\n")
        args = _parse_flow_args(["-f", str(p)])

        code = run_orchestrate(args)

        assert code == 1
        assert "spec file must contain a YAML/JSON object" in caplog.text
