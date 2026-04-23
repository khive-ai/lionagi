# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for #919: li o flow -f spec.yaml file-based flow specification."""

import json

import yaml

from lionagi.cli.orchestrate import _load_flow_spec


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
        assert result == 1

    def test_invalid_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(": : : invalid")
        result = _load_flow_spec(str(p))
        assert result == 1

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
