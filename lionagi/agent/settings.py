# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Load agent settings from .lionagi/settings.yaml — global + project-local merge.

Settings resolution (project-local wins):
    1. ~/.lionagi/settings.yaml        (global defaults)
    2. .lionagi/settings.yaml          (project-local overrides)

Hook configuration format::

    hooks:
      pre:
        bash:
          - command: "python ~/.lionagi/hooks/guard.py"
          - python: "lionagi.agent.hooks:guard_destructive"
        "*":
          - python: "my_project.hooks:custom_guard"
      post:
        editor:
          - command: "ruff format {file_path}"
        "*":
          - python: "lionagi.agent.hooks:log_tool_use"
      on_error:
        "*":
          - python: "lionagi.agent.hooks:error_reporter"
"""

from __future__ import annotations

import importlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from .config import AgentConfig


def load_settings(project_dir: str | Path | None = None) -> dict[str, Any]:
    """Load and merge settings from global + project-local .lionagi/settings.yaml."""
    merged: dict[str, Any] = {}

    global_path = Path.home() / ".lionagi" / "settings.yaml"
    if global_path.is_file():
        with open(global_path) as f:
            global_settings = yaml.safe_load(f) or {}
        _deep_merge(merged, global_settings)

    if project_dir:
        local_path = Path(project_dir) / ".lionagi" / "settings.yaml"
        if local_path.is_file():
            with open(local_path) as f:
                local_settings = yaml.safe_load(f) or {}
            _deep_merge(merged, local_settings)
    else:
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            candidate = parent / ".lionagi" / "settings.yaml"
            if candidate.is_file():
                with open(candidate) as f:
                    local_settings = yaml.safe_load(f) or {}
                _deep_merge(merged, local_settings)
                break

    return merged


def apply_hooks_from_settings(
    config: AgentConfig, settings: dict[str, Any] | None = None
) -> AgentConfig:
    """Apply hook configuration from settings dict to an AgentConfig.

    Resolves hook specs (shell commands, Python import paths) into callables
    and registers them on the config.
    """
    if settings is None:
        settings = load_settings()

    hooks_config = settings.get("hooks", {})

    for phase in ("pre", "post", "on_error"):
        phase_config = hooks_config.get(phase, {})
        for tool_name, hook_specs in phase_config.items():
            if not isinstance(hook_specs, list):
                hook_specs = [hook_specs]
            for spec in hook_specs:
                handler = _resolve_hook_spec(spec, phase, tool_name)
                if handler is None:
                    continue
                if phase == "pre":
                    config.pre(tool_name, handler)
                elif phase == "post":
                    config.post(tool_name, handler)
                elif phase == "on_error":
                    config.on_error(tool_name, handler)

    return config


def _resolve_hook_spec(spec: dict | str, phase: str, tool_name: str) -> Callable | None:
    """Resolve a hook spec into an async callable.

    Spec formats:
        {"python": "module.path:function_name"}  → import and return
        {"command": "shell command {file_path}"}  → wrap in shell executor
        "module.path:function_name"               → shorthand for python import
    """
    if isinstance(spec, str):
        return _import_hook(spec)

    if isinstance(spec, dict):
        if "python" in spec:
            return _import_hook(spec["python"])
        if "command" in spec:
            return _make_shell_hook(spec["command"], phase, tool_name)

    return None


def _import_hook(import_path: str) -> Callable | None:
    """Import a hook function from 'module.path:function_name'."""
    if ":" not in import_path:
        return None
    module_path, _, func_name = import_path.rpartition(":")
    try:
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
    except (ImportError, AttributeError):
        return None


def _make_shell_hook(command_template: str, phase: str, tool_name: str) -> Callable:
    """Create an async hook that runs a shell command.

    Pre-hooks: args passed as JSON on stdin. Non-zero exit = PermissionError.
    Post-hooks: result passed as JSON on stdin. Stdout captured but ignored.
    """
    if phase == "pre":

        async def shell_pre_hook(tn: str, action: str, args: dict) -> dict | None:
            cmd = command_template
            for k, v in args.items():
                cmd = cmd.replace(f"{{{k}}}", str(v))
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                input=json.dumps(args),
                timeout=10,
            )
            if result.returncode != 0:
                msg = result.stderr.strip() or f"Hook blocked: {command_template}"
                raise PermissionError(msg)
            return None

        return shell_pre_hook

    else:

        async def shell_post_hook(
            tn: str, action: str, args: dict, result: dict
        ) -> dict | None:
            cmd = command_template
            for k, v in {**args, **result}.items():
                cmd = cmd.replace(f"{{{k}}}", str(v))
            subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                input=json.dumps(result),
                timeout=10,
            )
            return None

        return shell_post_hook


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Lists are concatenated."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        elif k in base and isinstance(base[k], list) and isinstance(v, list):
            base[k] = base[k] + v
        else:
            base[k] = v
    return base
