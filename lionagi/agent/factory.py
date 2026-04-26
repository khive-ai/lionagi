# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from lionagi.session.branch import Branch

from .config import AgentConfig

if TYPE_CHECKING:
    pass


async def create_agent(
    config: AgentConfig,
    *,
    load_settings: bool = True,
    project_dir: str | None = None,
) -> Branch:
    """Create a fully configured Branch from an AgentConfig.

    Wires: settings → hooks → system prompt → model → tools.

    Args:
        config: Agent configuration.
        load_settings: If True, load hooks from .lionagi/settings.yaml
            (global + project-local) and apply to config before building.
        project_dir: Project root for settings resolution. Auto-detected if None.

    Usage::

        config = AgentConfig.coding(model="openai/gpt-4.1")
        branch = await create_agent(config)
        response = await branch.chat("Fix the bug in utils.py")

    Returns:
        A Branch ready to use with tools registered and hooks applied.
    """
    if load_settings:
        from .settings import apply_hooks_from_settings, load_settings as _load

        settings = _load(project_dir)
        apply_hooks_from_settings(config, settings)

    from lionagi.service.imodel import iModel

    branch_kwargs = {}

    if config.model:
        from lionagi.cli._providers import build_chat_model, parse_model_spec

        provider, model_name = parse_model_spec(config.model)
        effort_kwargs = {}
        if config.effort:
            from lionagi.cli._providers import PROVIDER_EFFORT_KWARG

            effort_kwargs = PROVIDER_EFFORT_KWARG.get(provider, {}).get(config.effort, {})

        chat_model = build_chat_model(config.model, **(effort_kwargs or {}))
        branch_kwargs["chat_model"] = chat_model

    branch = Branch(**branch_kwargs)

    if config.system_prompt:
        if config.lion_system:
            from lionagi.session.prompts import LION_SYSTEM_MESSAGE

            full_prompt = LION_SYSTEM_MESSAGE.strip() + "\n\n" + config.system_prompt
        else:
            full_prompt = config.system_prompt
        branch.msgs.set_system(
            branch.msgs.create_system(system=full_prompt)
        )

    _register_tools(branch, config)
    await _load_mcp(branch, config)

    return branch


def _register_tools(branch: Branch, config: AgentConfig) -> None:
    """Register tools based on config.tools list, applying hooks."""
    for tool_spec in config.tools:
        if tool_spec == "coding":
            _register_coding_tools(branch, config)
        elif tool_spec == "reader":
            from lionagi.tools.file.reader import ReaderTool

            branch.register_tools(ReaderTool().to_tool())
        elif tool_spec == "editor":
            from lionagi.tools.file.editor import EditorTool

            branch.register_tools(EditorTool().to_tool())
        elif tool_spec == "bash":
            from lionagi.tools.code.bash import BashTool

            branch.register_tools(BashTool().to_tool())
        elif tool_spec == "search":
            from lionagi.tools.code.search import SearchTool

            branch.register_tools(SearchTool().to_tool())


def _register_coding_tools(branch: Branch, config: AgentConfig) -> None:
    """Register CodingToolkit with hooks from config."""
    from lionagi.tools.coding import CodingToolkit

    toolkit = CodingToolkit()

    for key, handlers in config.hook_handlers.items():
        parts = key.split(":", 1)
        if len(parts) != 2:
            continue
        phase, tool_name = parts
        for handler in handlers:
            if phase == "pre":
                toolkit.pre(tool_name, handler)
            elif phase == "post":
                toolkit.post(tool_name, handler)
            elif phase == "error":
                toolkit.on_error(tool_name, handler)

    tools = toolkit.bind(branch)
    branch.register_tools(tools)


async def _load_mcp(branch: Branch, config: AgentConfig) -> None:
    """Auto-discover and load MCP tools from .mcp.json files.

    Discovery order:
        1. config.mcp_config_path (explicit)
        2. .lionagi/.mcp.json (project-local)
        3. cwd/.mcp.json (current directory)
        4. ~/.lionagi/.mcp.json (global)
    """
    from pathlib import Path

    mcp_path = None

    if config.mcp_config_path:
        p = Path(config.mcp_config_path)
        if p.is_file():
            mcp_path = str(p)
    else:
        candidates = []
        cwd = Path(config.cwd) if config.cwd else Path.cwd()

        for parent in [cwd, *cwd.parents]:
            candidates.append(parent / ".lionagi" / ".mcp.json")
            candidates.append(parent / ".mcp.json")
            if (parent / ".lionagi").is_dir():
                break

        candidates.append(Path.home() / ".lionagi" / ".mcp.json")

        for candidate in candidates:
            if candidate.is_file():
                mcp_path = str(candidate)
                break

    if mcp_path is None:
        return

    await branch.acts.load_mcp_config(
        mcp_path,
        server_names=config.mcp_servers,
    )
