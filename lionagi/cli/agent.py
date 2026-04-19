# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
"""`li agent` — single-agent one-shot or resumed conversation."""

from __future__ import annotations

import argparse
import json
import sys

from anyio import move_on_after

from lionagi import Branch, json_dumps
from lionagi.ln import acreate_path
from lionagi.ln.concurrency import run_async
from lionagi.protocols.generic.log import DataLoggerConfig
from lionagi.protocols.messages import AssistantResponse

from ._agents import load_agent_profile
from ._persistence import (
    LIONAGI_HOME,
    find_branch_json,
    load_last_branch,
    save_last_branch_pointer,
)
from ._providers import (
    PROVIDER_EFFORT_KWARG,
    PROVIDER_YOLO_KWARGS,
    add_common_cli_args,
    build_chat_model,
    parse_model_spec,
)


async def _run_agent(
    model_str: str | None,
    prompt: str,
    yolo: bool = False,
    verbose: bool = False,
    theme: str | None = None,
    resume: str | None = None,
    continue_last: bool = False,
    effort: str | None = None,
    agent_name: str | None = None,
    cwd: str | None = None,
    timeout: int | None = None,
) -> tuple[str, str, str]:
    """Execute one agent turn (new or resumed).

    Returns (result, provider, branch_id).
    """
    if resume and continue_last:
        raise ValueError(
            "--resume / -r and --continue-last / -c are mutually exclusive."
        )

    # Load agent profile if specified — provides defaults for model/effort/yolo
    profile = None
    if agent_name:
        profile = load_agent_profile(agent_name)
        if profile.model and model_str is None:
            model_str = profile.model
        if profile.effort and effort is None:
            effort = profile.effort
        if profile.yolo and not yolo:
            yolo = True

    branch: Branch | None = None
    if continue_last:
        _, branch_id = load_last_branch()
        _, branch_path = find_branch_json(branch_id)
        branch = Branch.from_dict(json.loads(branch_path.read_text()))
    elif resume:
        _, branch_path = find_branch_json(resume)
        branch = Branch.from_dict(json.loads(branch_path.read_text()))

    if model_str is not None:
        ms = parse_model_spec(model_str)
        if "/" in ms.model:
            provider, model = ms.model.split("/", 1)
        else:
            provider, model = ms.model, ms.model
        if ms.effort and not effort:
            effort = ms.effort
    elif branch is not None:
        ep_cfg = branch.chat_model.endpoint.config
        provider = ep_cfg.provider
        model = ep_cfg.kwargs.get("model")
    else:
        raise ValueError(
            "Provide a model spec (e.g. 'claude') for a new branch, "
            "or use --resume / --continue-last to reopen an existing one."
        )

    if branch is None:
        chat_model = build_chat_model(provider, model, yolo, verbose, theme, effort)
        branch = Branch(
            chat_model=chat_model,
            log_config=DataLoggerConfig(auto_save_on_exit=False),
        )
    else:
        # Update existing endpoint config — no new iModel needed
        cfg = branch.chat_model.endpoint.config.kwargs
        if model_str is not None:
            cfg["model"] = model
        if verbose:
            cfg["verbose_output"] = True
        if theme is not None:
            cfg["cli_display_theme"] = theme
        if effort is not None:
            kwarg = PROVIDER_EFFORT_KWARG.get(provider)
            if kwarg:
                cfg[kwarg] = effort
        if yolo:
            cfg.update(PROVIDER_YOLO_KWARGS.get(provider, {}))

    # Inject agent system prompt
    if profile and profile.system_prompt:
        branch.msgs.add_message(system=profile.system_prompt)

    # Build run kwargs
    run_kw = {}
    if cwd:
        run_kw["repo"] = cwd

    async def _do_run():
        res = ""
        async for msg in branch.run(prompt, **run_kw):
            if isinstance(msg, AssistantResponse):
                res = msg.response
        return res

    if timeout:
        with move_on_after(timeout) as cancel_scope:
            res = await _do_run()
        if cancel_scope.cancelled_caught:
            res = "[timed out]"
            print(f"Agent timed out after {timeout}s", file=sys.stderr)
    else:
        res = await _do_run()

    path = await acreate_path(
        directory=LIONAGI_HOME / "logs" / "agents" / provider,
        filename=str(branch.id),
        file_exist_ok=True,
    )
    await path.write_text(json_dumps(branch.to_dict()))
    save_last_branch_pointer(provider, str(branch.id))

    return res, provider, str(branch.id)


def add_agent_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register `li agent` sub-command."""
    agent = subparsers.add_parser(
        "agent",
        help="Spawn one-shot subagent (blocking); prints final response.",
        description=(
            "Spawn a single subagent and wait for its final response. "
            "Use -r / -c to continue a previous conversation. "
            "Use -a to load a profile from .lionagi/agents/."
        ),
    )
    agent.add_argument(
        "model",
        nargs="?",
        default=None,
        help=(
            "One of 'claude', 'codex', 'gemini-code' (defaults), or a full spec "
            "like 'claude/opus'. Optional when -a (agent profile) provides a model, "
            "or when --resume / --continue-last is set."
        ),
    )
    agent.add_argument("prompt", help="Prompt to send to the subagent.")
    agent.add_argument(
        "-a",
        "--agent",
        metavar="NAME",
        default=None,
        help=(
            "Load agent profile from .lionagi/agents/<NAME>.md. "
            "Profile provides system prompt, default model, effort, yolo. "
            "CLI flags override profile settings."
        ),
    )
    agent.add_argument(
        "-r",
        "--resume",
        metavar="BRANCH_ID",
        default=None,
        help="Resume a previous branch by ID.",
    )
    agent.add_argument(
        "-c",
        "--continue-last",
        action="store_true",
        help="Continue the most recently used branch.",
    )

    add_common_cli_args(agent)


def run_agent(args: argparse.Namespace) -> int:
    """Dispatch agent command."""
    has_model = args.model is not None or args.agent is not None
    if not has_model and not (args.resume or args.continue_last):
        print(
            "error: model or --agent is required unless --resume / -r or "
            "--continue-last / -c is set",
            file=sys.stderr,
        )
        return 1

    result, provider, branch_id = run_async(
        _run_agent(
            args.model,
            args.prompt,
            yolo=args.yolo,
            verbose=args.verbose,
            theme=args.theme,
            resume=args.resume,
            continue_last=args.continue_last,
            effort=args.effort,
            agent_name=args.agent,
            cwd=args.cwd,
            timeout=args.timeout,
        )
    )
    if not args.verbose:
        print(f"\n{result}" if result is not None else "", flush=True)

    print(f'\n[to resume] li agent -r {branch_id} "..."', file=sys.stderr)
    return 0
