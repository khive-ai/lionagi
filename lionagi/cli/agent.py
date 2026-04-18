# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
"""`li agent` — single-agent one-shot or resumed conversation."""

from __future__ import annotations

import argparse
import json
import sys

from lionagi import Branch, iModel, json_dumps
from lionagi.ln import acreate_path
from lionagi.ln.concurrency import run_async
from lionagi.protocols.generic.log import DataLoggerConfig
from lionagi.protocols.messages import AssistantResponse

from ._persistence import (
    LIONAGI_HOME,
    find_branch_json,
    load_last_branch,
    save_last_branch_pointer,
)
from ._providers import build_chat_model, parse_model_spec


async def _run_agent(
    model_str: str | None,
    prompt: str,
    yolo: bool = False,
    verbose: bool = False,
    theme: str | None = None,
    resume: str | None = None,
    continue_last: bool = False,
    effort: str | None = None,
) -> tuple[str, str, str]:
    """Execute one agent turn (new or resumed).

    Returns (result, provider, branch_id).
    """
    if resume and continue_last:
        raise ValueError(
            "--resume / -r and --continue-last / -c are mutually exclusive."
        )

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

    need_new_imodel = (
        branch is None
        or model_str is not None
        or yolo
        or verbose
        or theme is not None
        or effort is not None
    )

    if need_new_imodel:
        chat_model = build_chat_model(provider, model, yolo, verbose, theme, effort)
        if branch is None:
            branch = Branch(
                chat_model=chat_model,
                log_config=DataLoggerConfig(auto_save_on_exit=False),
            )
        else:
            if isinstance(chat_model, str):
                branch.chat_model = iModel(
                    provider=provider,
                    endpoint="query_cli",
                    model=model,
                    api_key="dummy",
                )
            else:
                branch.chat_model = chat_model
    elif branch is not None:
        # Resumed without explicit flags — reset runtime-only settings
        # so persisted verbose/yolo don't leak into new invocations
        branch.chat_model.endpoint.config.kwargs["verbose_output"] = verbose

    res = ""
    async for msg in branch.run(prompt):
        if isinstance(msg, AssistantResponse):
            res = msg.response

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
            "Use -r / -c to continue a previous conversation."
        ),
    )
    agent.add_argument(
        "model",
        nargs="?",
        default=None,
        help=(
            "One of 'claude', 'codex', 'gemini-code' (defaults), or a full spec "
            "like 'claude/opus'. Optional when --resume / --continue-last is set; "
            "providing it on resume overrides the saved chat_model."
        ),
    )
    agent.add_argument("prompt", help="Prompt to send to the subagent.")
    agent.add_argument(
        "--yolo",
        action="store_true",
        help=(
            "Auto-approve all tool calls. Maps per provider: "
            "claude→permission_mode=bypassPermissions, "
            "codex→full_auto (keeps workspace sandbox), gemini→--yolo. "
            "Provider request classes will emit their own safety warnings."
        ),
    )
    agent.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=(
            "Stream real-time output to terminal (live chunks, tools, thinking, "
            "final text). Sets verbose_output=True across all providers."
        ),
    )
    agent.add_argument(
        "--theme",
        choices=("light", "dark"),
        default=None,
        help=(
            "Terminal pretty-print theme. Default provider value (light) is used "
            "when unset. Applies to claude_code, codex, and gemini_code."
        ),
    )
    agent.add_argument(
        "--effort",
        metavar="LEVEL",
        default=None,
        help=(
            "Reasoning effort level. Provider-specific values: "
            "claude=low|medium|high|max, "
            "codex=none|minimal|low|medium|high|xhigh. "
            "Silently ignored for gemini_code (no equivalent)."
        ),
    )
    agent.add_argument(
        "-r",
        "--resume",
        metavar="BRANCH_ID",
        default=None,
        help=(
            "Resume a specific previous branch by ID. Loads "
            "~/.lionagi/logs/agents/*/<BRANCH_ID> and continues the conversation."
        ),
    )
    agent.add_argument(
        "-c",
        "--continue-last",
        action="store_true",
        help=(
            "Continue the most recently used branch "
            "(tracked in ~/.lionagi/last_branch.json)."
        ),
    )


def run_agent(args: argparse.Namespace) -> int:
    """Dispatch agent command."""
    if args.model is None and not (args.resume or args.continue_last):
        print(
            "error: model is required unless --resume / -r or "
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
        )
    )
    if not args.verbose:
        print(f"\n{result}" if result is not None else "")

    print(f'\n[to resume] li agent -r {branch_id} "..."', file=sys.stderr)
    return 0
