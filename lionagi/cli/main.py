# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
"""`li` — lionagi command line.

Examples:
    li agent claude/sonnet "Write a Python function to reverse a string."
    li agent codex/gpt-5.3-codex "..."
    li agent gemini-code/gemini-3.1-pro-preview "..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lionagi import Branch, iModel, json_dumps
from lionagi.ln import acreate_path
from lionagi.ln.concurrency import run_async
from lionagi.protocols.generic.log import DataLoggerConfig

BACKENDS: dict[str, str] = {
    "claude": "claude_code/sonnet",
    "claude-code": "claude_code/sonnet",
    "claude_code": "claude_code/sonnet",
    "codex": "codex/gpt-5.3-codex-spark",
    "gemini-code": "gemini_code/gemini-3.1-flash-lite-preview",
    "gemini_code": "gemini_code/gemini-3.1-flash-lite-preview",
    "gemini-cli": "gemini_code/gemini-3.1-flash-lite-preview",
    "gemini_cli": "gemini_code/gemini-3.1-flash-lite-preview",
}

# Per-provider auto-approve kwargs when --yolo is set.
#   claude_code: permission_mode="bypassPermissions" skips all permission prompts.
#   codex:       full_auto=True auto-approves but KEEPS the workspace-write sandbox
#                (bypass_approvals=True is the nuclear option that also disables
#                sandboxing — not wired to --yolo; request explicitly if needed).
#   gemini_code: yolo=True forwards --yolo to the gemini CLI.
PROVIDER_YOLO_KWARGS: dict[str, dict] = {
    "claude_code": {"permission_mode": "bypassPermissions"},
    "codex": {"full_auto": True},
    "gemini_code": {"yolo": True},
}


async def _run_agent(
    model_str: str,
    prompt: str,
    yolo: bool = False,
    verbose: bool = False,
    theme: str | None = None,
) -> str:
    spec = BACKENDS.get(model_str, model_str)
    provider, model = spec.split("/", 1)

    extra: dict = {}
    if yolo:
        extra.update(PROVIDER_YOLO_KWARGS.get(provider, {}))
    if verbose:
        extra["verbose_output"] = True
    if theme is not None:
        extra["cli_display_theme"] = theme

    if extra:
        chat_model = iModel(
            provider=provider,
            endpoint="query_cli",
            model=model,
            api_key="dummy",
            **extra,
        )
    else:
        chat_model = spec

    branch = Branch(
        chat_model=chat_model,
        log_config=DataLoggerConfig(auto_save_on_exit=False),
    )
    res = await branch.communicate(prompt)
    path = await acreate_path(
        directory=Path.home() / ".lionagi" / "logs" / "agents" / provider,
        filename=str(branch.id),
    )
    await path.write_text(json_dumps(branch.to_dict()))
    return res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="li",
        description="lionagi command line — spawn subagents via any CLI-backed provider.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    agent = sub.add_parser(
        "agent",
        help="Spawn one-shot subagent (blocking); prints final response.",
        description="Spawn a single subagent and wait for its final response.",
    )
    agent.add_argument(
        "model",
        help="One of 'claude', 'codex', 'gemini-code' to use default models, or a full spec like 'claude/opus'.",
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
    args = parser.parse_args(argv)

    if args.command == "agent":
        result = run_async(
            _run_agent(
                args.model,
                args.prompt,
                yolo=args.yolo,
                verbose=args.verbose,
                theme=args.theme,
            )
        )
        # In verbose mode the provider already streamed the final text to the
        # terminal — printing again would duplicate the output.
        if not args.verbose:
            print(result if result is not None else "")
        return 0


if __name__ == "__main__":
    sys.exit(main())
