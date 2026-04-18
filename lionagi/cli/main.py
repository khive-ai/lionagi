# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
"""`li` — lionagi command line.

Examples:
    li agent claude/sonnet "Write a Python function to reverse a string."
    li agent codex/gpt-5.3-codex "..."
    li agent -r <branch-id> "follow-up prompt"
    li agent claude -c "follow-up prompt"

    li o fanout codex/gpt-5.4-xhigh "audit for dead code" -n 3
    li o fanout claude/sonnet "suggest approaches" -n 3 --with-synthesis claude/opus-4-6-medium
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from .agent import add_agent_subparser, run_agent
from .orchestrate import add_orchestrate_subparser, run_orchestrate
from .team import add_team_subparser, run_team


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    # Check for verbose early to set log level before anything runs
    _argv = argv if argv is not None else sys.argv[1:]
    if "-v" not in _argv and "--verbose" not in _argv:
        logging.getLogger("claude-cli").setLevel(logging.WARNING)
        logging.getLogger("codex-cli").setLevel(logging.WARNING)
        logging.getLogger("gemini-cli").setLevel(logging.WARNING)
        logging.getLogger("lionagi").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(
        prog="li",
        description="lionagi command line — spawn subagents via any CLI-backed provider.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    add_orchestrate_subparser(sub)
    add_agent_subparser(sub)
    add_team_subparser(sub)

    args = parser.parse_args(argv)

    if args.command in ("orchestrate", "o"):
        return run_orchestrate(args)

    if args.command == "agent":
        return run_agent(args)

    if args.command == "team":
        return run_team(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
