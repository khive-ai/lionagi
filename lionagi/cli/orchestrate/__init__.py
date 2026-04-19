# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
"""`li orchestrate` — multi-agent orchestration patterns (fanout, flow)."""

from __future__ import annotations

import argparse
import sys

from lionagi._errors import TimeoutError as LionTimeoutError
from lionagi.ln.concurrency import run_async

from .._providers import add_common_cli_args
from .fanout import _run_fanout
from .flow import _run_flow


def add_orchestrate_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register `li orchestrate` (alias `li o`) with its sub-commands."""
    orch = subparsers.add_parser(
        "orchestrate",
        aliases=["o"],
        help="Multi-agent orchestration patterns.",
        description="Orchestrate multiple agents in structured patterns.",
    )
    orch_sub = orch.add_subparsers(dest="orch_command", required=True)

    fo = orch_sub.add_parser(
        "fanout",
        help="Fan-out N workers in parallel, optionally synthesize.",
        description=(
            "Orchestrator decomposes task into N agent requests, "
            "fans out to workers, optionally synthesizes. "
            "Effort can be embedded in model spec: claude/opus-4-7-high."
        ),
    )
    fo.add_argument(
        "model",
        nargs="?",
        default=None,
        help=(
            "Orchestrator model spec (provider/model-effort). "
            "Also used as default worker model unless --workers specified. "
            "Optional when -a/--agent provides a model."
        ),
    )
    fo.add_argument("prompt", help="Task prompt for the orchestrator to decompose.")
    fo.add_argument(
        "-a",
        "--agent",
        metavar="NAME",
        default=None,
        help=(
            "Load orchestrator profile from .lionagi/agents/<NAME>.md. "
            "Profile provides system prompt, default model, effort, yolo. "
            "CLI flags and positional model override profile settings."
        ),
    )

    fo.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=3,
        help="Number of workers (default: 3). Ignored if --workers set.",
    )
    fo.add_argument(
        "--workers",
        metavar="M1,M2,...",
        default=None,
        help="Comma-separated worker model specs (each can include effort).",
    )
    fo.add_argument(
        "--max-concurrent",
        type=int,
        default=0,
        help="Max concurrent workers (default: all).",
    )
    fo.add_argument(
        "--with-synthesis",
        nargs="?",
        const=True,
        default=False,
        metavar="MODEL",
        help="Enable synthesis. Bare flag uses orchestrator model; with arg uses that model.",
    )
    fo.add_argument(
        "--synthesis-prompt",
        default=None,
        help="Custom synthesis instruction.",
    )
    fo.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    fo.add_argument(
        "--save",
        metavar="DIR",
        default=None,
        help="Save outputs to directory.",
    )

    fo.add_argument(
        "--team-mode",
        nargs="?",
        const="fanout",
        default=None,
        metavar="NAME",
        help=(
            "Create a persistent team for this fanout. Workers get team context "
            "and results are posted as team messages. Bare flag uses 'fanout' as "
            "team name; with arg uses that name."
        ),
    )

    add_common_cli_args(fo)

    # ── flow sub-command ─────────────────────────────────────────────
    fl = orch_sub.add_parser(
        "flow",
        help="Auto-DAG pipeline: orchestrator plans DAG, engine executes.",
        description=(
            "Orchestrator analyzes the task, composes a DAG of agents "
            "with dependency edges, and executes with automatic "
            "parallelism where dependencies allow."
        ),
    )
    fl.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Orchestrator model spec. Optional when -a/--agent provides one.",
    )
    fl.add_argument("prompt", help="Task for the orchestrator to plan and execute.")
    fl.add_argument(
        "-a",
        "--agent",
        metavar="NAME",
        default=None,
        help="Load orchestrator profile from .lionagi/agents/<NAME>.md.",
    )
    fl.add_argument(
        "--with-synthesis",
        nargs="?",
        const=True,
        default=False,
        metavar="MODEL",
        help="Enable final synthesis. Bare flag uses orchestrator model.",
    )
    fl.add_argument(
        "--max-concurrent",
        type=int,
        default=0,
        help="Max concurrent agents within a phase (default: all).",
    )
    fl.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    fl.add_argument(
        "--save",
        metavar="DIR",
        default=None,
        help="Save outputs to directory.",
    )
    fl.add_argument(
        "--team-mode",
        nargs="?",
        const="flow",
        default=None,
        metavar="NAME",
        help="Create a persistent team for this flow.",
    )
    fl.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the DAG but don't execute. Shows agents, deps, and model resolution.",
    )
    fl.add_argument(
        "--show-graph",
        action="store_true",
        help="Render DAG as matplotlib visualization. With --save, saves PNG to save dir.",
    )
    fl.add_argument(
        "--background",
        action="store_true",
        help="Run flow in background. Requires --save. Check output in save dir.",
    )
    fl.add_argument(
        "--bare",
        action="store_true",
        help=(
            "Ignore agent profiles — all workers use the CLI model spec. "
            "Roles define behavioral focus only, no profile system prompts."
        ),
    )
    fl.add_argument(
        "--max-agents",
        type=int,
        default=0,
        metavar="N",
        help="Max total agents the orchestrator may plan (0 = unlimited).",
    )
    add_common_cli_args(fl)


def run_orchestrate(args: argparse.Namespace) -> int:
    """Dispatch orchestrate sub-commands."""
    if args.orch_command == "fanout":
        has_model = args.model is not None or args.agent is not None
        if not has_model:
            print(
                "error: model or --agent is required",
                file=sys.stderr,
            )
            return 1

        synth = args.with_synthesis
        with_synthesis = synth is not False
        synthesis_model = synth if isinstance(synth, str) else None

        try:
            output = run_async(
                _run_fanout(
                    model_spec=args.model or "",
                    prompt=args.prompt,
                    num_workers=args.num_workers,
                    workers_str=args.workers,
                    with_synthesis=with_synthesis,
                    synthesis_model=synthesis_model,
                    synthesis_prompt=args.synthesis_prompt,
                    max_concurrent=args.max_concurrent,
                    yolo=args.yolo,
                    verbose=args.verbose,
                    effort=args.effort,
                    theme=args.theme,
                    output_format=args.output,
                    save_dir=args.save,
                    team_name=args.team_mode,
                    cwd=args.cwd,
                    timeout=args.timeout,
                    agent_name=args.agent,
                )
            )
        except LionTimeoutError as e:
            print(str(e), file=sys.stderr)
            return 1
        if not args.verbose:
            print(output)
        return 0

    if args.orch_command == "flow":
        has_model = args.model is not None or args.agent is not None
        if not has_model:
            print("error: model or --agent is required", file=sys.stderr)
            return 1

        background = getattr(args, "background", False)
        if background and not args.save:
            print("error: --background requires --save", file=sys.stderr)
            return 1

        if background:
            import subprocess
            bg_args = [a for a in sys.argv[1:] if a != "--background"]
            proc = subprocess.Popen(
                [sys.executable, "-m", "lionagi.cli", *bg_args],
                stdout=open(f"{args.save}/flow.log", "w"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            print(f"Flow running in background (PID {proc.pid})", file=sys.stderr)
            print(f"Output: {args.save}/flow.log", file=sys.stderr)
            print(f"Monitor: tail -f {args.save}/flow.log", file=sys.stderr)
            return 0

        synth = args.with_synthesis
        with_synthesis = synth is not False
        synthesis_model = synth if isinstance(synth, str) else None

        try:
            output = run_async(
                _run_flow(
                    model_spec=args.model or "",
                    prompt=args.prompt,
                    with_synthesis=with_synthesis,
                    synthesis_model=synthesis_model,
                    max_concurrent=args.max_concurrent,
                    yolo=args.yolo,
                    verbose=args.verbose,
                    effort=args.effort,
                    theme=args.theme,
                    output_format=args.output,
                    save_dir=args.save,
                    team_name=args.team_mode,
                    cwd=args.cwd,
                    timeout=args.timeout,
                    agent_name=args.agent,
                    bare=args.bare,
                    max_agents=args.max_agents,
                    dry_run=args.dry_run,
                    show_graph=getattr(args, "show_graph", False),
                )
            )
        except LionTimeoutError as e:
            print(str(e), file=sys.stderr)
            return 1
        if not args.verbose:
            print(output)
        return 0

    print(f"Unknown orchestrate command: {args.orch_command}", file=sys.stderr)
    return 1
