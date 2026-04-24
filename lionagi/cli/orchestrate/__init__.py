# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
"""`li orchestrate` — multi-agent orchestration patterns (fanout, flow)."""

from __future__ import annotations

import argparse
import sys

from lionagi._errors import TimeoutError as LionTimeoutError
from lionagi.ln.concurrency import run_async

from .._logging import hint, log_error
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
    fl.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Task for the orchestrator to plan and execute.",
    )
    fl.add_argument(
        "-f",
        "--file",
        metavar="PATH",
        default=None,
        help=(
            "Load flow spec from YAML or JSON file. File values serve as "
            "defaults; CLI flags override them. Prompt can come from the "
            "file (prompt: key) or as a positional argument."
        ),
    )
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


def _load_flow_spec(path: str) -> dict | None:
    """Load a YAML or JSON flow spec file.

    Returns a dict on success, or None after logging a CLI-facing error.
    Empty specs are treated as an empty object.
    """
    from pathlib import Path

    p = Path(path).expanduser()
    if not p.is_file():
        log_error(f"spec file not found: {p}")
        return None
    text = p.read_text()
    suffix = p.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text) or {}
        elif suffix == ".json":
            import json

            data = json.loads(text)
        else:
            import yaml

            try:
                data = yaml.safe_load(text) or {}
            except Exception:
                import json

                data = json.loads(text)
    except Exception as e:
        log_error(f"failed to parse spec file {p}: {e}")
        return None

    if not isinstance(data, dict):
        log_error("spec file must contain a YAML/JSON object")
        return None
    return data


def _validate_spec_fields(spec: dict) -> str | None:
    """Validate spec field types and ranges. Returns an error message or None."""
    workers = spec.get("workers")
    if workers is not None:
        if not isinstance(workers, int) or isinstance(workers, bool):
            return f"spec field 'workers' must be an integer, got {type(workers).__name__}"
        if not (1 <= workers <= 32):
            return f"spec field 'workers' must be in [1, 32], got {workers}"

    max_agents = spec.get("max_agents")
    if max_agents is not None:
        if not isinstance(max_agents, int) or isinstance(max_agents, bool):
            return f"spec field 'max_agents' must be an integer, got {type(max_agents).__name__}"
        if not (1 <= max_agents <= 50):
            return f"spec field 'max_agents' must be in [1, 50], got {max_agents}"

    effort = spec.get("effort")
    if effort is not None:
        if not isinstance(effort, str):
            return f"spec field 'effort' must be a string, got {type(effort).__name__}"
        if effort not in {"low", "medium", "high", "xhigh"}:
            return f"spec field 'effort' must be one of ['high', 'low', 'medium', 'xhigh'], got {effort!r}"

    for bool_field in ("bare", "dry_run", "with_synthesis"):
        val = spec.get(bool_field)
        if val is not None and not isinstance(val, bool):
            return f"spec field {bool_field!r} must be a bool, got {type(val).__name__}"

    prompt = spec.get("prompt")
    if prompt is not None:
        if not isinstance(prompt, str):
            return f"spec field 'prompt' must be a string, got {type(prompt).__name__}"
        if len(prompt) > 8192:
            return "spec field 'prompt' exceeds maximum length of 8192 characters"

    save = spec.get("save")
    if save is not None and not isinstance(save, str):
        return f"spec field 'save' must be a string, got {type(save).__name__}"

    for str_field in ("model", "agent", "team_mode"):
        val = spec.get(str_field)
        if val is not None and not isinstance(val, str):
            return f"spec field {str_field!r} must be a string, got {type(val).__name__}"

    return None


def run_orchestrate(args: argparse.Namespace) -> int:
    """Dispatch orchestrate sub-commands."""
    if args.orch_command == "fanout":
        has_model = args.model is not None or args.agent is not None
        if not has_model:
            log_error("model or --agent is required")
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
            log_error(str(e))
            return 1
        if not args.verbose:
            print(output)
        return 0

    if args.orch_command == "flow":
        # ── Load spec file if -f/--file was given ────────────────
        file_spec = getattr(args, "file", None)
        if file_spec:
            spec = _load_flow_spec(file_spec)
            if spec is None:
                return 1
            spec_err = _validate_spec_fields(spec)
            if spec_err is not None:
                log_error(spec_err)
                return 1
            # If the file supplies the model/agent, argparse's lone positional
            # is a prompt override, not a model override.
            if (
                args.model
                and args.prompt is None
                and (spec.get("model") or spec.get("agent"))
            ):
                args.prompt = args.model
                args.model = None
            # File values are defaults; CLI flags override.
            if args.model is None and "model" in spec:
                args.model = spec["model"]
            if args.agent is None and spec.get("agent"):
                args.agent = spec["agent"]
            if args.prompt is None and spec.get("prompt"):
                args.prompt = spec["prompt"]
            if args.max_concurrent == 0 and spec.get("workers"):
                args.max_concurrent = spec["workers"]
            if args.effort is None and spec.get("effort"):
                args.effort = spec["effort"]
            if args.with_synthesis is False and spec.get("with_synthesis"):
                args.with_synthesis = spec["with_synthesis"]
            if args.team_mode is None and spec.get("team_mode"):
                args.team_mode = spec["team_mode"]
            if args.max_agents == 0 and spec.get("max_agents"):
                args.max_agents = spec["max_agents"]
            if not args.bare and spec.get("bare"):
                args.bare = True
            if not args.dry_run and spec.get("dry_run"):
                args.dry_run = True
            if args.save is None and spec.get("save"):
                args.save = spec["save"]
            if spec.get("critic_model"):
                pass  # reserved for future use

        # Argparse assigns a lone positional to `model`, leaving prompt
        # None. When --agent supplies the model and the user passed a
        # single positional, that positional is actually the prompt.
        if args.model and not args.prompt and args.agent:
            args.prompt = args.model
            args.model = None

        has_model = args.model is not None or args.agent is not None
        if not has_model:
            log_error("model or --agent is required")
            return 1

        if not args.prompt:
            log_error("prompt is required (positional or via -f spec file)")
            return 1

        if args.save is not None:
            from pathlib import Path as _Path

            _resolved_save = _Path(args.save).expanduser().resolve()
            _safe_save = False
            for _root in (_Path.cwd().resolve(), _Path.home().resolve()):
                try:
                    _resolved_save.relative_to(_root)
                    _safe_save = True
                    break
                except ValueError:
                    pass
            if not _safe_save:
                log_error(
                    f"save path {str(_resolved_save)!r} escapes allowed roots "
                    f"(must be under cwd or home)"
                )
                return 1

        background = getattr(args, "background", False)
        if background and not args.save:
            log_error("--background requires --save")
            return 1

        if background:
            import subprocess
            from pathlib import Path as _Path

            bg_args = [a for a in sys.argv[1:] if a != "--background"]
            log_root = _Path(args.save).expanduser()
            log_root.mkdir(parents=True, exist_ok=True)
            log_path = log_root / "flow.log"
            with open(log_path, "w") as log_f:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "lionagi.cli", *bg_args],
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            hint(f"Flow running in background (PID {proc.pid})")
            hint(f"Output: {log_path}")
            hint(f"Monitor: tail -f {log_path}")
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
            log_error(str(e))
            return 1
        if not args.verbose:
            print(output)
        return 0

    log_error(f"Unknown orchestrate command: {args.orch_command}")
    return 1
