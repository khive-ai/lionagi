# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
"""Fan-out execution: decompose → parallel workers → optional synthesis."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from lionagi import Branch, Session, json_dumps
from lionagi._errors import TimeoutError as LionTimeoutError
from lionagi.ln.concurrency import move_on_after
from lionagi.operations.builder import OperationGraphBuilder
from lionagi.operations.fields import Instruct
from lionagi.protocols.generic.log import DataLoggerConfig

from .._agents import load_agent_profile
from .._persistence import save_last_branch_pointer
from .._providers import build_imodel_from_spec, parse_model_spec
from ._common import (
    AGENT_REQUEST_FIELDS,
    BARE_WORKER_SYSTEM,
    TEAM_WORKER_SYSTEM,
    _create_fanout_team,
    _format_result_json,
    _format_result_text,
    _post_results_to_team,
    _resolve_worker_spec,
    persist_session_branches,
)


async def _run_fanout(
    model_spec: str,
    prompt: str,
    *,
    num_workers: int = 3,
    workers_str: str | None = None,
    with_synthesis: bool = False,
    synthesis_model: str | None = None,
    synthesis_prompt: str | None = None,
    max_concurrent: int = 0,
    yolo: bool = False,
    verbose: bool = False,
    effort: str | None = None,
    theme: str | None = None,
    output_format: str = "text",
    save_dir: str | None = None,
    team_name: str | None = None,
    cwd: str | None = None,
    timeout: int | None = None,
    agent_name: str | None = None,
) -> str:
    """Three-phase fan-out: decompose → fan out → synthesize."""
    _shared: dict = {}

    if timeout:
        with move_on_after(timeout) as cancel_scope:
            result = await _run_fanout_inner(
                model_spec,
                prompt,
                num_workers=num_workers,
                workers_str=workers_str,
                with_synthesis=with_synthesis,
                synthesis_model=synthesis_model,
                synthesis_prompt=synthesis_prompt,
                max_concurrent=max_concurrent,
                yolo=yolo,
                verbose=verbose,
                effort=effort,
                theme=theme,
                output_format=output_format,
                save_dir=save_dir,
                team_name=team_name,
                cwd=cwd,
                agent_name=agent_name,
                _shared=_shared,
            )
        if cancel_scope.cancelled_caught:
            session = _shared.get("session")
            if session:
                await persist_session_branches(session, save_dir)
            n_saved = len(_shared.get("saved_workers", []))
            msg = f"Fanout timed out after {timeout}s"
            if n_saved:
                msg += f" ({n_saved} worker results already saved to {save_dir})"
            print(msg, file=sys.stderr)
            raise LionTimeoutError(msg)
        return result
    return await _run_fanout_inner(
        model_spec,
        prompt,
        num_workers=num_workers,
        workers_str=workers_str,
        with_synthesis=with_synthesis,
        synthesis_model=synthesis_model,
        synthesis_prompt=synthesis_prompt,
        max_concurrent=max_concurrent,
        yolo=yolo,
        verbose=verbose,
        effort=effort,
        theme=theme,
        output_format=output_format,
        save_dir=save_dir,
        team_name=team_name,
        cwd=cwd,
        agent_name=agent_name,
        _shared=_shared,
    )


async def _run_fanout_inner(
    model_spec: str,
    prompt: str,
    *,
    num_workers: int = 3,
    workers_str: str | None = None,
    with_synthesis: bool = False,
    synthesis_model: str | None = None,
    synthesis_prompt: str | None = None,
    max_concurrent: int = 0,
    yolo: bool = False,
    verbose: bool = False,
    effort: str | None = None,
    theme: str | None = None,
    output_format: str = "text",
    save_dir: str | None = None,
    team_name: str | None = None,
    cwd: str | None = None,
    agent_name: str | None = None,
    _shared: dict | None = None,
) -> str:
    """Inner fanout logic (no timeout wrapper)."""
    t0 = time.monotonic()

    # Load orchestrator agent profile if specified
    orc_profile = None
    if agent_name:
        orc_profile = load_agent_profile(agent_name)
        if orc_profile.model and not model_spec:
            model_spec = orc_profile.model
        if orc_profile.effort and not effort:
            effort = orc_profile.effort
        if orc_profile.yolo and not yolo:
            yolo = True

    if not model_spec:
        raise ValueError(
            "Provide a model spec or use -a/--agent to load a profile with a model."
        )

    # Build orchestrator model
    orc_imodel = build_imodel_from_spec(
        model_spec,
        yolo=yolo,
        verbose=verbose,
        effort_override=effort,
        theme=theme,
    )

    # Determine available workers — resolve agent profile names
    worker_profiles = []
    if workers_str:
        worker_model_list = []
        for token in (s.strip() for s in workers_str.split(",")):
            wmodel, wprofile = _resolve_worker_spec(token)
            worker_model_list.append(wmodel)
            worker_profiles.append(wprofile)
    else:
        ms = parse_model_spec(model_spec)
        worker_model_list = [ms.model] * num_workers
        worker_profiles = [None] * num_workers

    # ── Team setup (if --team-mode) ─────────────────────────────────
    team_data = None
    worker_names = []
    for i, wp in enumerate(worker_profiles):
        if wp and wp.name:
            base = wp.name
            count = sum(
                1 for n in worker_names if n == base or n.startswith(f"{base}-")
            )
            worker_names.append(f"{base}-{count + 1}" if count > 0 else base)
        else:
            worker_names.append(f"worker-{i + 1}")
    if team_name:
        team_data = _create_fanout_team(team_name, worker_names)
        if not verbose:
            print(
                f"Team '{team_name}' created ({team_data['id']}): "
                f"{', '.join(worker_names)}",
                file=sys.stderr,
            )

    # Apply cwd to endpoint if specified
    if cwd:
        orc_imodel.endpoint.config.kwargs.setdefault("repo", Path(cwd))

    # ── Phase 1: Orchestrator decomposes task ─────────────────────────
    builder = OperationGraphBuilder("Fanout")
    orc_system = orc_profile.system_prompt if orc_profile else None
    orc_branch = Branch(
        chat_model=orc_imodel,
        system=orc_system,
        log_config=DataLoggerConfig(auto_save_on_exit=False),
        name="orchestrator",
    )
    session = Session(default_branch=orc_branch)
    if _shared is not None:
        _shared["session"] = session

    # Build guidance with role names when workers are agent profiles
    worker_descriptions = []
    for i, wm in enumerate(worker_model_list):
        wp = worker_profiles[i] if i < len(worker_profiles) else None
        if wp and wp.name:
            worker_descriptions.append(
                f"{worker_names[i]} (role: {wp.name}, model: {wm})"
            )
        else:
            worker_descriptions.append(f"{worker_names[i]} (model: {wm})")
    roster_guidance = "; ".join(worker_descriptions)

    root = builder.add_operation(
        "operate",
        branch=orc_branch,
        instruct=Instruct(
            instruction=(
                f"Generate {len(worker_model_list)} agent requests to address "
                f"the following task. Each agent should tackle a distinct angle "
                f"or perspective. Each agent is a WORKER that will DIRECTLY "
                f"answer its assigned sub-task — NOT delegate further."
            ),
            context={"user_prompt": prompt},
            guidance=(
                f"Available workers: {roster_guidance}. "
                f"CRITICAL: You MUST produce your output ONLY via the structured "
                f"output fields (the AgentRequest list). Do NOT use any provider-native "
                f"subagent or tool-spawning features (no Agent tool, no subprocess "
                f"spawning, no delegation tools). The ONLY correct way to dispatch "
                f"workers is by filling in the structured AgentRequest output. "
                f"Each AgentRequest instruction must be a DIRECT task the worker "
                f"will answer itself — not a meta-instruction to generate more "
                f"agents or decompose further. Workers are leaf executors, not "
                f"orchestrators. "
                f"Match each sub-task to the worker's role strengths. "
                f"If multiple models are provided, set the model field to "
                f"the exact model string for that worker. "
                f"If all workers use the same model, model can be null."
            ),
        ),
        field_models=[AGENT_REQUEST_FIELDS],
        reason=True,
    )

    if not verbose:
        print(
            f"Phase 1: Orchestrator decomposing task into "
            f"{len(worker_model_list)} agent requests...",
            file=sys.stderr,
        )

    result1 = await session.flow(builder.get_graph())
    t_decompose = time.monotonic() - t0

    # Extract agent requests
    root_result = result1.get("operation_results", {}).get(root)
    agents = getattr(root_result, "agents", None) or []

    if not agents:
        return "Orchestrator produced no agent requests."

    if not verbose:
        print(
            f"Phase 1 done ({t_decompose:.1f}s): {len(agents)} requests generated.",
            file=sys.stderr,
        )

    # ── Phase 2: Fan out ──────────────────────────────────────────────
    default_ms = parse_model_spec(model_spec)
    fanned_nodes = []
    fanned_labels = []

    for i, a in enumerate(agents):
        wprofile = worker_profiles[i] if i < len(worker_profiles) else None
        worker_model = a.model or default_ms.model
        if wprofile and wprofile.model:
            worker_model = wprofile.model
        w_effort = effort
        if wprofile and wprofile.effort and not effort:
            w_effort = wprofile.effort
        w_yolo = yolo
        if wprofile and wprofile.yolo:
            w_yolo = True
        worker_imodel = build_imodel_from_spec(
            worker_model,
            yolo=w_yolo,
            verbose=verbose,
            effort_override=w_effort,
            theme=theme,
        )
        if cwd:
            worker_imodel.endpoint.config.kwargs.setdefault("repo", Path(cwd))
        wname = worker_names[i]
        if team_data:
            teammates = [n for n in worker_names if n != wname]
            roster_lines = [f"- orchestrator (coordinator)"]
            roster_lines += [f"- {t}" for t in teammates]
            roster_lines.append(f"- **{wname}** (you)")
            worker_system = TEAM_WORKER_SYSTEM.format(
                worker_name=wname,
                team_name=team_data["name"],
                team_id=team_data["id"],
                roster_text="\n".join(roster_lines),
            )
        elif wprofile and wprofile.system_prompt:
            worker_system = wprofile.system_prompt
        else:
            worker_system = BARE_WORKER_SYSTEM
        worker_branch = Branch(
            chat_model=worker_imodel,
            system=worker_system,
            log_config=DataLoggerConfig(auto_save_on_exit=False),
            name=wname,
        )
        session.include_branches(worker_branch)
        worker_context = [
            {"overall_task": prompt},
            a.instruct.context or "",
        ]
        node = builder.add_operation(
            "operate",
            branch=worker_branch,
            depends_on=[root],
            instruction=a.instruct.instruction,
            guidance=a.instruct.guidance,
            context=worker_context,
        )
        fanned_nodes.append(node)
        fanned_labels.append(worker_model)

    if not verbose:
        labels = ", ".join(fanned_labels)
        print(
            f"Phase 2: Fanning out to {len(fanned_nodes)} workers: [{labels}]",
            file=sys.stderr,
        )

    t1 = time.monotonic()
    conc = max_concurrent if max_concurrent > 0 else len(fanned_nodes)
    result2 = await session.flow(
        builder.get_graph(),
        max_concurrent=conc,
        verbose=verbose,
    )
    t_fanout = time.monotonic() - t1

    # Collect results
    op_results = result2.get("operation_results", {})
    worker_results = []
    contexts = []
    for i, nid in enumerate(fanned_nodes):
        res = op_results.get(nid)
        response_text = str(res) if res is not None else "(no response)"
        worker_results.append(
            {
                "worker": i + 1,
                "model": fanned_labels[i],
                "response": response_text,
                "time_ms": t_fanout * 1000,
            }
        )
        contexts.append(response_text)

    if not verbose:
        print(f"Phase 2 done ({t_fanout:.1f}s).", file=sys.stderr)

    # ── Incremental save: persist worker results immediately ─────────
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for wr in worker_results:
            p = save_path / f"worker_{wr['worker']}.md"
            p.write_text(wr["response"])
        if not verbose:
            print(
                f"Saved {len(worker_results)} worker results to {save_path}",
                file=sys.stderr,
            )
    if _shared is not None:
        _shared["saved_workers"] = worker_results

    # ── Phase 3: Synthesis ────────────────────────────────────────────
    synthesis_result = None
    if with_synthesis and contexts:
        synth_spec = synthesis_model or model_spec
        synth_label = str(parse_model_spec(synth_spec))

        if not verbose:
            print(f"Phase 3: Synthesis [{synth_label}]...", file=sys.stderr)

        synth_instruction = synthesis_prompt or (
            f"Synthesize the following {len(contexts)} worker responses "
            f"into a cohesive analysis.\n\n"
            f"Original task: {prompt}"
        )

        synth_node = builder.add_operation(
            "operate",
            branch=orc_branch,
            depends_on=fanned_nodes,
            instruction=synth_instruction,
            context=contexts,
        )

        t2 = time.monotonic()
        result3 = await session.flow(builder.get_graph(), verbose=verbose)
        t_synth = time.monotonic() - t2

        synth_res = result3.get("operation_results", {}).get(synth_node)
        synthesis_result = {
            "model": synth_label,
            "response": str(synth_res) if synth_res is not None else "(no response)",
            "time_ms": t_synth * 1000,
        }

        if not verbose:
            print(f"Phase 3 done ({t_synth:.1f}s).", file=sys.stderr)

    # ── Output ────────────────────────────────────────────────────────
    if output_format == "json":
        output = _format_result_json(worker_results, synthesis_result)
    else:
        output = _format_result_text(worker_results, synthesis_result)

    # ── Save synthesis + meta (workers already saved incrementally) ──
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        if synthesis_result:
            (save_path / "synthesis.md").write_text(synthesis_result["response"])
        meta = {
            "prompt": prompt,
            "workers": fanned_labels,
            "synthesis_model": synthesis_result["model"] if synthesis_result else None,
            "total_time_ms": (time.monotonic() - t0) * 1000,
        }
        (save_path / "meta.json").write_text(json_dumps(meta))
        if not verbose:
            print(f"Saved to {save_path}", file=sys.stderr)

    # ── Post to team ─────────────────────────────────────────────────
    if team_data:
        _post_results_to_team(team_data, worker_results, worker_names, synthesis_result)
        if not verbose:
            print(
                f"\nTeam '{team_data['name']}' ({team_data['id']}): "
                f"{len(worker_results)} results posted.",
                file=sys.stderr,
            )
            print(
                f"  li team receive -t {team_data['id']} --as orchestrator",
                file=sys.stderr,
            )
            print(
                f"  li team show {team_data['id']}",
                file=sys.stderr,
            )

    # ── Persist all branches ─────────────────────────────────────────
    branch_ids = await persist_session_branches(session)
    orc_branch_id = str(orc_branch.id)
    save_last_branch_pointer(
        orc_branch.chat_model.endpoint.config.provider,
        orc_branch_id,
    )

    t_total = time.monotonic() - t0
    if not verbose:
        print(f"\nTotal: {t_total:.1f}s", file=sys.stderr)

    print(f'\n[orchestrator] li agent -r {orc_branch_id} "..."', file=sys.stderr)
    for provider, bid, bname in branch_ids:
        if bid != orc_branch_id:
            print(f'[{bname}]      li agent -r {bid} "..."', file=sys.stderr)

    return output
