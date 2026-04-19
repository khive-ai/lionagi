# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
"""Auto-DAG flow: orchestrator plans DAG → engine executes with deps."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from lionagi import Branch, FieldModel, Session, json_dumps
from lionagi._errors import TimeoutError as LionTimeoutError
from lionagi.ln.concurrency import move_on_after
from lionagi.models import HashableModel
from lionagi.operations.builder import OperationGraphBuilder
from lionagi.operations.fields import Instruct
from lionagi.protocols.generic.log import DataLoggerConfig

from .._agents import list_agents, load_agent_profile
from .._persistence import save_last_branch_pointer
from .._providers import build_imodel_from_spec, parse_model_spec
from ._common import (
    BARE_WORKER_SYSTEM,
    TEAM_WORKER_SYSTEM,
    _create_fanout_team,
    _format_result_json,
    _post_results_to_team,
    _resolve_worker_spec,
)

# ── Flow models ───────────────────────────────────────────────────────────


class FlowAgentSpec(HashableModel):
    """One agent in a DAG flow — dependencies expressed via depends_on."""

    id: str
    role: str
    instruction: str
    depends_on: list[str] | None = None
    control: bool = False
    guidance: str | None = None
    model: str | None = None


class FlowPlan(HashableModel):
    """DAG execution plan — agents reference each other via id/depends_on.

    Agents with control=true are flow control checkpoints. They receive
    all prior results and produce a structured verdict (should_continue).
    If should_continue=true, the orchestrator plans the next batch.
    """

    agents: list[FlowAgentSpec]
    synthesis: bool = False


FLOW_PLAN_FIELDS = FieldModel(FlowPlan, name="plan")


class FlowControlVerdict(HashableModel):
    """Structured output from a flow control node."""

    should_continue: bool
    reason: str
    next_steps: str | None = None


FLOW_VERDICT_FIELDS = FieldModel(FlowControlVerdict, name="verdict")


def _format_flow_result_text(
    agent_results: list[dict],
    synthesis_result: dict | None = None,
) -> str:
    lines = []
    for w in agent_results:
        deps = w.get("depends_on") or []
        dep_str = f"  deps: {', '.join(deps)}" if deps else ""
        lines.append(f"{'═' * 60}")
        lines.append(f"  {w['id']} ({w['name']})  [{w['model']}]{dep_str}")
        lines.append(f"  {w['time_ms']:.0f}ms")
        lines.append(f"{'═' * 60}")
        lines.append(w.get("response", "(no response)"))
        lines.append("")

    if synthesis_result is not None:
        lines.append(f"{'═' * 60}")
        lines.append(f"  Synthesis  [{synthesis_result['model']}]")
        lines.append(f"  {synthesis_result['time_ms']:.0f}ms")
        lines.append(f"{'═' * 60}")
        lines.append(synthesis_result.get("response", "(no response)"))
        lines.append("")

    return "\n".join(lines)


async def _run_flow(
    model_spec: str,
    prompt: str,
    *,
    with_synthesis: bool = False,
    synthesis_model: str | None = None,
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
    bare: bool = False,
    max_agents: int = 0,
    dry_run: bool = False,
    show_graph: bool = False,
) -> str:
    """Auto-DAG flow: orchestrator plans DAG → engine executes with deps."""
    if timeout:
        with move_on_after(timeout) as cancel_scope:
            result = await _run_flow_inner(
                model_spec,
                prompt,
                with_synthesis=with_synthesis,
                synthesis_model=synthesis_model,
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
                bare=bare,
                max_agents=max_agents,
                dry_run=dry_run,
                show_graph=show_graph,
            )
        if cancel_scope.cancelled_caught:
            raise LionTimeoutError(f"Flow timed out after {timeout}s")
        return result
    return await _run_flow_inner(
        model_spec,
        prompt,
        with_synthesis=with_synthesis,
        synthesis_model=synthesis_model,
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
        bare=bare,
        max_agents=max_agents,
        dry_run=dry_run,
        show_graph=show_graph,
    )


async def _run_flow_inner(
    model_spec: str,
    prompt: str,
    *,
    with_synthesis: bool = False,
    synthesis_model: str | None = None,
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
    bare: bool = False,
    max_agents: int = 0,
    dry_run: bool = False,
    show_graph: bool = False,
) -> str:
    """Inner flow logic (no timeout wrapper)."""
    t0 = time.monotonic()

    # Load orchestrator profile
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

    orc_imodel = build_imodel_from_spec(
        model_spec,
        yolo=yolo,
        verbose=verbose,
        effort_override=effort,
        theme=theme,
    )
    if cwd:
        orc_imodel.endpoint.config.kwargs.setdefault("repo", Path(cwd))

    # ── Phase 0: Orchestrator plans the DAG ──────────────────────────
    builder = OperationGraphBuilder("Flow")
    orc_system = orc_profile.system_prompt if orc_profile else None
    orc_branch = Branch(
        chat_model=orc_imodel,
        system=orc_system,
        log_config=DataLoggerConfig(auto_save_on_exit=False),
        name="orchestrator",
    )
    session = Session(default_branch=orc_branch)

    # Build role roster for orchestrator guidance
    available_roles = list_agents()
    if bare:
        roles_guidance = (
            f"Available roles: {', '.join(available_roles)}. "
            f"All workers use model {model_spec}. "
            f"Roles define behavioral focus only — model is fixed."
        )
    else:
        role_details = []
        for role in available_roles:
            try:
                rp = load_agent_profile(role)
                rm = rp.model or model_spec
                detail = f"{role} (model: {rm}"
                if rp.effort:
                    detail += f", effort: {rp.effort}"
                detail += ")"
                role_details.append(detail)
            except FileNotFoundError:
                role_details.append(f"{role} (model: {model_spec})")
        roles_guidance = f"Available agents: {'; '.join(role_details)}."

    budget_note = ""
    if max_agents > 0:
        budget_note = f"BUDGET: You may define at most {max_agents} agents total. "

    # Build guidance blocks for plan root
    artifact_guidance = ""
    if save_dir:
        artifact_guidance = (
            "ARTIFACT PROTOCOL: Each agent gets its own directory at "
            f"{save_dir}/{{agent_id}}/. In EVERY agent instruction you MUST specify: "
            "(1) WHERE to write: 'Write output to your current directory as <name>.md'. "
            "(2) WHAT to name files: descriptive names (inventory.md, gap_analysis.md). "
            "(3) WHERE to read upstream: 'Read ../{{dep_id}}/{{filename}}.md' for each depends_on. "
            "NEVER say 'use the prior output' — always give explicit relative paths. "
        )

    effort_guidance = (
        "EFFORT TIERS: Use the agent 'guidance' field for behavioral framing. "
        "low=skim structure quickly; medium=careful read; high=thorough analysis; "
        "xhigh=deep multi-step reasoning. Match effort to task weight. "
    )

    team_guidance = ""
    if team_name:
        team_guidance = (
            f"TEAM MODE active (team: {team_name}). In each agent instruction, "
            "tell agents to check inbox before starting and send coordination "
            "signals to relevant teammates if they discover something affecting them. "
        )

    plan_root = builder.add_operation(
        "instruct",
        branch=orc_branch,
        instruct=Instruct(
            instruction=(
                "Create a FlowPlan for the following task. "
                "Define agents as a flat list with dependency edges via depends_on. "
                "Each agent needs a short id (e.g. 'r1', 'a1'), a role, a concrete "
                "instruction, and optional depends_on listing ids of agents whose "
                "output it needs. Agents with no depends_on run immediately in parallel."
            ),
            context={"task": prompt},
            guidance=(
                f"{roles_guidance} "
                f"{budget_note}"
                f"{artifact_guidance}"
                f"{effort_guidance}"
                f"{team_guidance}"
                "CRITICAL: You MUST produce your output ONLY via the structured "
                "output fields (the FlowPlan). Do NOT use any provider-native "
                "subagent or tool-spawning features (no Agent tool, no subprocess "
                "spawning, no delegation tools). The ONLY correct way to define "
                "workers is by filling in the FlowPlan structured output. "
                "Use ONLY roles from the available list above — do not invent "
                "custom role names. "
                "Keep the agent count minimal — only add agents that provide "
                "distinct value. Prefer fewer agents with clear instructions "
                "over many agents with overlapping scope. "
                "CONTROL NODES: Set control=true on an agent to make it a "
                "flow control checkpoint. Control agents (typically critics) "
                "review prior work and decide whether the flow should continue "
                "with additional rounds. Place at most one control node, "
                "and it should depend on the agents whose work it reviews. "
                "Set synthesis=true if the task benefits from a final consolidated output."
            ),
        ),
        field_models=[FLOW_PLAN_FIELDS],
        reason=True,
    )

    if not verbose:
        print("Planning DAG...", file=sys.stderr)

    result0 = await session.flow(builder.get_graph())
    t_plan = time.monotonic() - t0

    plan_result = result0.get("operation_results", {}).get(plan_root)
    plan: FlowPlan | None = getattr(plan_result, "plan", None)

    if not plan or not plan.agents:
        return "Orchestrator produced no flow plan."

    if max_agents > 0 and len(plan.agents) > max_agents:
        plan.agents = plan.agents[:max_agents]
        if not verbose:
            print(
                f"Plan truncated to {max_agents} agents (--max-agents)", file=sys.stderr
            )

    if not verbose:
        dag_lines = []
        for a in plan.agents:
            deps = f" ← {','.join(a.depends_on)}" if a.depends_on else ""
            dag_lines.append(f"{a.id}:{a.role}{deps}")
        print(
            f"Plan done ({t_plan:.1f}s): {len(plan.agents)} agents: "
            f"{' | '.join(dag_lines)}",
            file=sys.stderr,
        )

    if plan.synthesis and not with_synthesis:
        with_synthesis = True

    # ── Dry run: dump plan and exit ─────────────────────────────────
    if dry_run:
        lines = [f"FlowPlan ({len(plan.agents)} agents, synthesis={plan.synthesis})"]
        lines.append("")
        for a in plan.agents:
            ctrl = " [CONTROL]" if a.control else ""
            deps = f"  depends_on: {', '.join(a.depends_on)}" if a.depends_on else ""
            model_note = f"  model: {a.model}" if a.model else ""
            lines.append(f"  {a.id}: {a.role}{ctrl}")
            lines.append(f"    instruction: {a.instruction[:120]}...")
            if deps:
                lines.append(deps)
            if model_note:
                lines.append(model_note)
            if a.guidance:
                lines.append(f"    guidance: {a.guidance[:80]}...")
            lines.append("")

        # Show resolved models
        lines.append("Model resolution:")
        for a in plan.agents:
            if bare:
                rm = a.model or model_spec
                lines.append(f"  {a.id}: {rm} (bare)")
            else:
                rm, rp = _resolve_worker_spec(a.role)
                if a.model:
                    rm = a.model
                src = "plan" if a.model else ("profile" if rp else "default")
                lines.append(f"  {a.id}: {rm} ({src})")

        if show_graph:
            from lionagi.operations._visualize_graph import visualize_graph
            graph_save = None
            if save_dir:
                graph_save = str(Path(save_dir) / "flow_dag.png")
            visualize_graph(
                builder,
                title=f"Flow DAG — {len(plan.agents)} agents",
                save_path=graph_save,
            )

        return "\n".join(lines)

    # ── Build agent name registry ───────────────────────────────────
    default_ms = parse_model_spec(model_spec)
    all_agent_names: list[str] = []
    name_counts: dict[str, int] = {}
    for a in plan.agents:
        base = a.role
        name_counts[base] = name_counts.get(base, 0) + 1
        if name_counts[base] > 1:
            all_agent_names.append(f"{base}-{name_counts[base]}")
        else:
            all_agent_names.append(base)

    team_data = None
    if team_name:
        team_data = _create_fanout_team(team_name, all_agent_names)
        if not verbose:
            print(f"Team '{team_name}' created ({team_data['id']})", file=sys.stderr)

    # ── Helper: resolve model/system/branch for an agent spec ─────
    def _make_worker_branch(a: FlowAgentSpec, idx: int) -> tuple[Branch, str]:
        w_profile = None
        if bare:
            w_model = a.model or default_ms.model
        else:
            w_model, w_profile = _resolve_worker_spec(a.role)
            if a.model:
                w_model = a.model
            elif not w_profile:
                w_model = default_ms.model
        w_effort = effort
        if not bare and w_profile and w_profile.effort and not effort:
            w_effort = w_profile.effort
        w_yolo = yolo
        if not bare and w_profile and w_profile.yolo:
            w_yolo = True
        w_imodel = build_imodel_from_spec(
            w_model,
            yolo=w_yolo,
            verbose=verbose,
            effort_override=w_effort,
            theme=theme,
        )
        # Per-agent artifact directory: {save_dir}/{agent_id}/
        agent_cwd = cwd
        if save_dir:
            agent_artifact_dir = Path(save_dir) / a.id
            agent_artifact_dir.mkdir(parents=True, exist_ok=True)
            agent_cwd = str(agent_artifact_dir)
        if agent_cwd:
            w_imodel.endpoint.config.kwargs["repo"] = Path(agent_cwd)
        wname = all_agent_names[idx]
        w_system = None
        if team_data:
            teammates = [n for n in all_agent_names if n != wname]
            roster_lines = ["- orchestrator (coordinator)"]
            roster_lines += [f"- {t}" for t in teammates]
            roster_lines.append(f"- **{wname}** (you)")
            w_system = TEAM_WORKER_SYSTEM.format(
                worker_name=wname,
                team_name=team_data["name"],
                team_id=team_data["id"],
                roster_text="\n".join(roster_lines),
            )
        elif not bare and w_profile and w_profile.system_prompt:
            w_system = w_profile.system_prompt
        else:
            w_system = BARE_WORKER_SYSTEM
        wb = Branch(
            chat_model=w_imodel,
            system=w_system,
            log_config=DataLoggerConfig(auto_save_on_exit=False),
            name=wname,
        )
        session.include_branches(wb)
        return wb, w_model

    # ── Build DAG: split regular agents and control nodes ──────────
    spec_to_node: dict[str, str] = {}
    agent_meta: dict[str, dict] = {}
    agent_results: list[dict] = []
    regular_agents = [a for a in plan.agents if not a.control]
    control_agents = [a for a in plan.agents if a.control]

    if not verbose:
        ctrl_note = f" ({len(control_agents)} control)" if control_agents else ""
        print(
            f"Executing DAG: {len(regular_agents)} agents{ctrl_note}...",
            file=sys.stderr,
        )

    # Build regular agent nodes
    for idx, a in enumerate(plan.agents):
        if a.control:
            continue
        w_branch, w_model = _make_worker_branch(a, idx)
        dep_nodes = []
        if a.depends_on:
            for dep_id in a.depends_on:
                if dep_id in spec_to_node:
                    dep_nodes.append(spec_to_node[dep_id])
        if not dep_nodes:
            dep_nodes = [plan_root]
        ctx = [{"original_task": prompt}]
        if save_dir:
            artifact_note = (
                f"Your artifact directory: {Path(save_dir) / a.id}/ — "
                "write ALL output files here with descriptive names."
            )
            if a.depends_on:
                dep_notes = [f"{d}: {Path(save_dir) / d}/ (read .md files there)" for d in a.depends_on]
                artifact_note += f" Upstream artifacts: {'; '.join(dep_notes)}."
            ctx.append({"artifact_instructions": artifact_note})
        if team_data:
            ctx.append({"team": {"id": team_data["id"], "name": team_data["name"], "your_name": all_agent_names[idx]}})
        w_effort = effort
        if not bare and w_profile and w_profile.effort:
            w_effort = w_profile.effort
        if w_effort:
            emap = {"low": "Skim quickly, structured output.", "medium": "Read carefully, balance depth/speed.",
                    "high": "Thorough analysis, take your time.", "xhigh": "Deep reasoning, maximum effort."}
            ctx.append({"effort_guidance": emap.get(w_effort, "")})

        node_id = builder.add_operation(
            "instruct",
            branch=w_branch,
            depends_on=dep_nodes,
            instruction=a.instruction,
            guidance=a.guidance,
            context=ctx,
        )
        spec_to_node[a.id] = node_id
        agent_meta[node_id] = {
            "name": all_agent_names[idx],
            "model": w_model,
            "spec_id": a.id,
            "depends_on": a.depends_on or [],
        }

    # Progress callback for real-time status
    def _progress(op_id, name, status, elapsed):
        if not verbose:
            if status == "started":
                print(f"  ▶ {name} started", file=sys.stderr)
            elif status == "completed":
                print(f"  ✓ {name} done ({elapsed:.1f}s)", file=sys.stderr)
            elif status == "failed":
                print(f"  ✗ {name} FAILED ({elapsed:.1f}s)", file=sys.stderr)

    # Execute regular agents
    t_exec = time.monotonic()
    conc = max_concurrent if max_concurrent > 0 else max(len(regular_agents), 1)
    dag_result = await session.flow(
        builder.get_graph(),
        max_concurrent=conc,
        verbose=verbose,
        on_progress=_progress,
    )
    t_exec_elapsed = time.monotonic() - t_exec

    op_results = dag_result.get("operation_results", {})
    for a in regular_agents:
        nid = spec_to_node[a.id]
        meta = agent_meta[nid]
        res = op_results.get(nid)
        agent_results.append(
            {
                "id": a.id,
                "name": meta["name"],
                "model": meta["model"],
                "depends_on": meta["depends_on"],
                "control": False,
                "response": str(res) if res is not None else "(no response)",
                "time_ms": t_exec_elapsed * 1000,
            }
        )

    if not verbose:
        print(f"DAG done ({t_exec_elapsed:.1f}s).", file=sys.stderr)

    # ── Execute control nodes: instruct with prior results ─────────
    max_rounds = 3
    round_num = 0
    for ca in control_agents:
        idx = next(i for i, a in enumerate(plan.agents) if a.id == ca.id)
        c_branch, c_model = _make_worker_branch(ca, idx)

        artifacts = [
            f"[{r['id']} ({r['name']})]: {r['response']}" for r in agent_results
        ]
        dep_nodes = []
        if ca.depends_on:
            for dep_id in ca.depends_on:
                if dep_id in spec_to_node:
                    dep_nodes.append(spec_to_node[dep_id])
        if not dep_nodes:
            all_node_ids = list(spec_to_node.values())
            dep_nodes = all_node_ids[-1:] if all_node_ids else [plan_root]

        if not verbose:
            print(f"Control [{ca.id}]: evaluating...", file=sys.stderr)

        artifact_read_note = ""
        if save_dir and ca.depends_on:
            dep_dirs = [f"{save_dir}/{d}/" for d in ca.depends_on]
            artifact_read_note = (
                "\n\nREAD ARTIFACTS: Agents wrote files to these directories. "
                f"Read ALL .md files before reaching your verdict: {', '.join(dep_dirs)}"
            )

        ctrl_node = builder.add_operation(
            "instruct",
            branch=c_branch,
            depends_on=dep_nodes,
            instruction=(
                f"{ca.instruction}\n\n"
                "Review all prior agent outputs and produce a verdict: "
                "should_continue (bool), reason (str), and optional "
                "next_steps (str) guidance if continuing."
                f"{artifact_read_note}"
                "\n\nVERDICT CONSEQUENCES: "
                "If should_continue=False, the flow ends. "
                "If should_continue=True, the orchestrator plans ADDITIONAL targeted agents "
                "to address your next_steps — a surgical re-plan, not a full restart. "
                "Be specific in next_steps: name EXACT gaps, not 'improve quality'."
            ),
            guidance=ca.guidance,
            context=[{"original_task": prompt}, {"agent_outputs": artifacts}],
            field_models=[FLOW_VERDICT_FIELDS],
            reason=True,
        )
        spec_to_node[ca.id] = ctrl_node

        t_ctrl = time.monotonic()
        ctrl_result = await session.flow(builder.get_graph(), verbose=verbose)
        t_ctrl_elapsed = time.monotonic() - t_ctrl

        ctrl_res = ctrl_result.get("operation_results", {}).get(ctrl_node)
        verdict: FlowControlVerdict | None = getattr(ctrl_res, "verdict", None)
        verdict_text = str(ctrl_res) if ctrl_res is not None else "(no response)"

        agent_results.append(
            {
                "id": ca.id,
                "name": all_agent_names[idx],
                "model": c_model,
                "depends_on": ca.depends_on or [],
                "control": True,
                "response": verdict_text,
                "time_ms": t_ctrl_elapsed * 1000,
            }
        )

        if not verbose:
            cont = verdict.should_continue if verdict else False
            print(
                f"Control [{ca.id}] done ({t_ctrl_elapsed:.1f}s): " f"continue={cont}",
                file=sys.stderr,
            )

        # ── If verdict says continue: orchestrator re-plans ────────
        if verdict and verdict.should_continue:
            round_num += 1
            if round_num >= max_rounds:
                if not verbose:
                    print(
                        f"Max rounds ({max_rounds}) reached, stopping.", file=sys.stderr
                    )
                break

            if not verbose:
                print(
                    f"Round {round_num + 1}: orchestrator re-planning...",
                    file=sys.stderr,
                )

            replan_node = builder.add_operation(
                "instruct",
                branch=orc_branch,
                depends_on=[ctrl_node],
                instruct=Instruct(
                    instruction=(
                        "The control node requested continuation. Plan additional agents "
                        "to address the feedback. Return a FlowPlan with new agents only."
                    ),
                    context={
                        "original_task": prompt,
                        "prior_results": artifacts,
                        "control_verdict": verdict_text,
                        "next_steps_guidance": verdict.next_steps or "",
                    },
                    guidance=(
                        f"{roles_guidance} "
                        f"This is round {round_num + 1}. "
                        "Add only the agents needed to address the control feedback. "
                        "Do NOT re-run agents that already succeeded. "
                        "Use ONLY the structured output. Do NOT spawn subagents."
                    ),
                ),
                field_models=[FLOW_PLAN_FIELDS],
                reason=True,
            )

            replan_result = await session.flow(builder.get_graph(), verbose=verbose)
            replan_res = replan_result.get("operation_results", {}).get(replan_node)
            next_plan: FlowPlan | None = getattr(replan_res, "plan", None)

            if next_plan and next_plan.agents:
                if not verbose:
                    ids = ", ".join(a.id for a in next_plan.agents)
                    print(
                        f"Re-plan: {len(next_plan.agents)} new agents: {ids}",
                        file=sys.stderr,
                    )

                # Build and execute new agents
                new_names = []
                for na in next_plan.agents:
                    base = na.role
                    name_counts[base] = name_counts.get(base, 0) + 1
                    new_names.append(f"{base}-{name_counts[base]}")
                all_agent_names.extend(new_names)

                for ni, na in enumerate(next_plan.agents):
                    nb, nm = _make_worker_branch(
                        na, len(all_agent_names) - len(new_names) + ni
                    )
                    nd = []
                    if na.depends_on:
                        for did in na.depends_on:
                            if did in spec_to_node:
                                nd.append(spec_to_node[did])
                    if not nd:
                        nd = [ctrl_node]
                    round_ctx = [
                        {"original_task": prompt},
                        {"critic_verdict": verdict_text},
                        {"critic_next_steps": verdict.next_steps or ""},
                    ]
                    if save_dir:
                        round_ctx.append({
                            "artifact_note": (
                                f"Your directory: {Path(save_dir) / na.id}/. "
                                f"Prior artifacts in {save_dir}/*/. "
                                "Read critic verdict and fix what was flagged."
                            )
                        })
                    nid = builder.add_operation(
                        "instruct",
                        branch=nb,
                        depends_on=nd,
                        instruction=na.instruction,
                        guidance=na.guidance,
                        context=round_ctx,
                    )
                    spec_to_node[na.id] = nid
                    agent_meta[nid] = {
                        "name": new_names[ni],
                        "model": nm,
                        "spec_id": na.id,
                        "depends_on": na.depends_on or [],
                    }

                t_new = time.monotonic()
                new_result = await session.flow(
                    builder.get_graph(),
                    max_concurrent=conc,
                    verbose=verbose,
                )
                t_new_elapsed = time.monotonic() - t_new
                new_op = new_result.get("operation_results", {})
                for na in next_plan.agents:
                    nid = spec_to_node[na.id]
                    meta = agent_meta[nid]
                    res = new_op.get(nid)
                    agent_results.append(
                        {
                            "id": na.id,
                            "name": meta["name"],
                            "model": meta["model"],
                            "depends_on": meta["depends_on"],
                            "control": False,
                            "response": (
                                str(res) if res is not None else "(no response)"
                            ),
                            "time_ms": t_new_elapsed * 1000,
                        }
                    )
                if not verbose:
                    print(
                        f"Round {round_num + 1} done ({t_new_elapsed:.1f}s).",
                        file=sys.stderr,
                    )

    # ── Synthesis ────────────────────────────────────────────────────
    synthesis_result = None
    if with_synthesis and agent_results:
        synth_spec = synthesis_model or model_spec
        synth_label = str(parse_model_spec(synth_spec))
        if not verbose:
            print(f"Synthesis [{synth_label}]...", file=sys.stderr)

        all_node_ids = set(spec_to_node.values())
        depended_on = set()
        for a in plan.agents:
            for d in a.depends_on or []:
                if d in spec_to_node:
                    depended_on.add(spec_to_node[d])
        leaf_nodes = list(all_node_ids - depended_on) or list(all_node_ids)

        artifacts = [
            f"[{r['id']} ({r['name']})]: {r['response']}" for r in agent_results
        ]
        artifact_chain_note = ""
        if save_dir:
            adirs = [f"{save_dir}/{a.id}/" for a in plan.agents]
            artifact_chain_note = (
                f"\n\nARTIFACT CHAIN: Read ALL files in: {', '.join(adirs)}. "
                "Trace how work flowed through the DAG."
            )
        team_synth_note = ""
        if team_data:
            team_synth_note = (
                f"\n\nTEAM MESSAGES: Review inter-agent messages (team {team_data['id']}) "
                "for coordination context not captured in artifacts."
            )

        synth_node = builder.add_operation(
            "instruct",
            branch=orc_branch,
            depends_on=leaf_nodes,
            instruction=(
                f"Synthesize all agent outputs into a final cohesive deliverable.\n\n"
                f"Original task: {prompt}\n\n"
                "Your synthesis must:\n"
                "1. RECONCILE: When agents disagree, present both views with evidence.\n"
                "2. FILL GAPS: Name what no agent covered.\n"
                "3. TRACE: Show how work flowed through the DAG.\n"
                "4. HONOR CRITIC: If a critic was in the pipeline, its verdict is authoritative.\n"
                "5. RESUME: End with branch IDs so the user can follow up with any agent."
                f"{artifact_chain_note}"
                f"{team_synth_note}"
            ),
            context=artifacts,
        )
        t_synth = time.monotonic()
        synth_result = await session.flow(builder.get_graph(), verbose=verbose)
        t_synth_elapsed = time.monotonic() - t_synth
        synth_res = synth_result.get("operation_results", {}).get(synth_node)
        synthesis_result = {
            "model": synth_label,
            "response": str(synth_res) if synth_res is not None else "(no response)",
            "time_ms": t_synth_elapsed * 1000,
        }
        if not verbose:
            print(f"Synthesis done ({t_synth_elapsed:.1f}s).", file=sys.stderr)

    # ── Output ───────────────────────────────────────────────────────
    if output_format == "json":
        output = _format_result_json(agent_results, synthesis_result)
    else:
        output = _format_flow_result_text(agent_results, synthesis_result)

    # ── Save ─────────────────────────────────────────────────────────
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for r in agent_results:
            p = save_path / f"{r['id']}_{r['name']}.md"
            p.write_text(r["response"])
        if synthesis_result:
            (save_path / "synthesis.md").write_text(synthesis_result["response"])
        if not verbose:
            print(f"Saved to {save_path}", file=sys.stderr)

    if team_data:
        _post_results_to_team(
            team_data, agent_results, all_agent_names, synthesis_result
        )

    # ── Persist branches ─────────────────────────────────────────────
    from ._common import persist_session_branches

    branch_ids = await persist_session_branches(session)
    orc_branch_id = str(orc_branch.id)
    save_last_branch_pointer(
        orc_branch.chat_model.endpoint.config.provider,
        orc_branch_id,
    )

    if show_graph:
        from lionagi.operations._visualize_graph import visualize_graph
        graph_save = None
        if save_dir:
            graph_save = str(Path(save_dir) / "flow_dag.png")
        visualize_graph(
            builder,
            title=f"Flow DAG — {len(plan.agents)} agents (completed)",
            save_path=graph_save,
        )

    t_total = time.monotonic() - t0
    if not verbose:
        print(f"\nTotal: {t_total:.1f}s", file=sys.stderr)

    print(f'\n[orchestrator] li agent -r {orc_branch_id} "..."', file=sys.stderr)
    for provider, bid, bname in branch_ids:
        if bid != orc_branch_id:
            print(f'[{bname}] li agent -r {bid} "..."', file=sys.stderr)

    return output
