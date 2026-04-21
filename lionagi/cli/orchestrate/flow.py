# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
"""Auto-DAG flow: orchestrator plans DAG → engine executes with deps."""

from __future__ import annotations

import time
from typing import ClassVar, Literal

from pydantic import Field

from lionagi import Branch, FieldModel
from lionagi._errors import TimeoutError as LionTimeoutError
from lionagi.ln.concurrency import move_on_after
from lionagi.models import HashableModel
from lionagi.operations.fields import Instruct
from lionagi.protocols.generic.pile import Pile
from lionagi.protocols.messages.instruction import Instruction, InstructionContent

from .._agents import AgentProfile, list_agents, load_agent_profile
from .._logging import progress
from .._providers import parse_model_spec
from ._common import _create_fanout_team, _format_result_json, _post_results_to_team
from ._orchestration import (
    OrchestrationEnv,
    build_worker_branch,
    finalize_orchestration,
    resolve_worker_spec,
    setup_orchestration,
    team_guidance,
)

# ── Flow models ───────────────────────────────────────────────────────────


class FlowAgent(HashableModel):
    """An agent in a flow — a Branch identity with persistent memory.

    Agents are defined once and can be invoked multiple times in the DAG.
    Every invocation on the same agent reuses the same Branch, which
    means the agent remembers everything it did before. Use this for
    iterative refinement (r1 → impl1 → r1 again sees r1's first turn).
    """

    id: str = Field(
        description=(
            "Short unique identifier for this agent, e.g. 'r1', 'impl1'. "
            "Reused across multiple FlowOp.agent_id references."
        ),
    )
    role: str = Field(
        description=(
            "Role name from the available-agents roster (e.g. 'researcher', "
            "'implementer', 'critic'). Determines the agent's profile "
            "(system prompt, default model, effort). Do not invent roles."
        ),
    )
    model: str | None = Field(
        default=None,
        description=(
            "Explicit model spec override (e.g. 'codex/gpt-5.4-high'). "
            "Leave null to use the role's profile default."
        ),
    )
    guidance: str | None = Field(
        default=None,
        description=(
            "Default behavioral guidance applied to every op on this agent. "
            "Op-level guidance overrides this when set."
        ),
    )


class FlowOp(HashableModel):
    """One DAG node — a single invocation on some agent.

    ``agent_id`` must reference a FlowAgent in the same FlowPlan. Multiple
    ops can share an agent_id; the framework reuses the Branch so the
    agent's conversation history carries across.
    """

    id: str = Field(
        description=(
            "Short unique op id, e.g. 'o1', 'review1'. Referenced by other ops via depends_on."
        ),
    )
    agent_id: str = Field(
        description=(
            "The id of the FlowAgent that executes this op. Multiple ops "
            "can share an agent_id — the second invocation has memory of "
            "the first. Reusing an existing agent is CHEAPER than spawning "
            "a new one (no re-context, less tokens)."
        ),
    )
    instruction: str = Field(
        description=(
            "Concrete task instruction for this invocation. Must specify "
            "what to write and where (e.g. 'Write an inventory.md in your "
            "artifact dir listing all API endpoints')."
        ),
    )
    guidance: str | None = Field(
        default=None,
        description=(
            "Per-op behavioral framing. Overrides the agent's default "
            "guidance when set (e.g. 'skim quickly' vs 'deep analysis')."
        ),
    )
    depends_on: list[str] | None = Field(
        default=None,
        description=(
            "Other FlowOp ids this op waits on. Results from those ops "
            "are available as upstream context. Same-agent deps are free "
            "(branch already has memory); cross-agent deps require the "
            "downstream agent to read the upstream's artifact dir."
        ),
    )
    control_type: Literal["iterate", "gate", "quorum", "halt"] | None = Field(
        default=None,
        description=(
            "Control semantic: 'iterate' for critic checkpoints that may "
            "trigger re-planning, 'gate' for artifact checks, 'quorum' "
            "for completion thresholds, 'halt' for flow-wide pause. "
            "None = regular work op."
        ),
    )


class FlowPlan(HashableModel):
    """Two-level DAG plan: agent identities + operation DAG.

    Agents are the branches (who); operations are the DAG nodes (what
    happens, in what order, to which branch). An operation with
    ``control_type`` set acts as a control checkpoint — 'iterate' for
    critic-driven re-planning, 'gate' for artifact checks, 'quorum'
    for completion thresholds.
    """

    agents: list[FlowAgent] = Field(
        description=(
            "Persistent agent identities (Branches). Keep count minimal — "
            "reusing an agent across ops is cheaper than spawning a new "
            "one. Spawn new only for fresh perspective or different role."
        ),
    )
    operations: list[FlowOp] = Field(
        description=(
            "DAG of invocations. Each op picks its executing agent via "
            "agent_id and declares upstream deps via depends_on. Ops form "
            "an acyclic graph; independent ops run in parallel."
        ),
    )
    synthesis: bool = Field(
        default=False,
        description=(
            "Set True if the task benefits from a final consolidated "
            "synthesis op after all others complete."
        ),
    )

    # ── Prompt templates consumed by the orchestrator planner ──────────
    PLANNING_INSTRUCTION: ClassVar[str] = (
        "Produce a FlowPlan with TWO levels:\n"
        "  1. agents: list of FlowAgent — each is a persistent Branch "
        "     (identity + memory). Same agent can run multiple ops.\n"
        "  2. operations: list of FlowOp — DAG of invocations. Each op "
        "     picks which agent runs it via agent_id.\n\n"
        "Example (research → implement → research re-reviews):\n"
        "  agents: [\n"
        "    FlowAgent(id='r1', role='researcher'),\n"
        "    FlowAgent(id='i1', role='implementer'),\n"
        "  ]\n"
        "  operations: [\n"
        "    FlowOp(id='o1', agent_id='r1', instruction='research X'),\n"
        "    FlowOp(id='o2', agent_id='i1', instruction='implement based on o1', depends_on=['o1']),\n"
        "    FlowOp(id='o3', agent_id='r1', instruction='review i1 work', depends_on=['o2']),\n"
        "  ]\n"
        "Because o3 reuses r1, r1 remembers its own research from o1 — "
        "no need to re-inject context. Reusing agents is cheaper than "
        "spawning new ones."
    )

    PLANNING_DISCIPLINE: ClassVar[str] = (
        "Output ONLY via FlowPlan structured fields. Do NOT use "
        "provider-native subagent/tool-spawning features. "
        "Use ONLY roles from the available list — do not invent names. "
        "Reuse agents across ops (cheaper than spawning new). "
        "Set op.control_type='iterate' for critic checkpoints. "
        "Set synthesis=true for a final consolidated output."
    )

    REPLAN_INSTRUCTION: ClassVar[str] = (
        "The control op requested continuation. Return a FlowPlan with:\n"
        "  - agents: ONLY new agents you need (reuse existing when possible — "
        "    list only the new ones here).\n"
        "  - operations: the NEW ops to run. You may reference existing "
        "    agents (by their original id) or your new agents.\n"
        "Reusing an existing agent is cheaper — its branch already has "
        "memory from its prior turns."
    )

    REPLAN_GUIDANCE: ClassVar[str] = (
        "Add only ops needed to address the control feedback. "
        "Do NOT re-run ops that already succeeded. "
        "Use ONLY the structured output. Do NOT spawn subagents."
    )


FLOW_PLAN_FIELDS = FieldModel(FlowPlan, name="plan")


from lionagi.operations.control import ControlDecision

CONTROL_VERDICT_CONTRACT: str = (
    "Produce a ControlDecision: action ('proceed' to end flow, "
    "'iterate' to request additional ops), reason (str), and "
    "metadata.next_steps (str) if iterating. Be specific."
)

FLOW_VERDICT_FIELDS = FieldModel(ControlDecision, name="verdict")


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



def _build_agent_results(op_meta: dict, op_to_node: dict, all_op_results: dict) -> list[dict]:
    """Build display-ready results list from op_meta + accumulated results."""
    results = []
    for op_id, meta in op_meta.items():
        nid = op_to_node.get(op_id)
        res = all_op_results.get(nid) if nid else None
        results.append({
            "id": op_id,
            "agent_id": meta["agent_id"],
            "name": meta["agent_name"],
            "model": meta["model"],
            "depends_on": meta.get("depends_on", []),
            "control_type": meta.get("control_type"),
            "response": str(res) if res is not None else "(no response)",
        })
    return results


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
    planning_guidance: str | None = None,
    synthesis_guidance: str | None = None,
) -> str:
    """Auto-DAG flow: orchestrator plans DAG → engine executes with deps."""
    env = setup_orchestration(
        pattern_name="Flow",
        model_spec=model_spec,
        agent_name=agent_name,
        save_dir=save_dir,
        cwd=cwd,
        yolo=yolo,
        verbose=verbose,
        effort=effort,
        theme=theme,
        bare=bare,
    )

    inner_kw = dict(
        env=env,
        with_synthesis=with_synthesis,
        synthesis_model=synthesis_model,
        max_concurrent=max_concurrent,
        output_format=output_format,
        team_name=team_name,
        max_agents=max_agents,
        dry_run=dry_run,
        show_graph=show_graph,
        planning_guidance=planning_guidance,
        synthesis_guidance=synthesis_guidance,
    )
    if timeout:
        with move_on_after(timeout) as cancel_scope:
            result = await _run_flow_inner(model_spec, prompt, **inner_kw)
        if cancel_scope.cancelled_caught:
            raise LionTimeoutError(f"Flow timed out after {timeout}s")
        return result
    return await _run_flow_inner(model_spec, prompt, **inner_kw)


async def _run_flow_inner(
    model_spec: str,
    prompt: str,
    *,
    env: OrchestrationEnv,
    with_synthesis: bool = False,
    synthesis_model: str | None = None,
    max_concurrent: int = 0,
    output_format: str = "text",
    team_name: str | None = None,
    max_agents: int = 0,
    dry_run: bool = False,
    show_graph: bool = False,
    planning_guidance: str | None = None,
    synthesis_guidance: str | None = None,
) -> str:
    """Inner flow logic (no timeout wrapper)."""
    t0 = time.monotonic()

    # Working objects: four subjects this function mutates across phases.
    # All config (bare/verbose/effort/yolo/theme/cwd) is read from `env`
    # directly — no aliases, one source of truth.
    run = env.run
    session = env.session
    orc_branch = env.orc_branch
    builder = env.builder

    # ── Phase 0: Orchestrator plans the DAG ──────────────────────────
    available_roles = list_agents()
    roles_guidance = f"Available roles: {', '.join(available_roles)}."

    budget_note = ""
    if max_agents > 0:
        budget_note = f"BUDGET: at most {max_agents} agents total. "

    guidance_parts = [
        roles_guidance,
        budget_note,
        team_guidance(team_name),
        FlowPlan.PLANNING_DISCIPLINE,
    ]
    if planning_guidance:
        guidance_parts.append(planning_guidance)

    plan_root = builder.add_operation(
        "operate",
        branch=orc_branch,
        instruct=Instruct(
            instruction=FlowPlan.PLANNING_INSTRUCTION,
            context={"task": prompt},
            guidance=" ".join(g for g in guidance_parts if g),
        ),
        field_models=[FLOW_PLAN_FIELDS],
        reason=True,
    )

    progress("Planning DAG...")

    result0 = await session.flow(builder.get_graph())
    t_plan = time.monotonic() - t0

    plan_result = result0.get("operation_results", {}).get(plan_root)
    plan: FlowPlan | None = getattr(plan_result, "plan", None)

    if not plan or not plan.agents or not plan.operations:
        return "Orchestrator produced no flow plan."

    # Validate op.agent_id references resolve to a defined agent
    agent_ids = {a.id for a in plan.agents}
    for op in plan.operations:
        if op.agent_id not in agent_ids:
            return (
                f"Invalid plan: op {op.id!r} references unknown agent "
                f"{op.agent_id!r} (known: {sorted(agent_ids)})"
            )

    # max_agents caps the total operation count — that is the real work
    # budget. Agent count is bounded implicitly by the op count since
    # every agent runs at least one op.
    if max_agents > 0 and len(plan.operations) > max_agents:
        plan.operations = plan.operations[:max_agents]
        progress(f"Plan truncated to {max_agents} operations (--max-agents)")

    dag_lines = []
    for op in plan.operations:
        deps = f" ← {','.join(op.depends_on)}" if op.depends_on else ""
        ctrl = f"!{op.control_type}" if op.control_type else ""
        dag_lines.append(f"{op.id}{ctrl}:{op.agent_id}{deps}")
    progress(
        f"Plan done ({t_plan:.1f}s): {len(plan.agents)} agents, "
        f"{len(plan.operations)} ops — {' | '.join(dag_lines)}"
    )

    if plan.synthesis and not with_synthesis:
        with_synthesis = True

    # ── Dry run: dump plan and exit ─────────────────────────────────
    if dry_run:
        lines = [
            f"FlowPlan ({len(plan.agents)} agents, {len(plan.operations)} ops, "
            f"synthesis={plan.synthesis})",
            "",
            "Agents:",
        ]
        for a in plan.agents:
            lines.append(f"  {a.id}: {a.role}")
            if a.model:
                lines.append(f"    model: {a.model}")
            if a.guidance:
                lines.append(f"    guidance: {a.guidance[:80]}...")
        lines.append("")
        lines.append("Operations:")
        for op in plan.operations:
            ctrl = f" [{op.control_type.upper()}]" if op.control_type else ""
            deps = f"  depends_on: {', '.join(op.depends_on)}" if op.depends_on else ""
            lines.append(f"  {op.id} → {op.agent_id}{ctrl}")
            lines.append(f"    instruction: {op.instruction[:120]}...")
            if deps:
                lines.append(deps)
            if op.guidance:
                lines.append(f"    guidance: {op.guidance[:80]}...")
            lines.append("")

        # Show resolved models per agent
        lines.append("Model resolution:")
        for a in plan.agents:
            if env.bare:
                rm = a.model or model_spec
                lines.append(f"  {a.id}: {rm} (bare)")
            else:
                rm, rp = resolve_worker_spec(a.role)
                if a.model:
                    rm = a.model
                src = "plan" if a.model else ("profile" if rp else "default")
                lines.append(f"  {a.id}: {rm} ({src})")

        if show_graph:
            # Pre-execution preview — draw from plan directly, NOT from the
            # builder (which has only the orchestrator's seed op at this point).
            from lionagi.operations._visualize_graph import visualize_plan

            visualize_plan(
                plan,
                title=(f"Flow DAG plan — {len(plan.agents)} agents / {len(plan.operations)} ops"),
                save_path=str(run.dag_image_path),
            )

        return "\n".join(lines)

    # ── Name allocation: one name per agent, deduped by role ────────
    name_counts: dict[str, int] = {}
    agent_id_to_name: dict[str, str] = {}
    all_agent_names: list[str] = []
    for a in plan.agents:
        base = a.role
        name_counts[base] = name_counts.get(base, 0) + 1
        wname = f"{base}-{name_counts[base]}" if name_counts[base] > 1 else base
        agent_id_to_name[a.id] = wname
        all_agent_names.append(wname)

    if team_name:
        env.team_data = _create_fanout_team(team_name, all_agent_names)
        progress(f"Team '{team_name}' created ({env.team_data['id']})")
    team_data = env.team_data

    # ── Helper: build branch for a single agent spec ───────────────
    def _build_agent_branch(
        a: FlowAgent,
    ) -> tuple[Branch, str, AgentProfile | None]:
        return build_worker_branch(
            env,
            agent_id=a.id,
            role=a.role,
            model_override=a.model,
            explicit_name=agent_id_to_name[a.id],
        )

    # ── Pass 1: build all agent branches ───────────────────────────
    agents_by_id: dict[str, Branch] = {}
    agent_model_by_id: dict[str, str] = {}
    agent_profile_by_id: dict[str, AgentProfile | None] = {}
    for a in plan.agents:
        b, m, p = _build_agent_branch(a)
        agents_by_id[a.id] = b
        agent_model_by_id[a.id] = m
        agent_profile_by_id[a.id] = p

    agent_spec_by_id: dict[str, FlowAgent] = {a.id: a for a in plan.agents}

    # ── Pass 2: create Instruction nodes + build operation graph ──
    all_instructions: Pile = Pile()
    op_to_node: dict[str, str] = {}
    op_meta: dict[str, dict] = {}
    ins_to_op: dict[str, str] = {}  # instruction_id → op_id
    all_op_results: dict = {}

    regular_ops = [op for op in plan.operations if not op.control_type]
    control_ops = [op for op in plan.operations if op.control_type]

    ctrl_note = f" ({len(control_ops)} control)" if control_ops else ""
    progress(f"Executing DAG: {len(plan.agents)} agents / {len(regular_ops)} ops{ctrl_note}...")

    for op in regular_ops:
        branch = agents_by_id[op.agent_id]
        agent = agent_spec_by_id[op.agent_id]

        ins = Instruction(
            content=InstructionContent(
                instruction=op.instruction,
                guidance=op.guidance or agent.guidance,
                prompt_context=[{"task": prompt}],
            ),
            sender=orc_branch.id,
            recipient=branch.id,
        )
        all_instructions.include(ins)

        dep_nodes = [op_to_node[d] for d in (op.depends_on or []) if d in op_to_node]
        if not dep_nodes:
            dep_nodes = [plan_root]

        node_id = builder.add_operation(
            "operate", branch=branch, depends_on=dep_nodes,
            instruction=ins.content.instruction,
            guidance=ins.content.guidance,
            context=ins.content.prompt_context,
        )
        op_to_node[op.id] = node_id
        ins_to_op[str(ins.id)] = op.id
        op_meta[op.id] = {
            "agent_id": op.agent_id,
            "agent_name": agent_id_to_name[op.agent_id],
            "model": agent_model_by_id[op.agent_id],
            "depends_on": op.depends_on or [],
            "instruction_id": str(ins.id),
        }

    # Progress callback for real-time status + JSONL event emission.
    # op_id here is the internal node UUID from the graph executor;
    # we reverse-map it to the human-readable FlowOp id when available.
    def _progress(op_id, name, status, elapsed):
        if status == "started":
            progress(f"  ▶ {name} started")
        elif status == "completed":
            progress(f"  ✓ {name} done ({elapsed:.1f}s)")
        elif status == "failed":
            progress(f"  ✗ {name} FAILED ({elapsed:.1f}s)")

    # Execute regular ops
    t_exec = time.monotonic()
    conc = max_concurrent if max_concurrent > 0 else max(len(regular_ops), 1)
    dag_result = await session.flow(
        builder.get_graph(),
        max_concurrent=conc,
        verbose=env.verbose,
        on_progress=_progress,
    )
    t_exec_elapsed = time.monotonic() - t_exec

    all_op_results.update(dag_result.get("operation_results", {}))

    progress(f"DAG done ({t_exec_elapsed:.1f}s).")

    # ── Execute control ops sequentially: each may trigger a re-plan ─
    max_rounds = 3
    round_num = 0
    for cop in control_ops:
        if cop.agent_id not in agents_by_id:
            progress(f"Control op {cop.id!r} references unknown agent, skipping.")
            continue
        c_branch = agents_by_id[cop.agent_id]
        c_model = agent_model_by_id[cop.agent_id]

        dep_nodes = [op_to_node[d] for d in (cop.depends_on or []) if d in op_to_node]
        if not dep_nodes:
            dep_nodes = list(op_to_node.values())[-1:] or [plan_root]

        progress(f"Control [{cop.id} via {cop.agent_id}]: evaluating...")

        ctrl_ins = Instruction(
            content=InstructionContent(
                instruction=f"{cop.instruction}\n\n{CONTROL_VERDICT_CONTRACT}",
                guidance=cop.guidance,
            ),
            sender=orc_branch.id,
            recipient=c_branch.id,
        )
        all_instructions.include(ctrl_ins)

        ctrl_node = builder.add_operation(
            "operate", branch=c_branch, depends_on=dep_nodes,
            instruction=ctrl_ins.content.instruction,
            guidance=ctrl_ins.content.guidance,
            field_models=[FLOW_VERDICT_FIELDS],
            reason=True,
        )
        op_to_node[cop.id] = ctrl_node

        t_ctrl = time.monotonic()
        ctrl_result = await session.flow(builder.get_graph(), verbose=env.verbose)
        t_ctrl_elapsed = time.monotonic() - t_ctrl

        ctrl_op_results = ctrl_result.get("operation_results", {})
        all_op_results.update(ctrl_op_results)
        ctrl_res = ctrl_op_results.get(ctrl_node)
        verdict: ControlDecision | None = getattr(ctrl_res, "verdict", None)
        op_meta[cop.id] = {
            "agent_id": cop.agent_id,
            "agent_name": agent_id_to_name[cop.agent_id],
            "model": c_model,
            "depends_on": cop.depends_on or [],
            "control_type": cop.control_type,
        }

        should_iterate = verdict and verdict.action == "iterate"
        progress(f"Control [{cop.id}] done ({t_ctrl_elapsed:.1f}s): {verdict.action if verdict else 'no verdict'}")

        # ── If verdict says iterate: orchestrator re-plans ────────
        if should_iterate:
            round_num += 1
            if round_num >= max_rounds:
                progress(f"Max rounds ({max_rounds}) reached, stopping.")
                break

            progress(f"Round {round_num + 1}: orchestrator re-planning...")

            existing_roster = ", ".join(f"{a.id} ({a.role})" for a in plan.agents)
            replan_node = builder.add_operation(
                "operate",
                branch=orc_branch,
                depends_on=[ctrl_node],
                instruct=Instruct(
                    instruction=(
                        f"{FlowPlan.REPLAN_INSTRUCTION}\n\n"
                        f"Existing agents you can reuse: {existing_roster}."
                    ),
                    context={
                        "original_task": prompt,
                        "prior_results": prior_results,
                        "control_verdict": verdict.reason if verdict else "",
                        "next_steps_guidance": verdict.metadata.get("next_steps", "") if verdict else "",
                    },
                    guidance=(
                        f"{roles_guidance} "
                        f"This is round {round_num + 1}. "
                        f"{FlowPlan.REPLAN_GUIDANCE}"
                    ),
                ),
                field_models=[FLOW_PLAN_FIELDS],
                reason=True,
            )

            replan_result = await session.flow(builder.get_graph(), verbose=env.verbose)
            replan_res = replan_result.get("operation_results", {}).get(replan_node)
            next_plan: FlowPlan | None = getattr(replan_res, "plan", None)

            if not next_plan or not next_plan.operations:
                progress("Re-plan produced no new operations; ending.")
                continue

            # Validate new agents have unique ids, new ops reference some agent
            combined_agent_ids = set(agents_by_id)
            for na in next_plan.agents:
                if na.id in combined_agent_ids:
                    progress(f"Re-plan: skipping duplicate agent id {na.id!r}")
                    continue
                combined_agent_ids.add(na.id)

                base = na.role
                name_counts[base] = name_counts.get(base, 0) + 1
                wname = f"{base}-{name_counts[base]}" if name_counts[base] > 1 else base
                agent_id_to_name[na.id] = wname
                all_agent_names.append(wname)
                nb, nm, np = _build_agent_branch(na)
                agents_by_id[na.id] = nb
                agent_model_by_id[na.id] = nm
                agent_profile_by_id[na.id] = np
                agent_spec_by_id[na.id] = na

            # Register new ops (and update op_id_to_agent for downstream
            # artifact-path resolution).
            new_ops: list[FlowOp] = []
            for nop in next_plan.operations:
                if nop.agent_id not in agents_by_id:
                    progress(f"Re-plan: skipping op {nop.id!r} — unknown agent {nop.agent_id!r}")
                    continue
                if nop.id in op_to_node:
                    progress(f"Re-plan: skipping duplicate op id {nop.id!r}")
                    continue
                op_id_to_agent[nop.id] = nop.agent_id
                new_ops.append(nop)

            if not new_ops:
                progress("Re-plan: no executable new ops; ending.")
                continue

            ids = ", ".join(o.id for o in new_ops)
            progress(f"Re-plan: +{len(next_plan.agents)} agents, +{len(new_ops)} ops: {ids}")

            for nop in new_ops:
                nb = agents_by_id[nop.agent_id]
                na = agent_spec_by_id[nop.agent_id]
                nd = [op_to_node[d] for d in (nop.depends_on or []) if d in op_to_node]
                if not nd:
                    nd = [ctrl_node]

                ins = Instruction(
                    content=InstructionContent(
                        instruction=nop.instruction,
                        guidance=nop.guidance or na.guidance,
                        prompt_context=[{"task": prompt}],
                    ),
                    sender=orc_branch.id,
                    recipient=nb.id,
                )
                all_instructions.include(ins)

                nid = builder.add_operation(
                    "operate", branch=nb, depends_on=nd,
                    instruction=ins.content.instruction,
                    guidance=ins.content.guidance,
                    context=ins.content.prompt_context,
                )
                op_to_node[nop.id] = nid
                ins_to_op[str(ins.id)] = nop.id
                op_meta[nop.id] = {
                    "agent_id": nop.agent_id,
                    "agent_name": agent_id_to_name[nop.agent_id],
                    "model": agent_model_by_id[nop.agent_id],
                    "depends_on": nop.depends_on or [],
                    "instruction_id": str(ins.id),
                }

            t_new = time.monotonic()
            new_result = await session.flow(
                builder.get_graph(),
                max_concurrent=conc,
                verbose=env.verbose,
            )
            t_new_elapsed = time.monotonic() - t_new
            all_op_results.update(new_result.get("operation_results", {}))
            progress(f"Round {round_num + 1} done ({t_new_elapsed:.1f}s).")

    # ── Build display results from op_meta + all_op_results ─────────
    agent_results = _build_agent_results(op_meta, op_to_node, all_op_results)

    # ── Synthesis ────────────────────────────────────────────────────
    synthesis_result = None
    if with_synthesis and agent_results:
        synth_spec = synthesis_model or model_spec
        synth_label = str(parse_model_spec(synth_spec))
        progress(f"Synthesis [{synth_label}]...")

        # Leaf ops = those not depended on by any other op. Walk op_meta
        # since it covers both initial plan and any re-planned ops.
        depended_on_nodes: set[str] = set()
        for _oid, meta in op_meta.items():
            for d in meta.get("depends_on", []):
                if d in op_to_node:
                    depended_on_nodes.add(op_to_node[d])
        all_op_node_ids = set(op_to_node.values())
        leaf_nodes = list(all_op_node_ids - depended_on_nodes) or list(all_op_node_ids)

        prior_results = [str(r) for r in all_op_results.values() if r is not None]

        synth_instruction = (
            f"Synthesize all op outputs into a final cohesive deliverable.\n\n"
            f"Original task: {prompt}\n\n"
            "Reconcile disagreements with evidence. Name gaps no op covered."
        )
        if synthesis_guidance:
            synth_instruction += f"\n\n{synthesis_guidance}"

        synth_node = builder.add_operation(
            "operate",
            branch=orc_branch,
            depends_on=leaf_nodes,
            instruction=synth_instruction,
            context=prior_results,
        )
        t_synth = time.monotonic()
        synth_result = await session.flow(builder.get_graph(), verbose=env.verbose)
        t_synth_elapsed = time.monotonic() - t_synth
        synth_res = synth_result.get("operation_results", {}).get(synth_node)
        synthesis_result = {
            "model": synth_label,
            "response": str(synth_res) if synth_res is not None else "(no response)",
            "time_ms": t_synth_elapsed * 1000,
        }
        progress(f"Synthesis done ({t_synth_elapsed:.1f}s).")

    # ── Output ───────────────────────────────────────────────────────
    if output_format == "json":
        output = _format_result_json(agent_results, synthesis_result)
    else:
        output = _format_flow_result_text(agent_results, synthesis_result)

    # ── Save synthesis (per-agent files are already in run.artifact_root/{id}/) ──
    if synthesis_result:
        run.synthesis_path.write_text(synthesis_result["response"])
    progress(f"Saved to {run.artifact_root}")

    if team_data:
        _post_results_to_team(team_data, agent_results, all_agent_names, synthesis_result)

    # ── Persist branches + run manifest + hints ──────────────────────
    finalize_orchestration(
        env,
        kind="flow",
        prompt=prompt,
        extras={
            "agents": [
                {
                    "id": agent_id,
                    "name": agent_id_to_name[agent_id],
                    "model": agent_model_by_id[agent_id],
                }
                for agent_id in agents_by_id
            ],
            "operations": [
                {
                    "id": r["id"],
                    "agent_id": r["agent_id"],
                    "control_type": r.get("control_type"),
                    "depends_on": r.get("depends_on") or [],
                }
                for r in agent_results
            ],
        },
    )

    if show_graph:
        from lionagi.operations._visualize_graph import visualize_graph

        visualize_graph(
            builder,
            title=f"Flow DAG — {len(plan.agents)} agents (completed)",
            save_path=str(run.dag_image_path),
        )

    t_total = time.monotonic() - t0
    progress(f"\nTotal: {t_total:.1f}s")

    return output
