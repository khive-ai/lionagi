"""ReAct + multi-round LNDL — phased structured codebase analysis.

Architecture:

    branch (single conversation)
      ├─ Phase 1 (Discovery)  → operate(lndl=True, lndl_rounds=N, schema=FormatterInventory)
      ├─ Phase 2 (Trace)      → operate(lndl=True, lndl_rounds=N, schema=PipelineFlow)
      ├─ Phase 3 (Critique)   → operate(lndl=True, lndl_rounds=N, schema=DesignReview)
      └─ Final (Synthesis)    → operate(lndl=True, lndl_rounds=N, schema=Synthesis)

Each phase is a self-contained multi-round LNDL operate call with a
narrow schema (one or two `list[Model]` fields), so the model only
juggles the LNDL grammar for one shape at a time. The same Branch is
reused, so prior phases' raw LNDL responses + tool messages live in
chat history — the final synthesis can reference them.

Run::

    uv run python cookbooks/lndl/chain_analysis.py [--model openai/gpt-5.4-mini]
                                                   [--rounds 8]
                                                   [--out OUT.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from lionagi import Branch, iModel
from lionagi.tools.coding import CodingToolkit

load_dotenv()

WORKSPACE = Path(__file__).resolve().parents[2]


# ─── Phase schemas ──────────────────────────────────────────────────────────


class Finding(BaseModel):
    name: str = Field(description="Short label, e.g. 'JsonFormatter'.")
    location: str = Field(description="path:line — must come from a tool result.")
    role: str = Field(description="One sentence on what this piece does.")


class FlowStep(BaseModel):
    step: str = Field(description="What happens at this step (1-2 sentences).")
    code_ref: str = Field(description="path:line anchoring the step.")


class Tension(BaseModel):
    issue: str = Field(description="The architectural tension or limitation.")
    evidence: str = Field(description="path:line plus a short explanation.")


class Recommendation(BaseModel):
    change: str = Field(description="Concrete proposed change.")
    rationale: str = Field(description="Why this helps, grounded in evidence above.")
    files_affected: list[str] = Field(
        ..., description="Workspace-relative paths likely to need edits."
    )


class FormatterInventory(BaseModel):
    """Phase 1 output."""

    formatters: list[Finding] = Field(
        ...,
        description=(
            "Every Formatter implementation found in the codebase, with "
            "file:line and role."
        ),
    )


class PipelineFlow(BaseModel):
    """Phase 2 output."""

    flow_trace: list[FlowStep] = Field(
        ...,
        description=(
            "Ordered steps from `branch.operate(response_format=X)` to a "
            "validated model. 5-10 steps."
        ),
    )


class DesignReview(BaseModel):
    """Phase 3 output."""

    design_tensions: list[Tension] = Field(
        ..., description="2-4 design tensions or limitations."
    )
    recommendations: list[Recommendation] = Field(
        ..., description="2-3 concrete improvements with rationale."
    )


class Synthesis(BaseModel):
    """Phase 4 output — assembled report header."""

    title: str = Field(description="Short title for the report.")
    summary: str = Field(
        description="3-5 sentence executive summary tying together all phases.",
    )
    confidence: float = Field(default=0.5, description="Self-rated 0.0-1.0.")


class FullReport(BaseModel):
    """Final assembled report — merged from all four phases."""

    title: str
    summary: str
    formatters: list[Finding]
    flow_trace: list[FlowStep]
    design_tensions: list[Tension]
    recommendations: list[Recommendation]
    confidence: float


# ─── Phase prompts ──────────────────────────────────────────────────────────


PHASE_1_INSTRUCTION = """**Phase 1 — Formatter Discovery**

Your goal: locate EVERY ``Formatter`` implementation in the lionagi codebase
that participates in the structured-output pipeline. For each, give:

- name (e.g. ``JsonFormatter``, ``LndlFormatter``, the ``Formatter`` protocol)
- location (path:line — must come from grep/read_file output, not invention)
- role (one sentence describing its job)

Use grep / find_files / read_file as needed. Take as many rounds as you need
to be thorough. CITE path:line from real tool output.
"""

PHASE_2_INSTRUCTION = """**Phase 2 — Flow Trace**

Building on the formatter inventory you just produced (visible above in
chat history), trace what happens when a caller does
``branch.operate(response_format=SomeModel, ...)``. Produce 5-10 ordered
steps showing the path from the call site to a validated Pydantic model.

Each step needs a path:line anchor in the source. Use grep / read_file
to confirm specific function names and line numbers — do not invent.
"""

PHASE_3_INSTRUCTION = """**Phase 3 — Design Review**

Based on your formatter inventory and flow trace, surface real design
tensions in the structured-output pipeline (2-4 of them). Then propose
2-3 concrete improvements.

Tensions should be specific architectural issues — coupling, fallback
behavior, validation paths that diverge, etc. Each tension needs a
path:line citation. Each recommendation needs a clear rationale and the
files it would touch.
"""

PHASE_4_INSTRUCTION = """**Phase 4 — Synthesis**

You have produced (visible in chat history above):
- A formatter inventory (Phase 1)
- A pipeline flow trace (Phase 2)
- Design tensions and recommendations (Phase 3)

Now produce a short executive header for the report:
- title: a concise title for the report
- summary: 3-5 sentences pulling together the formatter inventory, flow,
  tensions, and recommendations into a single narrative
- confidence: your self-rated 0.0-1.0 confidence in the analysis
"""


# ─── Runner ─────────────────────────────────────────────────────────────────


async def _phase(
    branch: Branch,
    *,
    name: str,
    instruction: str,
    schema: type[BaseModel],
    rounds: int,
    use_tools: bool,
) -> tuple[Any, dict]:
    print(f"\n── PHASE: {name} ──")
    t0 = time.time()
    msg_before = len(branch.messages)
    result = await branch.operate(
        instruction=instruction,
        actions=use_tools,
        lndl=True,
        lndl_rounds=rounds,
        response_format=schema,
    )
    elapsed = time.time() - t0
    msg_after = len(branch.messages)
    metrics = {
        "phase": name,
        "duration_s": round(elapsed, 1),
        "messages_added": msg_after - msg_before,
        "result_type": type(result).__name__,
    }
    if hasattr(result, "model_dump"):
        d = result.model_dump()
        # Show a quick preview
        preview = {
            k: (
                f"<{len(v)} items>"
                if isinstance(v, list)
                else (v[:120] if isinstance(v, str) else v)
            )
            for k, v in d.items()
        }
        print(f"  duration: {elapsed:.1f}s, +{msg_after - msg_before} messages")
        print(f"  result: {preview}")
    else:
        print(f"  [non-model result, type={type(result).__name__}]")
        print(f"  preview: {repr(result)[:300]}")
    return result, metrics


_SYSTEM_PROMPT = f"""You are a code-exploration agent with READ-ONLY access to {WORKSPACE}.

The coding toolkit is filtered to its read-only actions:
- reader(action='read', path=..., offset=0, limit=200) — bounded line read
- reader(action='list_dir', path=..., recursive=False) — vendor dirs skipped
- search(action='grep', pattern=..., path=..., include='*.py', max_results=50)
- search(action='find', pattern='*.py', path=..., max_results=100)
- context(action='status' | 'get_messages' | 'evict' | 'evict_action_results')

Editor / bash / sandbox / subagent are NOT registered — only the read-only
subset is available.

Cite EXACT path:line numbers from real tool output; never invent them.
"""


async def run_pipeline(
    model: str,
    rounds: int,
    out_path: Path | None = None,
) -> dict:
    model_kwargs: dict = {}
    if model.startswith("openrouter/"):
        model_kwargs["reasoning"] = {"effort": "none"}
    chat_model = iModel(model=model, **model_kwargs)
    branch = Branch(system=_SYSTEM_PROMPT, chat_model=chat_model)
    toolkit = CodingToolkit(workspace_root=WORKSPACE, notify=False)
    branch.register_tools(toolkit.bind(branch, include=CodingToolkit.READ_ONLY))

    print(f"\n=== ReAct-LNDL pipeline | {model} | rounds-per-phase={rounds} ===")
    t0 = time.time()
    all_metrics: list[dict] = []

    p1, m = await _phase(
        branch,
        name="Discovery",
        instruction=PHASE_1_INSTRUCTION,
        schema=FormatterInventory,
        rounds=rounds,
        use_tools=True,
    )
    all_metrics.append(m)

    p2, m = await _phase(
        branch,
        name="Trace",
        instruction=PHASE_2_INSTRUCTION,
        schema=PipelineFlow,
        rounds=rounds,
        use_tools=True,
    )
    all_metrics.append(m)

    p3, m = await _phase(
        branch,
        name="Critique",
        instruction=PHASE_3_INSTRUCTION,
        schema=DesignReview,
        rounds=rounds,
        use_tools=True,
    )
    all_metrics.append(m)

    p4, m = await _phase(
        branch,
        name="Synthesis",
        instruction=PHASE_4_INSTRUCTION,
        schema=Synthesis,
        rounds=max(2, rounds // 2),  # synthesis is short, doesn't need 8 rounds
        use_tools=False,  # final synthesis pulls from chat history, not new tools
    )
    all_metrics.append(m)

    elapsed = time.time() - t0

    # ── Assemble final report ──
    def _attr(obj, name, default):
        if obj is None:
            return default
        return (
            getattr(obj, name, default)
            if hasattr(obj, name)
            else (obj.get(name, default) if isinstance(obj, dict) else default)
        )

    report = FullReport(
        title=_attr(p4, "title", "Untitled analysis"),
        summary=_attr(p4, "summary", ""),
        formatters=_attr(p1, "formatters", []) or [],
        flow_trace=_attr(p2, "flow_trace", []) or [],
        design_tensions=_attr(p3, "design_tensions", []) or [],
        recommendations=_attr(p3, "recommendations", []) or [],
        confidence=_attr(p4, "confidence", 0.5) or 0.5,
    )

    # ── Print report ──
    print(
        f"\n{'='*70}\nFINAL REPORT  ({elapsed:.1f}s total, {len(branch.messages)} messages)\n{'='*70}"
    )
    print(f"\nTitle: {report.title}")
    print(f"\nSummary:\n  {report.summary}")
    print(f"\nConfidence: {report.confidence}")

    if report.formatters:
        print(f"\nFormatters ({len(report.formatters)}):")
        for f in report.formatters:
            print(f"  - {f.name} @ {f.location}")
            print(f"    {f.role}")

    if report.flow_trace:
        print(f"\nFlow trace ({len(report.flow_trace)} steps):")
        for i, s in enumerate(report.flow_trace, 1):
            print(f"  {i}. {s.step}")
            print(f"     ({s.code_ref})")

    if report.design_tensions:
        print(f"\nDesign tensions ({len(report.design_tensions)}):")
        for t in report.design_tensions:
            print(f"  - {t.issue}")
            print(f"    Evidence: {t.evidence}")

    if report.recommendations:
        print(f"\nRecommendations ({len(report.recommendations)}):")
        for r in report.recommendations:
            print(f"  - {r.change}")
            print(f"    Why: {r.rationale}")
            if r.files_affected:
                print(f"    Files: {', '.join(r.files_affected)}")

    # ── Citation accuracy ──
    refs: list[str] = []
    for f in report.formatters:
        refs.append(f.location)
    for s in report.flow_trace:
        refs.append(s.code_ref)
    for t in report.design_tensions:
        refs.extend(re.findall(r"[A-Za-z0-9_./\\-]+\.py(?::\d+)?", t.evidence or ""))
    for r in report.recommendations:
        refs.extend(r.files_affected)

    valid, total = 0, 0
    for ref in refs:
        file_part = ref.split(":")[0].strip().rstrip(",")
        if not file_part or not file_part.endswith(".py"):
            continue
        total += 1
        if (WORKSPACE / file_part).is_file():
            valid += 1
    citation_rate = f"{valid}/{total}" if total else "0/0"
    print(f"\nCitation accuracy: {citation_rate} ({valid}/{total} .py paths exist)")

    summary_metrics = {
        "model": model,
        "rounds_per_phase": rounds,
        "total_duration_s": round(elapsed, 1),
        "total_messages": len(branch.messages),
        "phases": all_metrics,
        "citation_accuracy": citation_rate,
        "n_formatters": len(report.formatters),
        "n_flow_steps": len(report.flow_trace),
        "n_tensions": len(report.design_tensions),
        "n_recommendations": len(report.recommendations),
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {"metrics": summary_metrics, "report": report.model_dump()},
                indent=2,
                default=str,
            )
        )
        print(f"\n[saved to {out_path}]")

    return summary_metrics


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-5.4-mini")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.out) if args.out else None
    metrics = await run_pipeline(args.model, args.rounds, out)
    print(f"\nMETRICS: {json.dumps(metrics, indent=2, default=str)}")


if __name__ == "__main__":
    asyncio.run(main())
