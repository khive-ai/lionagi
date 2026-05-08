"""ReAct + multi-round LNDL -- phased structured codebase analysis.

Architecture::

    branch (single conversation)
      Phase 1 (Discovery)  -> FormatterInventory
      Phase 2 (Trace)      -> PipelineFlow
      Phase 3 (Critique)   -> DesignReview
      Phase 4 (Synthesis)  -> Synthesis  (no tools, shorter)
                           => FullReport (assembled in Python)

Each phase is a ``branch.operate(lndl=True, lndl_rounds=N)`` call with
a narrow schema. The same Branch is reused so prior phases' tool
messages live in chat history for later reference.

Run::

    uv run python cookbooks/lndl/chain_analysis.py [--model openai/gpt-5.4-mini]
                                                   [--rounds 8]
                                                   [--out report.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from lionagi import Branch, iModel
from lionagi.tools.coding import CodingToolkit

load_dotenv()

WORKSPACE = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Phase schemas
# ---------------------------------------------------------------------------


class Finding(BaseModel):
    name: str = Field(description="Short label, e.g. 'JsonFormatter'.")
    location: str = Field(description="path:line from tool output.")
    role: str = Field(description="One sentence on what this piece does.")


class FlowStep(BaseModel):
    step: str = Field(description="What happens (1-2 sentences).")
    code_ref: str = Field(description="path:line anchoring the step.")


class Tension(BaseModel):
    issue: str = Field(description="The architectural tension or limitation.")
    evidence: str = Field(description="path:line plus short explanation.")


class Recommendation(BaseModel):
    change: str = Field(description="Concrete proposed change.")
    rationale: str = Field(description="Why, grounded in evidence.")
    files_affected: list[str] = Field(
        ..., description="Workspace-relative paths likely to need edits."
    )


class FormatterInventory(BaseModel):
    formatters: list[Finding] = Field(
        ..., description="Every Formatter implementation with file:line and role."
    )


class PipelineFlow(BaseModel):
    flow_trace: list[FlowStep] = Field(
        ..., description="Ordered steps from branch.operate() to validated model."
    )


class DesignReview(BaseModel):
    design_tensions: list[Tension] = Field(
        ..., description="2-4 design tensions or limitations."
    )
    recommendations: list[Recommendation] = Field(
        ..., description="2-3 concrete improvements."
    )


class Synthesis(BaseModel):
    title: str = Field(description="Short report title.")
    summary: str = Field(description="3-5 sentence executive summary.")
    confidence: float = Field(default=0.5, description="Self-rated 0.0-1.0.")


class FullReport(BaseModel):
    title: str
    summary: str
    formatters: list[Finding]
    flow_trace: list[FlowStep]
    design_tensions: list[Tension]
    recommendations: list[Recommendation]
    confidence: float


# ---------------------------------------------------------------------------
# Phase definitions (data-driven, not copy-pasted)
# ---------------------------------------------------------------------------


@dataclass
class PhaseSpec:
    name: str
    instruction: str
    schema: type[BaseModel]
    use_tools: bool = True
    rounds_factor: float = 1.0  # multiplied by the global --rounds arg


PHASES: list[PhaseSpec] = [
    PhaseSpec(
        name="Discovery",
        schema=FormatterInventory,
        instruction=(
            "**Phase 1 -- Formatter Discovery**\n\n"
            "Locate EVERY Formatter implementation in the lionagi codebase. "
            "For each, give name, location (path:line from tool output), and "
            "role (one sentence). Use grep/find/read_file. Cite real tool output."
        ),
    ),
    PhaseSpec(
        name="Trace",
        schema=PipelineFlow,
        instruction=(
            "**Phase 2 -- Flow Trace**\n\n"
            "Building on the formatter inventory above, trace what happens "
            "when ``branch.operate(response_format=SomeModel)`` is called. "
            "Produce 5-10 ordered steps with path:line anchors."
        ),
    ),
    PhaseSpec(
        name="Critique",
        schema=DesignReview,
        instruction=(
            "**Phase 3 -- Design Review**\n\n"
            "Based on your inventory and flow trace, surface 2-4 design "
            "tensions and propose 2-3 concrete improvements. Each tension "
            "needs a path:line citation. Each recommendation needs rationale "
            "and files affected."
        ),
    ),
    PhaseSpec(
        name="Synthesis",
        schema=Synthesis,
        use_tools=False,
        rounds_factor=0.5,  # synthesis is short
        instruction=(
            "**Phase 4 -- Synthesis**\n\n"
            "Produce a short executive header from Phases 1-3:\n"
            "- title: concise report title\n"
            "- summary: 3-5 sentences tying together all phases\n"
            "- confidence: self-rated 0.0-1.0"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = f"""You are a code-exploration agent with READ-ONLY access to {WORKSPACE}.

Tools available (coding toolkit, read-only subset):
- reader(action='read', path=..., offset=0, limit=200)
- reader(action='list_dir', path=..., recursive=False)
- search(action='grep', pattern=..., path=..., include='*.py', max_results=50)
- search(action='find', pattern='*.py', path=..., max_results=100)
- context(action='status' | 'get_messages' | 'evict' | 'evict_action_results')

Cite EXACT path:line numbers from real tool output; never invent them.
"""


async def _run_phase(
    branch: Branch,
    spec: PhaseSpec,
    rounds: int,
) -> tuple[Any, dict]:
    effective_rounds = max(2, int(rounds * spec.rounds_factor))
    print(f"\n-- {spec.name} (rounds={effective_rounds}, tools={spec.use_tools}) --")
    t0 = time.time()
    n_before = len(branch.messages)

    result = await branch.operate(
        instruction=spec.instruction,
        actions=spec.use_tools,
        lndl=True,
        lndl_rounds=effective_rounds,
        response_format=spec.schema,
    )

    elapsed = time.time() - t0
    n_added = len(branch.messages) - n_before
    metrics = {
        "phase": spec.name,
        "duration_s": round(elapsed, 1),
        "messages_added": n_added,
        "result_type": type(result).__name__,
    }

    if hasattr(result, "model_dump"):
        preview = {
            k: (
                f"<{len(v)} items>"
                if isinstance(v, list)
                else (v[:120] if isinstance(v, str) else v)
            )
            for k, v in result.model_dump().items()
        }
        print(f"  {elapsed:.1f}s, +{n_added} messages | {preview}")
    else:
        print(f"  {elapsed:.1f}s | non-model: {type(result).__name__}")
        print(f"  {repr(result)[:300]}")

    return result, metrics


def _safe_attr(obj: Any, name: str, default: Any) -> Any:
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name, default)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _check_citations(report: FullReport) -> str:
    refs: list[str] = []
    for f in report.formatters:
        refs.append(f.location)
    for s in report.flow_trace:
        refs.append(s.code_ref)
    for t in report.design_tensions:
        refs.extend(re.findall(r"[A-Za-z0-9_./\\-]+\.py(?::\d+)?", t.evidence or ""))
    for r in report.recommendations:
        refs.extend(r.files_affected)

    valid = total = 0
    for ref in refs:
        path = ref.split(":")[0].strip().rstrip(",")
        if not path or not path.endswith(".py"):
            continue
        total += 1
        if (WORKSPACE / path).is_file():
            valid += 1
    return f"{valid}/{total}"


def _print_report(report: FullReport, elapsed: float, n_messages: int) -> None:
    print(f"\n{'='*70}")
    print(f"REPORT  ({elapsed:.1f}s, {n_messages} messages)")
    print(f"{'='*70}")
    print(f"\nTitle: {report.title}")
    print(f"\nSummary:\n  {report.summary}")
    print(f"Confidence: {report.confidence}")

    for label, items, fmt in [
        (
            "Formatters",
            report.formatters,
            lambda f: f"  - {f.name} @ {f.location}\n    {f.role}",
        ),
        ("Flow trace", report.flow_trace, lambda s: f"  {s.step}\n    ({s.code_ref})"),
        (
            "Tensions",
            report.design_tensions,
            lambda t: f"  - {t.issue}\n    {t.evidence}",
        ),
        (
            "Recommendations",
            report.recommendations,
            lambda r: f"  - {r.change}\n    Why: {r.rationale}\n    Files: {', '.join(r.files_affected)}",
        ),
    ]:
        if items:
            print(f"\n{label} ({len(items)}):")
            for item in items:
                print(fmt(item))

    print(f"\nCitation accuracy: {_check_citations(report)}")


async def run_pipeline(model: str, rounds: int, out_path: Path | None = None) -> dict:
    model_kwargs: dict = {}
    if model.startswith("openrouter/"):
        model_kwargs["reasoning"] = {"effort": "none"}

    chat_model = iModel(model=model, **model_kwargs)
    branch = Branch(system=SYSTEM_PROMPT, chat_model=chat_model)
    toolkit = CodingToolkit(workspace_root=WORKSPACE, notify=False)
    branch.register_tools(toolkit.bind(branch, include=CodingToolkit.READ_ONLY))

    print(f"\n=== ReAct-LNDL pipeline | {model} | rounds={rounds} ===")
    t0 = time.time()

    # Run all phases in sequence, collecting results
    results: dict[str, Any] = {}
    all_metrics: list[dict] = []
    for spec in PHASES:
        result, metrics = await _run_phase(branch, spec, rounds)
        results[spec.name] = result
        all_metrics.append(metrics)

    elapsed = time.time() - t0

    # Assemble final report from phase outputs
    report = FullReport(
        title=_safe_attr(results.get("Synthesis"), "title", "Untitled"),
        summary=_safe_attr(results.get("Synthesis"), "summary", ""),
        formatters=_safe_attr(results.get("Discovery"), "formatters", []) or [],
        flow_trace=_safe_attr(results.get("Trace"), "flow_trace", []) or [],
        design_tensions=_safe_attr(results.get("Critique"), "design_tensions", [])
        or [],
        recommendations=_safe_attr(results.get("Critique"), "recommendations", [])
        or [],
        confidence=_safe_attr(results.get("Synthesis"), "confidence", 0.5) or 0.5,
    )

    _print_report(report, elapsed, len(branch.messages))

    summary = {
        "model": model,
        "rounds_per_phase": rounds,
        "total_duration_s": round(elapsed, 1),
        "total_messages": len(branch.messages),
        "phases": all_metrics,
        "citation_accuracy": _check_citations(report),
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {"metrics": summary, "report": report.model_dump()},
                indent=2,
                default=str,
            )
        )
        print(f"\n[saved to {out_path}]")

    return summary


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-5.4-mini")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    metrics = await run_pipeline(
        args.model, args.rounds, Path(args.out) if args.out else None
    )
    print(f"\nMETRICS: {json.dumps(metrics, indent=2, default=str)}")


if __name__ == "__main__":
    asyncio.run(main())
