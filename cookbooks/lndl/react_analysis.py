"""ReActStream + LNDL — full structured report assembled turn-by-turn.

Pattern::

    1. models.py: schemas for the final FullReport
    2. trace = LndlTrace()
    3. async for analysis in branch.ReActStream(lndl=True, trace=trace, ...):
           # inspect each turn (printed inline)
       # final yield = FullReport
    4. trace.summary() / classify_result() — release-grade diagnostics

Diagnostics come from the framework (``lionagi.lndl``):

    classify_chunk(text)   syntactic health of one LNDL chunk
    classify_result(value) what an operate result actually is
    LndlTrace              opt-in per-round telemetry container
    branch.lndl_chunks()   raw assistant LNDL extracted from messages

The cookbook is now a thin demo of these primitives — no inline diagnostic
machinery — so any caller of ``branch.operate(lndl=True)`` /
``branch.ReActStream(lndl=True)`` can use the same opt-in tracing pattern.

Expected variance & what the trace surfaces:

    Each ReAct beat by default uses ``ReActAnalysis`` as the per-beat schema
    (analysis + planned_actions + extension_needed + Literal action_strategy
    + nested ``PlannedAction`` list). Under LNDL, this schema is fragile —
    the model occasionally emits invalid action_strategy values, malformed
    nested PlannedAction lists, or duplicate aliases across rounds.

    Without retry (``lndl_retries=0``): you'll see ``outcome=failed`` rows
    with the exact validation error. The framework returns the partial
    output; the final yield may be a raw string or empty dict.

    With retry (``lndl_retries=1``, default below): the framework re-prompts
    the model with the validation error. This often recovers, but the model
    sometimes shifts into "fix mode" and stops issuing tool calls in
    subsequent beats — visible in the trace as ``actions=0`` everywhere.

    Either way, the trace tells you exactly what happened. Production users
    who hit this should either (a) constrain their per-beat schema (e.g.
    write a custom ``intermediate_response_options`` BaseModel that's
    LNDL-friendly: scalar fields only, no Literal types, no nested optional
    BaseModels) or (b) skip ReAct framing and use plain
    ``branch.operate(lndl=True, lndl_rounds=N, response_format=...)``.

Run::

    uv run python cookbooks/lndl/react_analysis.py [--rounds 1] [--extensions 4]
                                                   [--show-lndl] [--out report.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from lionagi import Branch, iModel
from lionagi.lndl import LndlTrace, classify_chunk, classify_result
from lionagi.tools.coding import CodingToolkit

load_dotenv()

WORKSPACE = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Schemas — what the final FullReport looks like
# ---------------------------------------------------------------------------


class CodeFinding(BaseModel):
    name: str = Field(description="Short label, e.g. 'LndlFormatter'.")
    location: str = Field(description="path:line from real tool output.")
    role: str = Field(description="One sentence on what this code does.")


class Tension(BaseModel):
    issue: str = Field(description="The tension or limitation in one sentence.")
    evidence: str = Field(description="path:line plus a short explanation.")


class Recommendation(BaseModel):
    change: str = Field(description="One-sentence proposed change.")
    rationale: str = Field(description="Why, grounded in evidence.")


class FullReport(BaseModel):
    """The final synthesized report (last yield from ReActStream)."""

    title: str
    summary: str = Field(description="3-5 sentence executive summary.")
    findings: list[CodeFinding] = Field(
        ..., description="Consolidated findings across all beats."
    )
    tensions: list[Tension] = Field(..., description="2-4 architectural tensions.")
    recommendations: list[Recommendation] = Field(
        ..., description="2-3 concrete improvements."
    )
    confidence: float = Field(
        description="Self-rated 0.0-1.0. Higher = more confident."
    )


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = f"""You are a code-exploration agent with READ-ONLY access to {WORKSPACE}.

Tools (CodingToolkit, read-only subset):
- reader(action='read', path=..., offset=0, limit=200)
- reader(action='list_dir', path=..., recursive=False)        # NEVER recursive=True
- search(action='grep', pattern=..., path=..., include='*.py', max_results=30)
- search(action='find', pattern='*.py', path=..., max_results=50)
- context(action='evict_action_results')                      # call to drop old tool output

CONTEXT DISCIPLINE — CRITICAL:
- Tool output is preserved across beats. Each beat costs prior beats' bytes.
- NEVER call `reader(action='list_dir', recursive=True)` — explodes context.
- Scope grep paths to specific files/dirs (e.g. lionagi/operations/operate/),
  never the whole repo.
- Use offset+limit on read; default to limit=120 unless you need more.
- After you have enough info, set extension_needed=False.

Output discipline:
- Cite EXACT path:line numbers from real tool output. Never invent locations.
- Path format: `lionagi/.../file.py:NN` (workspace-relative).
"""


INSTRUCTION = (
    "Build a FullReport on lionagi's LNDL-vs-JSON formatter dispatch.\n\n"
    "Scope this strictly to:\n"
    "  - lionagi/operations/operate/operate.py\n"
    "  - lionagi/protocols/messages/_helpers/_lndl_formatter.py\n"
    "  - lionagi/protocols/messages/_helpers/_json_formatter.py\n"
    "  - lionagi/lndl/parser.py\n"
    "  - lionagi/lndl/assembler.py\n\n"
    "Required deliverables:\n"
    "  1. AT LEAST 5 distinct findings — Formatter base class, JsonFormatter,\n"
    "     LndlFormatter, the parser entry point, the assembler entry point.\n"
    "     Each with real file:line.\n"
    "  2. AT LEAST 3 design tensions, each with file:line evidence.\n"
    "  3. AT LEAST 3 concrete improvement recommendations.\n"
    "  4. A 3-5 sentence summary tying everything together.\n\n"
    "Strategy:\n"
    "  - Beat 1: locate the formatters via grep, read each's class definition.\n"
    "  - Beat 2: trace the dispatch in operate.py and the LNDL parser entry.\n"
    "  - Beat 3 (if needed): synthesize tensions + recommendations.\n\n"
    "Set extension_needed=True until you have all 5+ findings AND visited\n"
    "all 5 files. ONLY then set extension_needed=False.\n\n"
    "Stay within the 5 files above. Do NOT scan the whole repo."
)


def _build_branch() -> Branch:
    chat_model = iModel(model="openai/gpt-5.4-mini")
    branch = Branch(system=SYSTEM_PROMPT, chat_model=chat_model)
    toolkit = CodingToolkit(workspace_root=WORKSPACE, notify=False)
    branch.register_tools(toolkit.bind(branch, include=CodingToolkit.READ_ONLY))
    return branch


# ---------------------------------------------------------------------------
# Diagnostic display helpers
# ---------------------------------------------------------------------------


def _safe_attr(obj: Any, name: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _print_turn(idx: int, analysis: Any, delta_msgs: int, elapsed: float) -> None:
    kind = classify_result(analysis)
    type_name = type(analysis).__name__
    n_findings = _safe_attr(analysis, "findings", None)
    n_findings = len(n_findings) if isinstance(n_findings, list) else 0
    n_actions = _safe_attr(analysis, "action_responses", None)
    n_actions = len(n_actions) if isinstance(n_actions, list) else 0
    print(
        f"  turn {idx}: {kind:<6} | {type_name:<28} | "
        f"+{delta_msgs:2d} msgs | findings={n_findings} | actions={n_actions} | "
        f"{elapsed:5.1f}s"
    )
    if hasattr(analysis, "analysis"):
        preview = str(analysis.analysis)[:140].replace("\n", " ")
        if preview:
            print(f"        analysis: {preview}")


def _print_lndl_chunks(chunks: list[str]) -> None:
    for ci, chunk in enumerate(chunks):
        h = classify_chunk(chunk)
        print(
            f"        ---- lndl chunk {ci+1}/{len(chunks)} "
            f"({h.status}, open={h.open_tags}, close={h.close_tags}) ----"
        )
        for line in chunk.splitlines():
            print(f"        | {line}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_react_analysis(
    rounds: int,
    extensions: int,
    out_path: Path | None,
    show_lndl: bool = False,
) -> dict:
    branch = _build_branch()
    trace = LndlTrace()

    print(f"\n=== ReActStream + LNDL | rounds={rounds} | extensions={extensions} ===")
    t_start = time.time()

    last_msgs = len(branch.messages)
    final: Any = None
    idx = 0

    async for analysis in branch.ReActStream(
        instruct={"instruction": INSTRUCTION},
        tools=True,
        response_format=FullReport,
        # NOTE: deliberately NOT using intermediate_response_options=BeatNote.
        # An earlier iteration showed the model trying
        # `<lact IntermediateResponseOptions.beatnote>` to fill a nested
        # BaseModel field with a tool return — wrong shape. Plain
        # ReActAnalysis per beat parses cleanly under LNDL; the typed
        # final FullReport is where the structure pays off.
        max_extensions=extensions,
        extension_allowed=True,
        lndl=True,
        lndl_rounds=rounds,
        lndl_retries=1,
        trace=trace,  # opt-in framework telemetry
    ):
        idx += 1
        now_msgs = len(branch.messages)
        delta = now_msgs - last_msgs
        chunks_this_turn = branch.lndl_chunks(since=last_msgs)
        last_msgs = now_msgs

        _print_turn(idx, analysis, delta, time.time() - t_start)
        if show_lndl and chunks_this_turn:
            _print_lndl_chunks(chunks_this_turn)

        final = analysis

    elapsed = time.time() - t_start
    print(f"\n[done in {elapsed:.1f}s | total messages = {len(branch.messages)}]")
    print()
    print("=" * 70)
    print("LndlTrace (per-round framework telemetry)")
    print("=" * 70)
    print(f"  {trace.summary()}")
    for i, r in enumerate(trace.rounds):
        h = r.health.status
        err_hint = f" — {r.error[:80]}" if r.error else ""
        print(
            f"  round {i}: outcome={r.outcome:<8} schema={r.schema or '?':<20} "
            f"health={h} actions={r.actions_executed}{err_hint}"
        )

    # Final yield should be the FullReport (or the response-wrapper variant).
    final_kind = classify_result(final)
    if isinstance(final, BaseModel) and (
        isinstance(final, FullReport)
        or (hasattr(final, "title") and hasattr(final, "findings"))
    ):
        _print_full_report(final)
    else:
        print(f"\n⚠  final yield was {final_kind} ({type(final).__name__}); no report.")

    summary = {
        "model": "openai/gpt-5.4-mini",
        "rounds": rounds,
        "extensions": extensions,
        "total_duration_s": round(elapsed, 1),
        "total_messages": len(branch.messages),
        "trace_summary": trace.summary(),
        "trace_health": trace.health(),
        "trace_outcomes": trace.outcomes(),
        "final_kind": final_kind,
        "final_type": type(final).__name__,
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_dump: Any = (
            final.model_dump()
            if isinstance(final, BaseModel)
            else final if isinstance(final, dict) else repr(final)
        )
        out_path.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "report": report_dump,
                    "rounds": [
                        {
                            "outcome": r.outcome,
                            "schema": r.schema,
                            "error": r.error,
                            "actions_executed": r.actions_executed,
                            "health": r.health.status,
                            "raw_preview": r.raw[:400],
                        }
                        for r in trace.rounds
                    ],
                },
                indent=2,
                default=str,
            )
        )
        print(f"\n[saved to {out_path}]")

    return summary


def _print_full_report(report: Any) -> None:
    print("\n" + "=" * 70)
    print(f"REPORT — {_safe_attr(report, 'title', '(no title)')}")
    print("=" * 70)
    print(f"\nSummary:\n  {_safe_attr(report, 'summary', '(none)')}")
    print(f"Confidence: {_safe_attr(report, 'confidence', 0)}")

    findings = _safe_attr(report, "findings", []) or []
    if findings:
        print(f"\nFindings ({len(findings)}):")
        for f in findings:
            print(
                f"  - {_safe_attr(f, 'name', '?')} @ {_safe_attr(f, 'location', '?')}"
            )
            print(f"    {_safe_attr(f, 'role', '')}")

    tensions = _safe_attr(report, "tensions", []) or []
    if tensions:
        print(f"\nTensions ({len(tensions)}):")
        for t in tensions:
            print(f"  - {_safe_attr(t, 'issue', '?')}")
            print(f"    {_safe_attr(t, 'evidence', '')}")

    recs = _safe_attr(report, "recommendations", []) or []
    if recs:
        print(f"\nRecommendations ({len(recs)}):")
        for r in recs:
            print(f"  - {_safe_attr(r, 'change', '?')}")
            print(f"    Why: {_safe_attr(r, 'rationale', '')}")

    # Citation accuracy: how many path:line references resolve to real files?
    refs: list[str] = []
    for f in findings:
        loc = _safe_attr(f, "location", "")
        if isinstance(loc, str):
            refs.append(loc)
    for t in tensions:
        ev = _safe_attr(t, "evidence", "")
        if isinstance(ev, str):
            refs.append(ev)
    valid = total = 0
    for ref in refs:
        path = ref.split(":")[0].strip().rstrip(",").rstrip(".")
        if not path or "/" not in path:
            continue
        candidate = Path(path) if Path(path).is_absolute() else WORKSPACE / path
        if path.endswith(".py"):
            total += 1
            if candidate.is_file():
                valid += 1
    if total:
        print(f"\nCitation accuracy: {valid}/{total}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    # Defaults chosen to maximise quality on gpt-5.4-mini: rounds=3 lets
    # each ReAct beat run multi-round LNDL exploration, extensions=3 caps
    # the outer ReAct loop. Lower values trade depth for visibility — see
    # the docstring for what the trace surfaces in different configs.
    ap.add_argument("--rounds", type=int, default=3, help="LNDL rounds per ReAct beat.")
    ap.add_argument(
        "--extensions",
        type=int,
        default=3,
        help="Max ReAct beats (max_extensions).",
    )
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--show-lndl",
        action="store_true",
        help="Dump raw assistant LNDL per beat (loud but maximally diagnostic).",
    )
    args = ap.parse_args()

    summary = await run_react_analysis(
        rounds=args.rounds,
        extensions=args.extensions,
        out_path=Path(args.out) if args.out else None,
        show_lndl=args.show_lndl,
    )
    print("\nSUMMARY:")
    print(f"  {summary['trace_summary']}")
    print(f"  total_duration_s: {summary['total_duration_s']}")
    print(f"  final: {summary['final_kind']} ({summary['final_type']})")


if __name__ == "__main__":
    asyncio.run(main())
