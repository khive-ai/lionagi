"""LNDL coding agent -- codebase exploration with read-only tools.

Multi-round LNDL (``lndl_rounds=4``): the model issues tool calls in
early rounds without an OUT block, sees results in chat history, then
commits OUT once it has enough information.

Tools come from ``CodingToolkit`` filtered to ``READ_ONLY`` (reader,
search, context).  Editor / bash / sandbox / subagent are NOT registered.

Run::

    uv run python cookbooks/lndl/explore_agent.py [--task=N] [--rounds=4]

Tasks:
    1. Locate: find files implementing LNDL parsing
    2. Summarize: what does operate.py do (3 bullets)
    3. Survey: map operations/ -- one-line purpose per file
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from lionagi import Branch, iModel
from lionagi.tools.coding import CodingToolkit

load_dotenv()

WORKSPACE = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class ExploreFindings(BaseModel):
    summary: str = Field(description="2-5 sentence summary of what was found.")
    files: list[str] = Field(
        default_factory=list,
        description="Workspace-relative paths inspected.",
    )
    key_lines: list[str] = Field(
        default_factory=list,
        description="Notable 'path:line: snippet' strings.",
    )
    confidence: float = Field(default=0.5, description="Self-rated 0.0-1.0.")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = f"""You are a code-exploration agent with READ-ONLY access to {WORKSPACE}.

Tools (coding toolkit, read-only subset):
- reader(action='read', path=..., offset=0, limit=200)
- reader(action='list_dir', path=..., recursive=False)
- search(action='grep', pattern=..., path=..., include='*.py', max_results=50)
- search(action='find', pattern='*.py', path=..., max_results=100)
- context(action='status' | 'get_messages' | 'evict' | 'evict_action_results')

Multi-round LNDL strategy:
1. First rounds -- <lact> calls to gather info. No OUT yet.
2. Middle rounds -- narrow reads, use <lvar note.X> to jot findings.
3. Final round -- synthesize via <lvar> and commit OUT.

Cite EXACT path:line from tool output; never invent them.
"""


TASKS = {
    1: "Find all files that implement LNDL parsing in lionagi. List entry "
    "points and what each does in one sentence each.",
    2: "Summarize what lionagi/operations/operate/operate.py does in 3 "
    "bullets. Cite specific function names and line numbers.",
    3: "Survey lionagi/operations/. For each .py file, give a one-line "
    "purpose statement. Order by importance.",
}


def _build_branch() -> Branch:
    chat_model = iModel(model="openai/gpt-5.4-mini")
    branch = Branch(system=SYSTEM_PROMPT, chat_model=chat_model)
    toolkit = CodingToolkit(workspace_root=WORKSPACE, notify=False)
    branch.register_tools(toolkit.bind(branch, include=CodingToolkit.READ_ONLY))
    return branch


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_task(task_id: int, question: str, rounds: int) -> None:
    print(f"\n{'='*70}\nTASK {task_id}: {question}\n{'='*70}")
    branch = _build_branch()
    t0 = time.time()

    try:
        result = await branch.operate(
            instruction=question,
            actions=True,
            lndl=True,
            lndl_rounds=rounds,
            response_format=ExploreFindings,
        )
    except Exception as e:
        print(f"[FAILED {time.time()-t0:.1f}s] {type(e).__name__}: {e}")
        return

    elapsed = time.time() - t0
    print(f"[{elapsed:.1f}s, {len(branch.messages)} messages]")

    if not hasattr(result, "summary"):
        print(f"Unexpected: {type(result).__name__} | {repr(result)[:400]}")
        return

    print(f"\nSummary:\n  {result.summary}")
    print(f"Confidence: {result.confidence}")
    for label, items in [("Files", result.files), ("Key lines", result.key_lines)]:
        if items:
            print(f"\n{label}:")
            for item in items[:10]:
                print(f"  - {item}")

    # Citation check
    real = [f for f in (result.files or []) if isinstance(f, str) and "\n" not in f]
    exist = [f for f in real if (WORKSPACE / f).exists()]
    print(f"\nCitation accuracy: {len(exist)}/{len(real)}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=None, help="Run task N (1-3)")
    ap.add_argument("--rounds", type=int, default=4)
    args = ap.parse_args()

    tasks = {args.task: TASKS[args.task]} if args.task else TASKS
    for tid, q in tasks.items():
        await run_task(tid, q, args.rounds)


if __name__ == "__main__":
    asyncio.run(main())
