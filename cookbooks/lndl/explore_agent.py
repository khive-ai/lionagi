"""LNDL coding agent — codebase exploration with read-only tools.

Multi-round LNDL (``lndl_rounds=4``): the model issues tool calls in
early rounds without an OUT block, sees results in chat history, then
commits OUT once it has enough information. ReAct-flavored LNDL — the
right shape for codebase exploration where the answer depends on tool
output.

Tools come from ``lionagi.tools.coding.CodingToolkit`` filtered to its
``READ_ONLY`` subset (``reader``, ``search``, ``context``). Editor /
bash / sandbox / subagent are NOT registered, so any attempt to call
them fails at lookup — defense in depth, not just a prompt promise.

Run::

    uv run python cookbooks/_lndl_explore_agent.py [--task=N]

Tasks (each scoped to the lionagi repo):
    1. Locate: "Find files implementing LNDL parsing"
    2. Summarize: "Summarize what operate.py does in 3 bullets"
    3. Survey: "Map operations/ — list each file's purpose"
"""

import argparse
import asyncio
import time
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from lionagi import Branch, iModel
from lionagi.tools.coding import CodingToolkit

load_dotenv()

WORKSPACE = Path("/Users/lion/projects/libs/opensrc/lionagi").resolve()


# ─── Output schema ──────────────────────────────────────────────────────────


class ExploreFindings(BaseModel):
    summary: str = Field(description="2-5 sentence summary of what was found.")
    files: list[str] = Field(
        default_factory=list,
        description="Workspace-relative paths inspected during exploration.",
    )
    key_lines: list[str] = Field(
        default_factory=list,
        description="Notable 'path:line: snippet' strings worth a human read.",
    )
    confidence: float = Field(default=0.5, description="Self-rated 0.0-1.0.")


# ─── Agent setup ────────────────────────────────────────────────────────────


SYSTEM_PROMPT = f"""You are a code-exploration agent with READ-ONLY access to {WORKSPACE}.

You have a coding toolkit filtered to read-only actions:
- reader(action='read', path=..., offset=0, limit=200)
- reader(action='list_dir', path=..., recursive=False)
- search(action='grep', pattern=..., path=..., include='*.py', max_results=50)
- search(action='find', pattern='*.py', path=..., max_results=100)
- context(action='status' | 'get_messages' | 'evict' | 'evict_action_results')

Multi-round LNDL strategy:
1. First rounds — issue <lact> calls to gather information. No OUT yet.
2. Middle rounds — read specific files or narrow the grep. Use
   ``<lvar note.X ...>`` to jot findings to reference later.
3. Final round — synthesize via <lvar> declarations and a single OUT.

Cite EXACT path:line numbers from real tool output; never invent them.
"""


TASKS = {
    1: (
        "Find all files that implement LNDL parsing in lionagi. List entry "
        "points and what each does in one sentence each."
    ),
    2: (
        "Summarize what lionagi/operations/operate/operate.py does in 3 "
        "bullets. Cite specific function names and line numbers."
    ),
    3: (
        "Survey lionagi/operations/. For each .py file, give a one-line "
        "purpose statement. Order by importance."
    ),
}


def build_branch() -> Branch:
    chat_model = iModel(model="openai/gpt-5.4-mini")
    branch = Branch(system=SYSTEM_PROMPT, chat_model=chat_model)
    toolkit = CodingToolkit(workspace_root=WORKSPACE, notify=False)
    branch.register_tools(toolkit.bind(branch, include=CodingToolkit.READ_ONLY))
    return branch


# ─── Runner ─────────────────────────────────────────────────────────────────


def _print_quality(result: ExploreFindings) -> None:
    def looks_like_path(s: str) -> bool:
        return isinstance(s, str) and len(s) <= 500 and "\n" not in s

    real = [f for f in (result.files or []) if looks_like_path(f)]
    exist = [f for f in real if (WORKSPACE / f).exists()]
    print("\n── quality ──")
    print(f"  files cited:   {len(result.files or [])}")
    print(f"  files exist:   {len(exist)}/{len(real)}")
    if hallucinated := [f for f in real if f not in exist]:
        print(f"  hallucinated:  {hallucinated[:3]}")


async def run_task(task_id: int, question: str) -> None:
    print(f"\n{'=' * 70}\nTASK {task_id}: {question}\n{'=' * 70}")
    branch = build_branch()
    t0 = time.time()
    try:
        result = await branch.operate(
            instruction=question,
            actions=True,
            lndl=True,
            lndl_rounds=4,
            response_format=ExploreFindings,
        )
    except Exception as e:
        print(f"[FAILED after {time.time() - t0:.1f}s] {type(e).__name__}: {e}")
        return

    elapsed = time.time() - t0
    print(f"[duration {elapsed:.1f}s, messages {len(branch.messages)}]")

    if not hasattr(result, "summary"):
        print(f"Unexpected result type: {type(result).__name__}")
        print(repr(result)[:400])
        return

    print(f"\nSummary:\n  {result.summary}")
    print(f"\nConfidence: {result.confidence}")
    if result.files:
        print("\nFiles cited:")
        for f in result.files[:10]:
            print(f"  - {f}")
    if result.key_lines:
        print("\nKey lines:")
        for line in result.key_lines[:10]:
            print(f"  {line}")
    _print_quality(result)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=int, default=None, help="Run only task N (1, 2, or 3)"
    )
    args = parser.parse_args()
    if args.task is not None:
        if args.task not in TASKS:
            print(f"Unknown task {args.task}. Available: {list(TASKS)}")
            return
        await run_task(args.task, TASKS[args.task])
    else:
        for tid, q in TASKS.items():
            await run_task(tid, q)


if __name__ == "__main__":
    asyncio.run(main())
