"""LNDL coding agent — codebase exploration with read-only tools.

Single-round LNDL: the model writes ONE structured response containing tool
calls and final OUT in one go. This works when the model can predict its
output before tools execute (e.g. counting balls, doing math) but is weak
for codebase exploration — the model commits OUT before it can interpret
tool results. For true exploration agents, multi-round LNDL (the krons
``react`` pattern) is needed: model issues tools, observes results in the
next turn, then commits OUT.

Run:
    uv run python cookbooks/_lndl_explore_agent.py [--task=N]

Tasks (each scoped to the lionagi repo):
    1. Locate: "Find files implementing LNDL parsing"
    2. Summarize: "Summarize what operate.py does in 3 bullets"
    3. Survey: "Map operations/ — list each file's purpose"
"""

import argparse
import asyncio
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from lionagi import Branch, iModel

load_dotenv()

WORKSPACE = Path("/Users/lion/projects/libs/opensrc/lionagi").resolve()


def _safe_path(p: str) -> Path:
    """Resolve a (possibly relative) path under the workspace, refusing escapes."""
    raw = Path(p)
    abs_path = raw if raw.is_absolute() else (WORKSPACE / raw)
    abs_path = abs_path.resolve()
    if not str(abs_path).startswith(str(WORKSPACE)):
        raise ValueError(f"Path '{p}' is outside the workspace")
    return abs_path


# ─── Read-only tools ────────────────────────────────────────────────────────


def list_dir(path: str = ".", recursive: bool = False) -> str:
    """List files and subdirectories under ``path`` (relative to workspace).

    Args:
        path: directory path relative to workspace root (default ".").
        recursive: if true, walk the tree (capped at 200 entries).
    """
    target = _safe_path(path)
    if not target.is_dir():
        return f"Error: {path!r} is not a directory"
    items: list[str] = []
    if recursive:
        for p in target.rglob("*"):
            if any(part.startswith(".") and part not in (".", "..") for part in p.parts):
                continue
            rel = p.relative_to(WORKSPACE)
            items.append(("  " * (len(rel.parts) - 1)) + rel.parts[-1])
            if len(items) >= 200:
                items.append("... (truncated at 200 entries)")
                break
    else:
        for p in sorted(target.iterdir()):
            if p.name.startswith("."):
                continue
            items.append(p.name + ("/" if p.is_dir() else ""))
    return "\n".join(items) if items else "(empty)"


def read_file(path: str, offset: int = 0, limit: int = 200) -> str:
    """Read a text file with line numbers.

    Args:
        path: file path relative to workspace root.
        offset: zero-based line offset (default 0).
        limit: max number of lines to return (default 200, hard cap 600).
    """
    target = _safe_path(path)
    if not target.is_file():
        return f"Error: {path!r} is not a file"
    limit = max(1, min(int(limit), 600))
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error: {e}"
    lines = text.splitlines()
    sl = lines[offset : offset + limit]
    numbered = [f"{i + offset + 1:>5}  {line}" for i, line in enumerate(sl)]
    out = "\n".join(numbered)
    if len(out) > 8000:
        out = out[:8000] + "\n[... truncated, call read_file again with offset to continue ...]"
    return out or "(empty range)"


def grep(pattern: str, path: str = ".", include: Optional[str] = None, max_results: int = 30) -> str:
    """Recursively grep for a regex pattern.

    Args:
        pattern: regex pattern (extended).
        path: directory or file path (default ".").
        include: glob like "*.py" to filter file types (optional).
        max_results: max matches to return (default 30).
    """
    target = _safe_path(path)
    cmd = ["grep", "-rn", "-E", pattern, str(target)]
    if include:
        cmd += ["--include", include]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return "Error: grep timed out"
    except FileNotFoundError:
        return "Error: grep not installed"
    if r.returncode == 2:
        return f"Error: {r.stderr.strip()}"
    lines = [ln for ln in r.stdout.splitlines() if ln][:max_results]
    return "\n".join(lines) if lines else f"No matches for {pattern!r}"


def find_files(pattern: str, path: str = ".", max_results: int = 50) -> str:
    """Find files by name glob (e.g. ``*.py``).

    Args:
        pattern: name glob pattern.
        path: starting directory (default ".").
        max_results: cap on results (default 50).
    """
    target = _safe_path(path)
    cmd = ["find", str(target), "-name", pattern, "-type", "f"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return "Error: find timed out"
    except FileNotFoundError:
        return "Error: find not installed"
    if r.returncode != 0 and r.stderr:
        return f"Error: {r.stderr.strip()}"
    lines = r.stdout.splitlines()[:max_results]
    rel = [str(Path(p).relative_to(WORKSPACE)) for p in lines if p.startswith(str(WORKSPACE))]
    return "\n".join(rel) if rel else f"No files matching {pattern!r}"


TOOLS = [list_dir, read_file, grep, find_files]


# ─── Output schema ──────────────────────────────────────────────────────────


class ExploreFindings(BaseModel):
    summary: str = Field(description="2-5 sentence summary of what was found.")
    files: list[str] = Field(default_factory=list, description="Relative paths inspected.")
    key_lines: list[str] = Field(
        default_factory=list,
        description="Notable 'path:line: snippet' strings worth a human read.",
    )
    confidence: float = Field(default=0.5, description="Self-rated 0.0-1.0.")


# ─── Agent setup ────────────────────────────────────────────────────────────


SYSTEM_PROMPT = f"""You are a code-exploration agent. You have read-only access to:

- list_dir(path, recursive)
- read_file(path, offset, limit)
- grep(pattern, path, include, max_results)
- find_files(pattern, path, max_results)

Workspace: {WORKSPACE}
All paths are RELATIVE to the workspace root. You may NOT write, edit, or run
shell commands beyond the provided tools.

Strategy:
1. Start by listing the relevant directory to orient yourself.
2. Use grep / find_files to locate candidates by name or content.
3. Read the most relevant files to confirm.
4. Cite EXACT paths and line numbers — never invent them.

Use LNDL: <lact alias>tool(args)</lact> for tool calls, <lvar Field.name a>...</lvar>
for direct values. Bind tool results to fields via <lact Field.name a>tool(...)</lact>.
Commit your final answer in OUT{{summary: [s], files: [...], key_lines: [...], confidence: [c]}}.
"""


def build_branch() -> Branch:
    chat_model = iModel(model="openai/gpt-5.4-mini")
    return Branch(system=SYSTEM_PROMPT, tools=TOOLS, chat_model=chat_model)


# ─── Tasks ──────────────────────────────────────────────────────────────────


TASKS = {
    1: "Find all files that implement LNDL parsing in lionagi. List entry points and what each does in one sentence each.",
    2: "Summarize what lionagi/operations/operate/operate.py does in 3 bullets. Cite specific function names and line numbers.",
    3: "Survey lionagi/operations/. For each .py file, give a one-line purpose statement. Order by importance.",
}


async def run_task(task_id: int, question: str) -> None:
    print("=" * 70)
    print(f"TASK {task_id}: {question}")
    print("=" * 70)

    branch = build_branch()
    t0 = time.time()
    try:
        result = await branch.operate(
            instruction=question,
            actions=True,
            lndl=True,
            lndl_retries=2,
            response_format=ExploreFindings,
        )
        elapsed = time.time() - t0
        msg_count = len(branch.messages)

        print(f"\n[duration: {elapsed:.1f}s, messages: {msg_count}]\n")

        # Show last assistant LNDL for inspection
        for m in reversed(branch.messages):
            if hasattr(m.content, "assistant_response"):
                raw = m.content.assistant_response
                print("--- raw LNDL ---")
                for ln in str(raw).splitlines()[:60]:
                    print(f"  | {ln}")
                print("--- end ---\n")
                break

        if hasattr(result, "summary"):
            print(f"Summary:\n  {result.summary}\n")
            print(f"Confidence: {result.confidence}\n")
            print("Files cited:")
            for f in (result.files or [])[:10]:
                print(f"  - {f}")
            print("\nKey lines:")
            for line in (result.key_lines or [])[:10]:
                print(f"  {line}")

            print("\n--- quality assessment ---")

            def _looks_like_path(s: str) -> bool:
                if not isinstance(s, str):
                    return False
                if len(s) > 500 or "\n" in s:
                    return False
                return True

            real_check = [f for f in (result.files or []) if _looks_like_path(f)]
            files_real = [f for f in real_check if (WORKSPACE / f).exists()]
            print(f"Files that actually exist: {len(files_real)}/{len(result.files or [])}")
            malformed = [f for f in (result.files or []) if not _looks_like_path(f)]
            if malformed:
                print(f"  MALFORMED paths (not single-line): {len(malformed)}")
            missing = [f for f in real_check if f not in files_real]
            if missing:
                print(f"  HALLUCINATED: {missing[:5]}")

            if hasattr(result, "action_responses"):
                print(f"Tool calls executed: {len(result.action_responses or [])}")
        else:
            print(f"Unexpected type: {type(result).__name__}")
            print(repr(result)[:500])
    except Exception as e:
        print(f"\n[FAILED after {time.time() - t0:.1f}s] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
    print()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=None)
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
