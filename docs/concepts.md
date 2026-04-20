# Concepts

Six terms you need to use lionagi. For full API tables see [api/](api/index.md).

---

## Branch

A single conversation thread. Holds message history, registered tools, and model config.
All LLM calls happen through a Branch.

```python
import lionagi as li
branch = li.Branch(
    chat_model=li.iModel(model="gpt-4o-mini"),
    system="You are a technical writer.",
)
result = await branch.communicate("What is a DAG?")
```

**Use when**: you need a stateful unit of LLM work in Python.
**Don't use a single Branch for parallel workers** — give each worker its own Branch
(or let `li o fanout` / `li o flow` do it for you).

→ Full reference: [`Branch`](api/branch.md)

---

## Session

Orchestrates multiple Branches with a shared in-process message bus and a DAG engine.

```python
session = li.Session()
session.include_branches([b1, b2])
session.send(sender=b1.id, recipient=b2.id, content="draft ready")
await session.sync()
```

**Use when**: embedding lionagi in Python and needing programmatic multi-branch control.
**Don't use directly** if you're running `li o flow` — the CLI manages Session internally.

→ Full reference: [`Session`](api/session.md)

---

## flow

A DAG of named agent operations. Dependencies are declared explicitly; independent steps
execute in parallel. The orchestrator LLM plans the graph before execution starts.

```bash
li o flow claude/sonnet "Research async patterns and write a guide" --save ./out
li o flow claude/sonnet "Research async patterns and write a guide" --dry-run
```

**Use when**: you have ≥2 steps that could run concurrently, or want role-typed agents
(researcher → implementer → reviewer) with separate memory.
**Don't use** if your pipeline is strictly sequential — `branch.operate()` in a loop is simpler.

→ Cookbook: [Multi-model pipeline](cookbook/multi-model-pipeline.md)

---

## team

Persistent, file-backed inbox for coordinating agents across separate CLI invocations.
Messages survive process restarts, stored at `~/.lionagi/teams/{id}.json`.

```bash
li team create "research-team" -m "researcher,writer,reviewer"
li team send "draft ready" --team research-team --to writer --from researcher
li team receive --team research-team --as writer
```

**Use when**: coordination spans separate CLI runs or background processes.
**Don't use** within a single `li o flow` — the orchestrator's dependency graph is sufficient.

→ Cookbook: [Team coordination](cookbook/team-coordination.md)

---

## operate

The universal Python method for an LLM turn with tool invocation, structured output,
and streaming. Automatically routes to the right backend (API vs CLI endpoint).

```python
from pydantic import BaseModel
class Result(BaseModel):
    summary: str
    risk: str

result = await branch.operate(
    instruction="Analyze this diff for security issues:\n" + diff,
    response_format=Result,
)
```

**Use when**: you need tools, structured output, or streaming in Python.
**Don't use** for raw message objects — use `branch.chat()`. For live stream chunks — use `branch.run()`.

→ Full reference: [`operate()`](api/operations.md)

---

## persist — run_id

Every CLI invocation writes state to `~/.lionagi/runs/{run_id}/` (format: `YYYYMMDDTHHMMSS-{uuid6}`).

```text
~/.lionagi/runs/20260420T103404-abc123/
├── run.json          ← manifest
├── branches/         ← branch snapshots
├── stream/           ← live chunks (stream_persist=True)
└── artifacts/        ← agent-written files
```

Resume a previous run:

```bash
li agent -r "follow up on the auth module"          # last branch
li agent -r b_abc456 "deepen section 3"             # specific branch
```

**Use when**: you want reproducibility, inspection, or resume without re-running.
**Persist is CLI-only** — driving Branch from Python? Use `branch.to_df()` for history export.

→ Cookbook: [Resumable background](cookbook/resumable-background.md)

---

Next: [CLI reference](cli-reference.md) for full flag tables, or [API reference](api/index.md)
for the Python SDK surface.
