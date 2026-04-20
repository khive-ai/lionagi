# Your First Flow

Run a single agent, then a parallel fanout, then a DAG flow — five commands total.

## Single agent turn

```bash
li agent claude "What are the three laws of thermodynamics?"
```

The model's response prints to stdout. Continue the same conversation:

```bash
li agent -c "And what is the zeroth law?"
```

## Parallel fanout

```bash
li o fanout claude "audit the file structure in this directory" -n 3 --save ./results
```

3 workers run in parallel. Artifacts land in `./results/` — one file per worker.

## DAG flow

Preview the plan before executing:

```bash
li o flow claude "research Python async best practices and write a summary" --dry-run
```

```text
# output:
Planning DAG...
FlowPlan (2 agents, 3 ops, synthesis=True)

Agents:
  r1: researcher
  c1: critic

Operations:
  o1 → r1
  o2 → r1    depends_on: o1
  o3 → c1    depends_on: o2  [CONTROL]

Model resolution:
  r1: codex/gpt-5.4 (profile)
  c1: claude_code/opus-4-7 (profile)
```

Execute:

```bash
li o flow claude "research Python async best practices and write a summary" --save ./results
```

## Run directory

Every `li` invocation writes to `~/.lionagi/runs/{run_id}/`:

```text
~/.lionagi/runs/
└── 20260420T103404-a1b2c3/
    ├── run.json       # manifest: command, branches, agents
    ├── branches/      # branch snapshots
    └── artifacts/     # agent output files
```

`run_id` format: `YYYYMMDDTHHMMSS-{uuid6}`. Resume any branch: `li agent -r <branch_id> "..."`.

## Python API

```python
import asyncio
from lionagi import Branch

async def main():
    b = Branch()
    response = await b.operate(instruction="What is the observer pattern?")
    print(response)

asyncio.run(main())
```

`Branch.operate()` returns the model response as a string. The CLI and Python share the same internals.

Next: [CLI reference](../cli-reference.md) for full flag tables, or [Cookbook scenarios](../cookbook/index.md).
