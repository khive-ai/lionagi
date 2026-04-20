# lionagi

Provider-agnostic LLM orchestration SDK. Run a multi-agent flow from the CLI in one command; use the Python API for programmatic control.

---

## Start here

New to lionagi? Follow this path:

1. [Install](getting-started/install.md) — `uv add lionagi` + one env var, verify with `li --help`
2. [First flow](getting-started/first-flow.md) — run `li o flow` in under 5 minutes
3. [Concepts](concepts.md) — Branch, Session, flow, team, operate

---

## Cookbook

Five runnable scenarios with copy-paste commands and expected output.

| Scenario | What it builds |
|----------|----------------|
| [Codebase audit](cookbook/codebase-audit.md) | Structured repo report via fan-out workers |
| [Research synthesis](cookbook/research-synthesis.md) | N parallel workers merged into one document |
| [Multi-model pipeline](cookbook/multi-model-pipeline.md) | Tasks routed across providers in a DAG |
| [Team coordination](cookbook/team-coordination.md) | Agents exchange signals mid-run |
| [Resumable background run](cookbook/resumable-background.md) | Long jobs that survive terminal close |

---

## Reference

| What you need | Where to go |
|---------------|-------------|
| All CLI flags and subcommands | [CLI reference](cli-reference.md) |
| `Branch`, `Session`, `flow` Python API | [API reference](api/index.md) |
| Troubleshooting | [Troubleshooting](reference/troubleshooting.md) |
| How to contribute | [Contributing](contributing.md) |

---

## Migration

Upgrading from 0.22.5? The [migration guide](migration/0.22.5-to-0.22.6.md) covers every breaking change.
