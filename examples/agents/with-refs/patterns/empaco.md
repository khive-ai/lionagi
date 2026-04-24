# Pattern: EMPACO (Embarrassingly Parallel Codex)

Use when: audit scope fits "one prompt per module", and modules are
independent (no cross-module reasoning required at the scan stage).

## Shape

1. **Enumerate** modules (Cargo crates, Python packages, Node workspaces).
2. **Fire** one worker per module, all in parallel. Each worker reads ONLY
   its own module.
3. **Harvest** per-module reports.
4. **Consolidate** — cross-module dedup, priority sort, group by theme.
5. **File** issues or a consolidation report.

## Wall clock

Roughly `max(per_module_time)` regardless of N, because workers run in
parallel. 25–30 Rust crates ≈ 15 minutes of codex/gpt-5.4 xhigh.

## Audit modes

| Mode | Focus |
|------|-------|
| `dry` | DRY violations, repeated patterns, extraction candidates |
| `security` | PII handling, auth gaps, unsafe blocks, input validation |
| `dead-code` | Unused `pub` items, stale `#[allow]`, commented blocks |
| `api-surface` | Handler consistency, missing docs, schema drift |

## When NOT to use

- Cross-module reasoning required at scan stage (e.g. "trace this call chain")
- Modules share so much state that isolating one is meaningless
- Scope is a single file or a handful of functions — one agent is enough

## Output template for each worker

```
You are reviewing the {language} module at `{crate_path}/` inside the monorepo.
Read every source file under `{crate_path}/src/`.

Audit focus: {mode_specific_instructions}

Output format (per finding):
- **Location**: file:line (all occurrences)
- **Impact**: estimated severity or LOC savings
- **Suggestion**: one-line fix

Focus on findings worth ≥10 LOC or ≥3 occurrences. Skip trivial.
```

The consolidation step is NOT parallel. One agent (typically the orchestrator)
reads all N reports, dedupes, prioritizes, and files issues.
