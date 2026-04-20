# AGENT.md

Practical guide for all coding agents working in the LionAGI repository.

## Mission

LionAGI is an async-first SDK for building and orchestrating AI workflows with structured operations, tool calling, and graph-style execution.

Make minimal, correct changes. Preserve API behavior unless explicitly requested. Add or update tests with every behavioral change. Keep docs in sync when user-facing behavior changes.

## Commands

```bash
uv sync --all-extras                        # Install all deps (NEVER use pip)
uv run pytest                               # Run all tests (parallel, -n auto)
uv run pytest tests/path.py -v              # Run specific test file
uv run pytest tests/path.py::test_func -v   # Run specific test function
uv run pytest -m unit                       # By marker: unit, integration, slow, asyncio, performance
uv run pytest -n0 -s tests/path.py          # Debug: no parallelism, show stdout
uv run pytest --cov=lionagi                 # With coverage
uv run black . && uv run isort .            # Format
pre-commit run -a                           # All pre-commit hooks (black, isort, pyupgrade)
uv build                                    # Build wheel
```

Pytest defaults: `-n auto --dist loadfile --maxfail=5 --tb=short`, async mode auto-detected. CI runs on Python 3.10, 3.11, 3.12, 3.13.

## Repository Map

- `lionagi/session/`: Branch and session-layer orchestration entry points
- `lionagi/operations/`: Operation implementations (communicate, operate, chat, parse, ReAct, run, etc.); `operate` is the universal structured operation routed through a `Middle` protocol (see `operations/types.Middle`)
- `lionagi/protocols/`: Foundational types (messages, graph, generic/action abstractions)
- `lionagi/service/`: Model/service integration and provider connection logic
- `lionagi/ln/`: Low-level utilities and concurrency primitives
- `lionagi/tools/`: Tool interfaces and built-in tools
- `lionagi/cli/`: The `li` command and its subcommands. Entry point in `cli/main.py`; orchestration in `cli/orchestrate/` (`flow.py`, `fanout.py`, shared `_orchestration.py`); persistence in `cli/_runs.py` (`RunDir`, `allocate_run`); logging in `cli/_logging.py`; team messaging in `cli/team.py`. `cli/__main__.py` exists so `python -m lionagi.cli` works (used by `--background`).
- `tests/`: Mirrors package areas; prefer colocated test updates for each code change
- `benchmarks/`: Micro-benchmark runners and baselines for performance regression checks

## Architecture Invariants

Branch behavior contracts (from `lionagi/session/branch.py`):
- `Branch` composes internal managers: MessageManager, ActionManager, iModelManager, DataLogger, OperationManager.
- If `chat_model` is not provided, defaults come from settings: `LIONAGI_CHAT_PROVIDER` and `LIONAGI_CHAT_MODEL`.
- If `parse_model` is not provided, it defaults to the active `chat_model`.
- Passing `system`-related args creates a system message through `MessageManager.add_message(...)`.
- `use_lion_system_message=True` prepends Lion's system prompt before any developer/system content.
- `branch.operate()` is the universal structured-output entry point. CLI endpoints stream (via `run_and_collect`); API endpoints one-shot (via `communicate`). Override dispatch with `middle=<callable>` or force streaming with `stream_persist=True` / `persist_dir=<path>`.
- `Session.flow()` honors pre-set `branch_id` on operations — multiple ops with the same `branch=` reference reuse the Branch without cloning. This is the mechanism behind the CLI's two-level flow (one `FlowAgent` branch, many `FlowOp` nodes).

When editing `session`, `operations`, or `cli/orchestrate/`, preserve these contracts unless the task explicitly requires a behavior change and tests/docs are updated with it.

## Testing Strategy by Change Type

- `lionagi/session/*` -> `tests/session/*`
- `lionagi/operations/*` -> `tests/operations/*` and related `tests/operatives/*`
- `lionagi/protocols/*` -> `tests/protocols/*`
- `lionagi/service/*` -> `tests/service/*`
- `lionagi/ln/*` -> `tests/libs/concurrency/*`, `tests/ln/*`
- `lionagi/cli/*` -> verify via `li` smoke test (editable install, then `li o flow ... --dry-run` for structure, `--bare --yolo` for a short real run). CLI has no unit test suite yet; add smoke assertions in `tests/docs/` when changing user-facing output shape.

**Editable install for CLI smoke**: `uv pip install -e . --python <venv-python>` then `uv cache clean` if the install claims editable but imports from site-packages. Hatchling editable installs sometimes copy files the first time; clearing pycache and reinstalling forces the `.pth` linker.

For performance-sensitive changes:
```bash
uv run python -m benchmarks.concurrency_bench
uv run python -m benchmarks.ln_bench
```

## Coding Standards

- Line length: 79 chars (black, isort, ruff all enforce this)
- Keep code async-safe; avoid blocking calls in async execution paths.
- Follow existing typing patterns; add type hints on new/changed public APIs.
- Keep changes surgical: do not refactor unrelated modules in the same patch.
- Maintain optional dependency boundaries; do not make extras mandatory.
- Preserve backward compatibility unless the request explicitly allows breaking changes.

## Common Pitfalls

- Forgetting `await` in async flows.
- Introducing sync I/O inside hot async paths.
- Changing `Branch` defaults or message behavior without test updates.
- Adding provider-specific assumptions to generic protocol/operation layers.
- Modifying benchmarks without recording why baseline expectations changed.

## Change Workflow

1. Identify the smallest correct fix.
2. Read adjacent code and existing tests before editing.
3. Implement minimal patch.
4. Add/update tests to protect behavior.
5. Run focused tests first, then broader validation.
6. Update docs/examples for user-visible changes.
7. Summarize: what changed, why, how validated, and any residual risk.
