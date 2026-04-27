
# AGENT.md

Practical guide for all coding agents working in the LionAGI repository.

## Mission

LionAGI is an async-first SDK for building AI workflows: structured operations, tool calling, graph execution.

Make minimal, correct changes. Preserve API behavior unless explicitly requested. Update tests and docs with every behavioral change.

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

CI runs on Python 3.10, 3.11, 3.12, 3.13. Async mode is auto-detected.

## Repository Map

- `lionagi/session/` — Branch, session orchestration
- `lionagi/operations/` — chat, parse, operate, ReAct, run, act; `Middle` dispatch
- `lionagi/protocols/` — core types: messages, graph, generic, action
- `lionagi/service/` — iModel, provider connections, rate limiting
- `lionagi/ln/` — concurrency utils (alcall, bcall), fuzzy JSON, sentinels
- `lionagi/tools/` — tool interfaces and built-ins
- `lionagi/cli/` — `li` command; `cli/orchestrate/` (flow, fanout); `cli/_runs.py` (RunDir); `cli/_logging.py`; `cli/team.py`
- `tests/` — mirrors package structure
- `benchmarks/` — micro-benchmark runners and baselines

## Architecture Invariants

- `Branch` composes five managers: MessageManager, ActionManager, iModelManager, DataLogger, OperationManager.
- `branch.operate()`: CLI endpoints stream (via `run_and_collect`); API endpoints one-shot (via `communicate`). Override with `middle=<callable>` or force streaming with `stream_persist=True`.
- `Session.flow()`: multiple ops with the same `branch=` reference reuse the Branch without cloning — this is the CLI two-level flow pattern.

When editing `session/`, `operations/`, or `cli/orchestrate/`: preserve these contracts unless the task explicitly changes behavior with tests + docs.

## Testing Strategy by Change Type

- `lionagi/session/*` → `tests/session/*`
- `lionagi/operations/*` → `tests/operations/*` and related `tests/operatives/*`
- `lionagi/protocols/*` → `tests/protocols/*`
- `lionagi/service/*` → `tests/service/*`
- `lionagi/ln/*` → `tests/libs/concurrency/*`, `tests/ln/*`
- `lionagi/cli/*` → smoke with `li o flow ... --dry-run` for structure or `--bare --yolo` for a short real run; add smoke assertions in `tests/docs/` when changing user-facing output shape.

CLI has no dedicated unit test suite.

## Coding Standards

- Line length: 79 chars (black, isort, ruff all enforce this)
- Keep code async-safe; avoid blocking calls in async execution paths.
- Follow existing typing patterns; add type hints on new/changed public APIs.
- Keep changes surgical: do not refactor unrelated modules in the same patch.
- Preserve backward compatibility unless the request explicitly allows breaking changes.

## Common Pitfalls

- Forgetting `await` in async flows.
- Introducing sync I/O inside hot async paths.
- Changing `Branch` defaults or message behavior without test updates.
- Adding provider-specific assumptions to generic protocol/operation layers.

## Change Workflow

1. Read adjacent code and existing tests.
2. Implement minimal patch.
3. Add/update tests to protect behavior.
4. Run focused tests first; then `uv run pytest`.
5. Update docs for user-visible changes.

## Agent Development

Use the `lionagi/agent/` module to build sandboxed, permission-aware coding agents.

**Create a coding agent**

```python
import asyncio
from lionagi.agent.config import AgentConfig
from lionagi.agent.factory import create_agent

async def main():
    config = AgentConfig.coding()          # file tools + guard hooks + strict path policy
    agent = await create_agent(config)     # returns a wired Branch
    reply = await agent.communicate("Refactor auth.py to use async/await throughout.")
    print(reply)

asyncio.run(main())
```

**Create a research agent**

```python
config = AgentConfig.research()            # web + reader tools + log-only policy
agent = await create_agent(config)
result = await agent.operate(
    instruction="Summarize the latest papers on diffusion models.",
    response_format=Summary,
)
```

**Register custom hooks**

```python
from lionagi.agent.hooks import guard_paths, log_tool_use
from lionagi.agent.config import AgentConfig

config = AgentConfig.coding()
config.hooks.append(guard_paths(allowed=["/tmp/sandbox", "./src"]))
config.hooks.append(log_tool_use(sink="tool_calls.jsonl"))
agent = await create_agent(config)
```

**Use Sandbox for isolated edits**

```python
from lionagi.tools.sandbox import SandboxSession

async with await SandboxSession.create(base_branch="main") as session:
    # agent edits happen inside the worktree
    agent = await create_agent(AgentConfig.coding(), cwd=session.path)
    await agent.communicate("Add type hints to all public functions in auth.py.")
    print(await session.diff())            # inspect changes before committing
    await session.commit("feat: add type hints to auth module")
    await session.merge()                  # fast-forward into main; or session.discard()
```

**Permission policies**

```python
from lionagi.agent.permissions import PermissionPolicy

# Allowlist mode: only listed tools may run
policy = PermissionPolicy(mode="allowlist", tools=["read_file", "list_dir"])

# Confirm mode: prompt before each tool execution
policy = PermissionPolicy(mode="confirm")

config = AgentConfig.coding()
config.permission_policy = policy
```

**Settings** — place `.lionagi/settings.yaml` in the project root to override defaults. Global settings live at `~/.lionagi/settings.yaml`; project settings win on conflict.
