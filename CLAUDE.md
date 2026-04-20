# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Read AGENT.md first — it covers commands, workflow, and coding standards. This file adds architecture depth.

## Architecture

**lionagi** is a provider-agnostic LLM orchestration SDK. The core abstraction flow:

```text
Session (multi-branch orchestrator)
  └─ Branch (single conversation thread)
       ├─ MessageManager  → Pile[RoledMessage] + Progression (message history)
       ├─ ActionManager   → Pile[Tool] (tool registry + invocation)
       ├─ iModelManager   → iModel instances (chat, parse, embed)
       └─ DataLogger      → activity logs

iModel (unified provider interface)
  ├─ Endpoint (provider-specific config + request/response handling)
  ├─ RateLimitedAPIExecutor (queue + rate limiting)
  └─ HookRegistry (pre/post-invoke aspect hooks)
```

### Core Primitives (`protocols/`)

- **Element** (`protocols/generic/element.py`): Base for everything. UUID identity + timestamp + metadata. All significant objects inherit from it.
- **Pile** (`protocols/generic/pile.py`): O(1) dict-keyed collection (by UUID, not index). Thread-safe and async-safe via `@synchronized`/`@async_synchronized` decorators. Use `pile[uuid]`, not `pile[0]`.
- **Progression** (`protocols/generic/progression.py`): Ordered deque of UUIDs, decoupled from Pile storage. Allows multiple orderings over the same Pile without copying.
- **Node** (`protocols/graph/node.py`): Element + arbitrary content + optional embedding vector.
- **Graph** (`protocols/graph/graph.py`): Directed graph of Nodes and Edges with adjacency mapping.

### Message Hierarchy (`protocols/messages/`)

```text
RoledMessage (base)
├── System, Instruction, AssistantResponse
├── ActionRequest (tool call from LLM)
└── ActionResponse (tool result back to LLM)
```

### Session & Branch (`session/`)

**Session** manages multiple Branches. **Branch** is the primary API surface — a facade over four managers. All LLM operations are Branch methods that delegate to managers:

- `branch.chat()` → simple LLM call (API providers only)
- `branch.run()` → async generator streaming chunks from CLI endpoints (claude_code, codex)
- `branch.parse()` → structured extraction into Pydantic models
- `branch.operate()` → **universal structured operation** — tool calling with iteration, structured output, optional streaming. Routes via the `Middle` protocol: `communicate` for API endpoints, `run_and_collect` for CLI endpoints. Accepts `stream_persist` / `persist_dir` for CLI chunk logging and `middle=<callable>` to override dispatch explicitly.
- `branch.ReAct()` → think-act-observe reasoning loops

**Middle protocol** (`operations/types.Middle`): a callable that advances the branch by one assistant turn — takes (branch, instruction, chat_param, parse_param, clear_messages, skip_validation), returns text/dict/BaseModel. Canonical implementations: `operations/communicate/communicate.py` (one-shot chat+parse) and `operations/run/run.run_and_collect` (stream accumulation + parse). Custom middles can be injected for caching, retry-wrapping, recorded-replay in tests.

### Service Layer (`service/`)

**iModel** wraps any LLM provider behind a uniform interface. Provider resolution happens via `match_endpoint.py`. Providers (in `connections/providers/`): OpenAI, Anthropic, Gemini, Ollama, NVIDIA NIM, Perplexity, Groq/OpenRouter.

Rate limiting, circuit breaking, and retry logic are built into iModel automatically. Hooks provide aspect-oriented extension points.

### Operations (`operations/`)

Operations (chat, parse, operate, ReAct, select, interpret, communicate, run, act) are standalone modules that Branch methods delegate to. `OperationGraphBuilder` composes them into DAGs executed by `Session.flow()`.

**Multi-op-per-branch**: `Session.flow()` pre-allocates branches once; operations with the same `branch=` reference reuse that Branch (no clone). This enables an agent to run several DAG nodes with accumulating message history — the core building block for the CLI's two-level flow pattern (one FlowAgent → one Branch → multiple FlowOp invocations).

### Tools (`protocols/action/`)

Tool schemas auto-generate from function signatures via `function_to_schema()`. Register with `branch.register_tools()`. Supports sync/async functions and MCP tool configs.

### Utilities (`ln/`)

- `ln/concurrency/`: `alcall()` (parallel async), `bcall()` (batch), `race()`, `retry()`
- `ln/fuzzy/`: `fuzzy_json()` for repairing malformed LLM JSON output
- `ln/types/`: Sentinel system — `Undefined` (intentionally missing) vs `Unset` (not provided) vs `None` (null). Check with `is_sentinel()`.

### Config (`config.py`)

`AppSettings` (pydantic-settings) loads API keys from env vars. Defaults: `LIONAGI_CHAT_PROVIDER=openai`, `LIONAGI_CHAT_MODEL=gpt-4.1-mini`.

### CLI Layer (`lionagi/cli/`)

The `li` command in `cli/main.py` is the user-facing entry point. Subcommands:

- `cli/agent.py` — `li agent`: single-agent one-shot or resumed turn.
- `cli/team.py` — `li team`: persistent inbox-style messaging (`~/.lionagi/teams/{id}.json`). Concurrent read-modify-write is serialized under `fcntl.flock` via `_locked_team`. Messages carry `from_op` (tie-to-op) and timestamped `read_by` dict.
- `cli/orchestrate/` — `li o fanout` / `li o flow`:
  - `_orchestration.py` — pattern-agnostic primitives (`OrchestrationEnv`, `setup_orchestration`, `build_worker_branch`, `finalize_orchestration`). Shared phases A/C/G (setup, worker construction, finalize).
  - `flow.py` — two-level DAG: `FlowAgent` (Branch identity with memory) + `FlowOp` (DAG node). Same agent can run multiple ops; branch memory carries across.
  - `fanout.py` — flat parallel workers from one decomposition.
  - `_common.py` — shared schemas (`AgentRequest`), worker/team system-prompt templates.
  - `__main__.py` — makes `python -m lionagi.cli` work for the `--background` subprocess path.

**Schema-driven prompting**: `FlowAgent`, `FlowOp`, `FlowPlan`, `FlowControlVerdict`, `AgentRequest` all use `Field(description=...)` so the LLM sees purpose-built docs in the structured-output schema. Long orchestrator prompts (planning instruction, re-plan guidance, verdict contract) live as `ClassVar[str]` on the schemas — one source of truth next to the shape they describe.

### Persistence layer (`lionagi/cli/_runs.py`)

Every CLI invocation allocates a `RunDir` under `~/.lionagi/runs/{run_id}/`:
- `run.json` — manifest (command, branches, agents, operations, artifact_root)
- `branches/{branch_id}.json` — canonical branch snapshots
- `stream/{branch_id}.buffer.jsonl` — live chunk buffer during stream
- `artifacts/` — only when `--save` is not provided (else artifacts go to user dir)

`run_id` format: `YYYYMMDDTHHMMSS-{uuid6}`. `find_branch(branch_id)` scans runs first, falls back to the legacy `logs/agents/{provider}/` layout. `LIONAGI_RUN_ID` env var lets subprocesses (e.g. `--background`) inherit their parent's run.

### CLI Logging (`lionagi/cli/_logging.py`)

Four named loggers, set up once in `main()` via `configure_cli_logging(verbose)`:
- `lionagi.cli.progress` — `progress()` — INFO when normal, WARNING when verbose (silenced so provider stream takes over)
- `lionagi.cli.hint` — `hint()` — post-run pointers (resume commands), always on
- `lionagi.cli.warn` — `warn()` — prefixes `warning: `, always on
- `lionagi.cli.error` — `log_error()` — prefixes `error: `, always on

Never `print(..., file=sys.stderr)` in CLI code — use these helpers.

## Key Design Patterns

- **Lazy imports**: `__init__.py` uses `__getattr__` to defer all module loading — import time stays O(1).
- **Manager facade**: Branch is thin; real logic lives in MessageManager, ActionManager, iModelManager, DataLogger.
- **Pile + Progression separation**: Storage (dict) and ordering (deque) are independent. Multiple Progressions can index the same Pile.
- **Observable protocol** (`protocols/contracts.py`): Structural typing (V1) — Element auto-satisfies without explicit protocol inheritance.
- **Adaptive serialization**: `element.to_dict(mode="python"|"json"|"db")` handles different output contexts.
