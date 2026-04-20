
# CLAUDE.md

Claude Code guidance. Read AGENT.md first (commands, workflow, standards). This file adds architecture depth.

## Architecture

```text
Session (multi-branch orchestrator)
  └─ Branch (single conversation thread)
       ├─ MessageManager  → Pile[RoledMessage] + Progression
       ├─ ActionManager   → Pile[Tool]
       ├─ iModelManager   → iModel (provider wrapper with rate limiting + hooks)
       └─ DataLogger      → activity logs
```

### Core Primitives (`protocols/`)

- **Element** (`protocols/generic/element.py`): UUID + timestamp + metadata. Base for all objects.
- **Pile** (`protocols/generic/pile.py`): O(1) UUID-keyed. Thread/async-safe. `pile[uuid]` not `pile[0]`.
- **Progression** (`protocols/generic/progression.py`): Ordered UUID deque, decoupled from Pile.
- **Node/Graph** (`protocols/graph/`): Node = Element + content + embedding. Graph = directed.

### Message Types (`protocols/messages/`)

```text
RoledMessage
├── System, Instruction, AssistantResponse
├── ActionRequest (tool call from LLM)
└── ActionResponse (tool result back to LLM)
```

### Session & Branch (`session/`)

**Branch** — facade over four managers, primary API surface:

- `branch.chat()` / `branch.run()` — LLM call (API) / async stream (CLI: claude_code, codex)
- `branch.parse()` — structured extraction into Pydantic models
- `branch.operate(instruction=...)` — universal op: tools, structured output, Middle routing. **`branch.instruct()` removed in 0.22.6.**
- `branch.ReAct()` — think-act-observe loops

### Middle Protocol (`operations/types.py`)

`Middle` = callable `(branch, instruction, ...) → text|dict|BaseModel` advancing branch one turn.

- `operations/communicate/communicate.py` — one-shot chat+parse (API endpoints)
- `operations/run/run.run_and_collect` — stream accumulation + parse (CLI endpoints)

Override: `branch.operate(instruction=..., middle=my_callable)`. Force stream: `stream_persist=True`.

### Service Layer (`service/`)

**iModel** wraps any provider via `match_endpoint.py`. Providers in `connections/providers/`: OpenAI, Anthropic, Gemini, Ollama, NVIDIA NIM, Perplexity, Groq/OpenRouter.

### Operations (`operations/`)

Modules: chat, parse, operate, ReAct, select, interpret, communicate, run, act. `Session.flow()` executes DAGs via `OperationGraphBuilder`. Same `branch=` reuses Branch without cloning — state accumulates (FlowAgent → many FlowOps).

### Tools / Utilities / Config

- **Tools** (`protocols/action/`): schemas via `function_to_schema()`, registered with `branch.register_tools()`. Sync/async + MCP.
- **Utilities** (`ln/`): `alcall()`, `bcall()`, `race()`, `retry()` · `fuzzy_json()` for malformed LLM JSON · `Undefined`/`Unset` sentinels (`is_sentinel()`).
- **Config** (`config.py`): `AppSettings` from env. Defaults: `LIONAGI_CHAT_PROVIDER=openai`, `LIONAGI_CHAT_MODEL=gpt-4.1-mini`.

### CLI Architecture (`lionagi/cli/`)

- `cli/agent.py` — `li agent`: one-shot or resumed turn
- `cli/team.py` — `li team`: inbox (`~/.lionagi/teams/{id}.json`), concurrent writes via `fcntl.flock`
- `cli/orchestrate/` — `li o fanout` / `li o flow`:
  - `flow.py` — FlowAgent + FlowOp DAG. `--team-mode` enables `li team` routing mid-pipeline.
  - `_common.py` — `AgentRequest` schema + `TEAM_COORD_SECTION` worker prompt template
  - `fanout.py` — flat parallel workers · `_orchestration.py` — shared setup/finalize

`FlowAgent`, `FlowOp`, `FlowPlan`, `FlowControlVerdict` use `Field(description=...)` — schema-driven prompting.

### Persistence (`lionagi/cli/_runs.py`)

Every run: `~/.lionagi/runs/{run_id}/` (`YYYYMMDDTHHMMSS-{uuid6}`).

- `run.json` — manifest · `branches/{id}.json` — snapshots · `stream/{id}.buffer.jsonl` — live chunks
- `artifacts/` when `--save` not provided · `find_branch(id)` scans `~/.lionagi/runs/` manifests

### CLI Logging (`lionagi/cli/_logging.py`)

`configure_cli_logging(verbose)`. Never `print(..., file=sys.stderr)` in CLI code:
`progress()` (silenced when verbose) · `hint()` (post-run) · `warn()` (warning prefix) · `log_error()` (error prefix)

## Key Design Patterns

- **Lazy imports**: `__init__.py` uses `__getattr__` — import time O(1).
- **Manager facade**: Branch thin; logic in MessageManager, ActionManager, iModelManager, DataLogger.
- **Pile + Progression**: Storage (dict) and ordering (deque) are independent. Multiple orderings over same Pile.
- **Observable** (`protocols/contracts.py`): Structural typing — Element auto-satisfies without explicit inheritance.
- **Serialization**: `element.to_dict(mode="python"|"json"|"db")`.
