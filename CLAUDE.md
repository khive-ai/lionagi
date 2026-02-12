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

- `branch.chat()` → simple LLM call
- `branch.parse()` → structured extraction into Pydantic models
- `branch.operate()` → tool calling with iteration
- `branch.ReAct()` → think-act-observe reasoning loops

### Service Layer (`service/`)

**iModel** wraps any LLM provider behind a uniform interface. Provider resolution happens via `match_endpoint.py`. Providers (in `connections/providers/`): OpenAI, Anthropic, Gemini, Ollama, NVIDIA NIM, Perplexity, Groq/OpenRouter.

Rate limiting, circuit breaking, and retry logic are built into iModel automatically. Hooks provide aspect-oriented extension points.

### Operations (`operations/`)

Operations (chat, parse, operate, ReAct, select, interpret, communicate, act) are standalone modules that Branch methods delegate to. `OperationGraphBuilder` composes them into DAGs executed by `Session.flow()`.

### Tools (`protocols/action/`)

Tool schemas auto-generate from function signatures via `function_to_schema()`. Register with `branch.register_tools()`. Supports sync/async functions and MCP tool configs.

### Utilities (`ln/`)

- `ln/concurrency/`: `alcall()` (parallel async), `bcall()` (batch), `race()`, `retry()`
- `ln/fuzzy/`: `fuzzy_json()` for repairing malformed LLM JSON output
- `ln/types/`: Sentinel system — `Undefined` (intentionally missing) vs `Unset` (not provided) vs `None` (null). Check with `is_sentinel()`.

### Config (`config.py`)

`AppSettings` (pydantic-settings) loads API keys from env vars. Defaults: `LIONAGI_CHAT_PROVIDER=openai`, `LIONAGI_CHAT_MODEL=gpt-4.1-mini`.

## Key Design Patterns

- **Lazy imports**: `__init__.py` uses `__getattr__` to defer all module loading — import time stays O(1).
- **Manager facade**: Branch is thin; real logic lives in MessageManager, ActionManager, iModelManager, DataLogger.
- **Pile + Progression separation**: Storage (dict) and ordering (deque) are independent. Multiple Progressions can index the same Pile.
- **Observable protocol** (`protocols/contracts.py`): Structural typing (V1) — Element auto-satisfies without explicit protocol inheritance.
- **Adaptive serialization**: `element.to_dict(mode="python"|"json"|"db")` handles different output contexts.
