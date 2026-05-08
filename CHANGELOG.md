
# Changelog

All notable changes to lionagi are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.24.0] - 2026-05-08

### Added

- **LNDL** — tag-based structured-output protocol; `branch.operate(lndl=True, lndl_rounds=N, lndl_retries=K)` collapses many parallel tool calls into one response. Three `OUT{}` forms (explicit, shortcut, nested groups), `note.X` cross-round scratchpad, auto-injected system prompt
- **`branch.ReActStream(lndl=True)`** — LNDL threaded through every internal operate call; `lndl_rounds` controls within-beat exploration, `max_extensions` controls outer ReAct loop
- **`LndlTrace`** opt-in diagnostics — pass `trace=LndlTrace()` to record per-round outcomes, errors, schema, action counts. `classify_chunk()` (syntactic), `classify_result()` (final), `branch.lndl_chunks(since=)` helpers. `trace=None` (default) = zero overhead
- **`Formatter` protocol** — `JsonFormatter` (default) and `LndlFormatter`; pluggable `render_schema`/`render_format`/`render_tools`/`parse`
- **TypeScript-style schema display** — ~47% smaller than raw `model_json_schema()`; one-line tool signatures replace 12-line YAML blocks
- **CodingToolkit hardening** — bounded `islice` reads, deny-globs (`*.pem`, `.ssh/*`, `secrets.*`), `--exclude-dir` defaults, grep `-e --` injection guard, `find -prune` (~10x faster), `READ_ONLY` bind whitelist
- **Cookbooks** — `lndl/_cases_runner.py` (15 cases), `lndl/explore_agent.py`, `lndl/chain_analysis.py` (phased pipeline), `lndl/react_analysis.py` (ReActStream + trace)
- **Docs** — `docs/reference/lndl.md` (when to use vs JSON, three modes, syntax, diagnostics, failure modes)

### Changed

- `ReActAnalysis` trimmed — removed `planned_actions` (unconsumed) and `action_strategy: Literal[...]` (LNDL-fragile); actions always concurrent. `PlannedAction` still exported for back-compat
- `LndlFormatter` alias generator emits 2-char aliases (`a1, b1, c1, ...`) from the start — 676 collision-free, prevents single-letter reuse on large schemas
- LNDL system prompt adds Rule 8 (aliases must be unique within a response)
- LNDL reparse prompt preserves intended `<lact>` tool calls (anti-fix-mode-contagion); continuation prompt requires real tool calls vs prose plans
- `InstructionContent` uses `prompt_transformer` instead of internal `_schema_dict`/`_model_class`; `response_model_cls`/`schema_dict`/`request_model` deprecated (removed in v1.0)
- Markdown section headers (`## Task Instruction`, `## Tools`, `## Schema`, `## ResponseFormat`) for prompt layout

### Fixed

- `operate()` leaked `action_responses` field into LLM prompt — now uses `request_type`
- `ActionResponse` messages had `None` sender/recipient — now `sender=tool_id, recipient=branch_id`
- `fuzzy_validate_pydantic` crashed when `fuzzy_match=False` (unbound `model_data`)

## [0.23.1] - 2026-05-02

### Added

- Provider registry (`EndpointRegistry`, `@register` decorator, `iModel(provider=..., endpoint=...)`)
- AG2 GroupChat endpoint (`pip install lionagi[ag2]`)
- `Note` model with `deep_update()` and nested path access

### Changed

- Provider modules reorganized to `providers/{company}/{endpoint}/` packages
- `CLIEndpoint` → `AgenticEndpoint` (alias kept one version)

### Fixed

- Registry thread-safety on concurrent `iModel` instantiation
- Single-endpoint provider matching without `endpoint=` kwarg

## [0.23.0] - 2026-04-27

### Added

- Coding agent infrastructure — `AgentConfig` presets, `create_agent()`, `PermissionPolicy`, hooks, settings
- `SandboxSession` — git-worktree-isolated editing
- DeepSeek native provider, Pi CLI endpoint
- `li play NAME --help`, 250+ new tests

### Fixed

- `li o flow --save` regression, flaky timing tests, `ValidationError` kwargs

## [0.22.9] - 2026-04-24

### Security

- Symlink containment, FlowAgent/FlowOp ID validation, `--max-ops` enforcement, save-path containment
- Pin `lxml>=6.1.0`, `python-dotenv>=1.2.2`

### Added

- `li skill NAME`, `li play NAME [args]` with typed args schema
- `--team-attach`, `--bypass`, `--add-dir`, agent directory layout, `examples/`

### Fixed

- Playbook args collision, codex worker sandbox scope

## [0.22.8] - 2026-04-21

### Fixed

- `StreamChunk` propagation through iModel layer

## [0.22.7] - 2026-04-20

### Added

- `li --version`, `--background` flow respawn, background progress tracking

### Fixed

- `--show-graph` on macOS, codex `reasoning_effort=max` → `xhigh` clamp

### Changed

- Docs overhaul — CLI-first restructure, 74% byte reduction

## [0.22.6] - 2026-04-20

### Added

- Two-level flow DAG: `FlowAgent` (persistent branch) + `FlowOp` (DAG node)
- Run-scoped persistence at `~/.lionagi/runs/{run_id}/`
- `Middle` protocol for `operate()` callable contract
- Team file locking, timestamped read receipts, `--from-op` on team send
- Per-agent artifact directories with cross-agent relative paths

### Changed

- `branch.operate()` absorbs `branch.instruct()` — auto-dispatches API vs CLI
- Branch storage moved to `runs/{run_id}/branches/`

### Removed

- `branch.instruct()` — use `branch.operate(instruction=...)`
- Flat artifact dumps — superseded by per-agent directories

### Fixed

- `operate()` crash on raw-text CLI output
- Flow DAG schema conflation (agent + node + instruction were one type)
- Team concurrent-write race

## [0.22.2–0.22.5] - 2026-04-19

- `li o flow` — DAG orchestration with `depends_on` edges and critic control nodes
- Stream persist (write-ahead logging for `branch.run()`)
- Unified `instruct` → `operate` routing; CLI payload filtering fixes
- Security: authlib >=1.6.11, python-multipart >=0.0.26, pillow >=12.2.0

## [0.22.0–0.22.1] - 2026-04-18

- `branch.run()` async generator for CLI streams; `StreamChunk` type
- Agent profiles (`.lionagi/agents/{name}.md`), `li team`, `--team-mode`
- Multi-model conversations via `chat_model` param

## [0.21.1] - 2026-04-17

- `li orchestrate fanout` — parallel fan-out with optional synthesis

## [0.21.0] - 2026-04-15

- `li` CLI launch with `--theme`, `--yolo`, `--verbose`; model spec parsing

## [0.20.2–0.20.4] - 2026-03-16 to 2026-04-11

- Firecrawl integration, Tavily search, 4 cookbook recipes
- Event lifecycle → template method; CLI provider updates
- Fixes: `FieldModel` list annotation, SIGINT isolation, 20 discovery sprint issues

## [0.20.0–0.20.1] - 2026-02-13

- `NodeConfig`/`create_node` factory, `Flow` container, `Broadcaster` pub/sub
- Graph algorithms (topo sort, pathfinding), `Pile` callable filters
- 187 doc example tests; MCP thread-safety fixes

## [0.19.0–0.19.2] - 2026-02-11

- Native Gemini API integration
- `CLIEndpoint`, async context managers on `iModel`/`Branch`

## [0.18.0–0.18.6] - 2025-10-09

- `ChatParam`/`ParseParam` replace context classes; `operate_v0` removed → unified `operate`
- `LION_SYSTEM_MESSAGE` env var; `Analysis` class; AnyIO structured concurrency

## [0.17.0] - 2025-09-14

- `ClaudeCodeEndpoint` removed (deprecated sdk); dependency cleanup

## [0.16.0] - 2025-09-02

- V1 Observable Protocol; `CompletionStream`

## [0.15.0] - 2025-08-16

- Structured concurrency (`CancelScope`, `TaskGroup`); `Pile` generic TypeVar; `hash_dict`

## [0.14.0] - 2025-07-22

- `DependencyAwareExecutor`; `Session.flow()` DAG execution; AnyIO task groups

## [0.13.0] - 2025-07-13

- Claude Code provider with session management and path validation

## [0.12.0] - 2025-05-14

- XML/JSON parsing utils; async file I/O; pre-computed adapter registry

## [0.11.0] - 2025-05-01

- `Research` models (`ResearchFinding`, `PotentialRisk`); `concat` utility

## [0.10.0] - 2025-03-19

- Pandas adapters; `BaseForm`/`FlowDefinition`/`Report`; field models; `ln` namespace

## [0.9.0] - 2025-01-24

- `LION_SYSTEM_MESSAGE`; `Analysis` class; `as_readable` YAML formatting

## [0.8.0] - 2025-01-18

- `FlowStep`/`FlowDefinition`; Exa search; `OperationManager`; action batching

## [0.7.0] - 2025-01-13

- `interpret`/`select`/`translate` ops; Groq/Perplexity/OpenRouter providers

## [0.6.0] - 2025-01-04

- `MailManager`; Branch JSON serialization; `LiteiModel`

## [0.5.0] - 2024-12-16

- LION2 protocol; class registry; `ReactInstruct`

## [0.4.0] - 2024-10-30

- `lion-core` integration; LangChain/LlamaIndex adapters

## [0.3.0] - 2024-10-06

- `uv` replaces Poetry; pre-commit hooks; CI with dependabot

## [0.2.0] - 2024-05-28

- Ollama integration; token compressor (experimental)

## [0.1.0] - 2024-04-10

- `Branch` and tree-node architecture; tool manager; async queue; knowledge graph; form/report system

---

See git history for versions before v0.1.0 (tags v0.0.102–v0.0.316).
