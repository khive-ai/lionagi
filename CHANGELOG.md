
# Changelog

All notable changes to lionagi are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.23.0] - 2026-04-27

### Added

- **Coding agent infrastructure** (`lionagi/agent/`) — `AgentConfig` presets, `create_agent()` factory, `PermissionPolicy` (allowlist/denylist/confirm), built-in hooks, `.lionagi/settings.yaml` loading
- **Sandbox tool** — `SandboxSession` for git-worktree-isolated development (create/diff/commit/merge/discard)
- **DeepSeek native provider** — direct API support with reasoning effort mapping
- **Pi CLI endpoint** — Pi Code subprocess integration with NDJSON streaming
- `li play NAME --help` — playbook-specific help (description + args)
- 250+ new tests (agent infrastructure, operations, regression)

### Fixed

- `li o flow --save` regression from 0.22.6 — agent results now written in both flow and fanout paths
- Flaky timing-based tests replaced with event barriers
- `ValidationError` kwargs bug in test infrastructure

## [0.22.9] - 2026-04-24

### Security

- Symlink containment on `li skill` + `li play` — hostile per-entry symlinks blocked
- `FlowAgent.id` / `FlowOp.id` validated (`^[A-Za-z0-9_-]{1,64}$`) — prevents path escape
- `--max-ops` enforced cumulatively across re-plan rounds
- Save-path containment — `--save DIR` must live under `cwd` or `$HOME`
- Pin `lxml>=6.1.0` (CVE-2026-41066), `python-dotenv>=1.2.2` (CVE-2026-28684)

### Added

- `li skill NAME` — read skill files from `~/.lionagi/skills/`. Also `li skill list`
- `li play NAME [args]` — load and run playbooks from `~/.lionagi/playbooks/`
- Playbook args schema (typed `args:` or `argument-hint:` fallback) with template interpolation
- `--team-attach NAME` — attach to existing team (preserves message history)
- `--bypass` flag — bypass all codex approvals/sandbox (for cloud/codespace)
- `--add-dir` auto-injected for project root (codex workers can write source files)
- Agent directory layout (`<name>/<name>.md` resolved before flat `<name>.md`)
- `examples/` directory with agent, skill, and playbook templates

### Changed

- `--max-agents` renamed to `--max-ops` (deprecated alias kept)
- Effort/synthesis spec validation accepts full CLI value range

### Fixed

- Playbook args collision with base CLI flags
- Codex workers sandboxed to artifact dir only — now get project root via `--add-dir`

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

See git history for the 0.0.x series (v0.0.102–v0.0.316).

---

See git history for versions before v0.1.0 (tags v0.0.102–v0.0.316).
