
# Changelog

All notable changes to lionagi are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

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

### Added

- `li o flow` — DAG-based multi-agent orchestration with `depends_on` edges
- Control nodes for critic-driven iteration loops (up to 3 rounds)
- Stream persist (write-ahead logging for `branch.run()`)

### Fixed

- Unified operation routing (`instruct` → `operate`) across all orchestration paths
- CLI endpoint payload filtering (no more `Invalid parameter` crashes)
- Double `create_payload` bug in CLI endpoints

### Security

- authlib >=1.6.11, python-multipart >=0.0.26, pytest >=9.0.3, pillow >=12.2.0

## [0.22.0–0.22.1] - 2026-04-18

### Added

- `branch.run()` — async generator for CLI endpoint streams
- `StreamChunk` — provider-agnostic streaming type
- Multi-model conversations via `chat_model` param
- Agent profiles (`.lionagi/agents/{name}.md`) with `li agent -a NAME`
- `li team` subcommand + `li o fanout --team-mode`
- Unified Message class with Exchange system

### Fixed

- CLI orchestration guidance coercion and output dedup

## [0.21.1] - 2026-04-17

### Added

- `li orchestrate fanout` — three-phase parallel agent fan-out with optional synthesis
- CLI refactored into `agent.py`, `orchestrate.py`, `_providers.py`, `_persistence.py`

## [0.21.0] - 2026-04-15

### Added

- `li` subagent CLI with `--theme`, `--yolo`, `--verbose` flags
- Model spec format with effort suffix parsing (`claude/opus-4-7-high`)

## [0.20.4] - 2026-04-11

### Fixed

- `Branch.from_dict` round-trip serialization for session persistence

## [0.20.3] - 2026-04-10

### Fixed

- Branch syntax error from pre-commit formatting pass

## [0.20.2] - 2026-03-16

### Added

- Firecrawl integration and ReAct research cookbook
- Async generator safety for CLI providers
- Updated Claude Code and Codex CLI providers to latest API

### Changed

- Event lifecycle refactored to template method pattern

### Fixed

- `FieldModel` double-wrapped list annotation (`list[list[T]]` → `list[T]`)
- Spec validators not attached to models; exception propagation in operations
- CLI subprocess SIGINT isolation with regression tests
- Bumped vulnerable transitive dependencies
- 20 discovery sprint issues (P0–P3)

## [0.20.1] - 2026-03-07

### Added

- Tavily search and extract integration
- 4 new cookbook recipes

## [0.20.0] - 2026-02-13

### Added

- `NodeConfig`, `create_node` factory, and Node lifecycle methods
- `Flow` container and `Broadcaster` pub/sub system
- Graph algorithms: topological sort, pathfinding, `get_tails`
- O(1) membership and workflow ops to `Progression`
- Callable filter support on `Pile`
- Enhanced `Event` with retryable flag, error accumulation, `assert_completed`
- 187 doc example tests verifying code snippets

### Fixed

- Lazy `asyncio.Lock` in `MCPConnectionPool`
- MCP env patterns, `ExceptionGroup` capture, thread-safety

## [0.19.2] - 2026-02-12

### Added

- `CLIEndpoint`, async context managers on `iModel`/`Branch`, error propagation

## [0.19.1] - 2026-02-12

### Added

- Native Gemini API integration via OpenAI-compatible endpoint

## [0.19.0] - 2026-02-11

### Added

- Native Gemini API integration (initial)

---

See git history for versions before v0.19.0.
