# Changelog

All notable changes to lionagi are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.22.0] - 2026-04-18

### Added
- `branch.run()` — async generator yielding typed `Message` objects from CLI endpoint streams
- `StreamChunk` — provider-agnostic streaming chunk type for normalizing CLI output
- CLI endpoints (`claude_code`, `codex`, `gemini`) `stream()` now yields `StreamChunk`
- Multi-model conversations — `chat_model` param on `branch.run()` switches providers mid-branch
- `.lionagi/agents/{name}.md` — agent profile convention (YAML frontmatter + system prompt)
- `li agent -a NAME` — load agent profile for system prompt, default model, effort, yolo
- `--cwd` and `--timeout` as shared CLI flags across all subcommands
- Unified Message class with `ClassVar` role, Exchange system, channel field
- `LionMessenger` tool + ReAct `between_rounds` hook for inter-branch messaging
- `Team` orchestrator pattern with persistence
- `li team` subcommand (create, list, show, send, receive)
- `li o fanout --team-mode` for persistent team-based fan-out

### Changed
- `li agent` uses `branch.run()` instead of `branch.communicate()`
- `agent.py` refactored to use `add_common_cli_args()` (no duplicated flags)
- CLI guide updated with agent profiles, `branch.run()` programmatic usage, message types

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
