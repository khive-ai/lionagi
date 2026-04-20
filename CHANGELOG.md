# Changelog

All notable changes to lionagi are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed
- `branch.operate()` absorbs `branch.instruct()` — the only real difference was the middle function (`communicate` for API, `run`-and-collect for CLI). `operate` now accepts a `middle` callable and auto-selects based on `chat_model.is_cli`. Added `stream_persist` and `persist_dir` kwargs for CLI streaming.
- `run_and_collect()` added to `lionagi.operations.run.run` — stream via `run()`, accumulate assistant text, optionally parse. Drop-in `communicate`-compatible middle for CLI endpoints.
- Orchestrator DAG nodes (`li o flow`, `li o fanout`) now add `"operate"` operations; prior `"instruct"` name removed.

### Removed
- `branch.instruct()` method and `lionagi.operations.instruct` module. Callers migrate to `branch.operate(instruction=..., stream_persist=..., persist_dir=...)`.

## [0.22.5] - 2026-04-19

### Fixed
- Unified all orchestration operations to `instruct` — flow/fanout orchestrator, workers, and synthesis nodes were mixing `operate`, `communicate`, and `instruct` with incompatible parameter sets, causing `Invalid parameter` crashes on CLI endpoints
- Fixed `_prepare_run_kwargs` return type annotation (`tuple[Instruction, dict]`)

## [0.22.4] - 2026-04-19

### Fixed
- CLI endpoints crash with `Invalid parameter: stream_persist` when flow/fanout passes ChatParam fields (guidance, context, response_format, etc.) through to `CodexCodeRequest` / `ClaudeCodeRequest` — now filters `req_dict` to only keys in the request model's `model_fields`

## [0.22.3] - 2026-04-19

### Added
- Stream persist — three-phase write-ahead logging for `branch.run()` / `branch.instruct()`:
  - Init: branch snapshot created at start
  - Stream: JSONL buffer appended per chunk via `imodel.streaming_process_func`
  - Consolidate: final `branch.to_dict()` written in `try/finally`, buffer removed
- `RunParam` frozen dataclass for clean parameter passing through the run pipeline
- `imodel.provider_session_id` for tracking CLI provider sessions
- `persist_session_branches()` shared helper for orchestration session persistence
- `LIONAGI_HOME` centralized in `utils.py` (single source of truth for `~/.lionagi`)
- `Instruct.handle()` for structured output result handling

### Changed
- `branch.instruct()` is now the universal operation — `li agent` uses it instead of manual `run()` + file write
- `_prepare_run_kwargs` extracted from `chat.py` into `chat/_prepare.py` (shared by chat and run)
- Orchestration `fanout` and `flow` use `persist_session_branches()` instead of inline persistence
- `run()` integrates `stream_persist` and `persist_dir` via `RunParam`

### Fixed
- Double `create_payload` bug in all 3 CLI endpoints — `APICalling._core_stream` was re-calling `endpoint.create_payload` on an already-built payload (claude, codex, gemini)
- `fanout` timeout recovery now catches `TimeoutError` properly

## [0.22.2] - 2026-04-19

### Added
- `li o flow` — DAG-based multi-agent orchestration with dependency edges
  - Agents declare `depends_on` for true DAG execution (not just phases)
  - Control nodes (`control=true`) produce structured `FlowControlVerdict` for critic-driven iteration loops (up to 3 rounds)
  - `--bare` flag: all workers use CLI model, ignore agent profiles
  - `--max-agents N`: cap total agents the orchestrator may plan
  - `--dry-run`: plan the DAG without executing — shows agents, deps, model resolution
- `branch.instruct()` routes automatically based on endpoint type:
  - CLI endpoints → `run()` stream + `parse()` extraction
  - API endpoints → `operate()` with structured output
- Orchestrator guidance now includes per-role model and effort info for informed routing
- `reason=True` on `instruct()` now works without `field_models` (was silently dropped)

### Fixed
- Flow workers pass `guidance` as proper kwarg (was buried in context dict)
- Default model fallback when role has no agent profile (was crashing with "Provider must be provided")
- Default system prompt for workers without profiles (was `None`)
- Orchestrator prompts hardened against provider-native subagent spawning

### Changed
- `orchestrate.py` (1610 lines) split into `orchestrate/` package: `_common.py`, `fanout.py`, `flow.py`, `__init__.py`
- Flow plan model redesigned: flat agent list with `depends_on` edges replaces phase-based pipeline
- All flow operations use `instruct` (unified routing) instead of `communicate`

### Security
- authlib bumped to >=1.6.11 (CSRF when using cache)
- python-multipart bumped to >=0.0.26 (DoS via large preamble/epilogue)
- pytest bumped to >=9.0.3 (vulnerable tmpdir handling)
- pillow bumped to >=12.2.0 (FITS GZIP decompression bomb)

## [0.22.1] - 2026-04-18

### Fixed
- CLI orchestration: guidance coercion, worker tool access, output dedup (#902)
- `chat`: improved guidance handling in chat function (#903)

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
