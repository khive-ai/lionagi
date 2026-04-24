
# Changelog

All notable changes to lionagi are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.22.9] - 2026-04-24

### Security

- **Symlink containment on `li skill` + `li play`** — reject paths whose resolved target escapes the resolved skills/playbooks root. Root symlinks (users pointing `~/.lionagi/skills/` at any directory they manage) still accepted; hostile per-entry symlinks blocked. Addresses PR #930 review finding.
- **`FlowAgent.id` / `FlowOp.id` validation** — `^[A-Za-z0-9_-]{1,64}$` enforced via Pydantic `field_validator` to block model-controlled strings from becoming filesystem path escapes via `RunDir.agent_artifact_dir()`. Defense-in-depth re-check inside `RunDir`.
- **Duplicate `FlowAgent.id` / `FlowOp.id` rejected at plan validation** (were silently overwriting).
- **`--max-ops` enforced cumulatively across re-plan rounds** (previously only initial plan — control op could bypass cap across rounds).
- **Save-path containment** — resolved `--save DIR` must live under `cwd` or `$HOME`.
- **Pin `lxml>=6.1.0`** (CVE-2026-41066 — XXE via default `iterparse` / `ETCompatXMLParser`) and **`python-dotenv>=1.2.2`** (CVE-2026-28684 — symlink follow on `set_key` cross-device rename).

### Added

- **`li skill NAME`** — CC-compatible skill reader. Reads `~/.lionagi/skills/<NAME>/SKILL.md`, strips YAML frontmatter, prints body to stdout. Agents can shell out mid-run to fetch reference content. Also `li skill list` and `li skill show NAME`.
- **`li play NAME [args]`** — sugar for `li o flow -p NAME [args]`. Plus `li play list` to enumerate installed playbooks.
- **`li o flow -p NAME`** — load a playbook from `~/.lionagi/playbooks/<NAME>.playbook.yaml`. Declared args are injected into argparse so flag values aren't eaten by positional arguments.
- **Playbook args schema** — playbook YAML may declare `args:` (typed schema with `type`, `default`, `help`) or a CC-compatible `argument-hint:` string (e.g. `'[--tabs N] [--poll]'`) for fallback parsing. Explicit schema wins when both are present.
- **Template interpolation** — `prompt:` fields in flow specs / playbooks now substitute `{input}` (positional CLI prompt) and `{arg_name}` (declared args). If no placeholders are present, positional text is appended with a blank line (CC slash-command style).
- **Agent directory layout** — `.lionagi/agents/<name>/<name>.md` (directory form) resolved before flat `.lionagi/agents/<name>.md` (legacy form). Supplementary references under `<name>/patterns/`, `<name>/refs/` can be read on demand.
- **`examples/`** directory — ready-to-install templates under `examples/{agents,skills,playbooks}/` with a README explaining when to use each primitive.
- **`--team-attach NAME`** on `li o flow` — upsert semantics: attach to an existing team by name (preserving message history) or create if missing. First use never requires a manual `li team create`. Mutually exclusive with `--team-mode` (which keeps "always fresh" semantics). Also supported as `team_attach:` in playbook YAML.

### Changed

- **`add_orchestrate_subparser`** now returns `{"fanout": fo, "flow": fl}` so callers can post-hoc extend the flow sub-parser with playbook-declared flags. Non-breaking for existing callers that ignore the return value.
- **`li agent -a NAME` / `li o flow -a NAME`** resolve `<name>/<name>.md` first, then fall back to the flat `<name>.md`. Existing flat profiles continue to work.
- **`--max-agents` renamed to `--max-ops`** — the flag caps DAG operation count, not agent count. `--max-agents` kept as a deprecated alias; `max_agents:` spec field also still accepted. Planner budget guidance now says "ops (DAG nodes)" instead of "agents", matching enforcement. When truncation happens, a warning now names the number of dropped ops instead of silently slicing. Fixes a footgun where `--max-agents 10` silently dropped the terminal critic op from an 11-op plan.
- **Spec validation for `effort`** now accepts all values in `cli/_providers.EFFORT_LEVELS` (`none | minimal | low | medium | high | xhigh | max`) — previously only `low | medium | high | xhigh` passed, rejecting playbook values the CLI itself accepts.
- **Spec validation for `with_synthesis`** now accepts both `bool` and `str` (the latter being a model spec), matching `--with-synthesis [MODEL]` CLI surface. Previously only bool passed.

### Fixed

- **Path-containment hardening (security)**: `FlowAgent.id` and `FlowOp.id` are now validated against `^[A-Za-z0-9_-]{1,64}$` via Pydantic `field_validator`. A model-produced id like `/etc/passwd` or `../../tmp/evil` previously became an artifact directory path and could escape the run artifact root. `RunDir.agent_artifact_dir()` adds defense-in-depth: path-separator / dot checks plus a `relative_to(artifact_root)` containment assertion.
- **Playbook args collision leaked base-flag values into templates**: when a playbook declared an arg whose name collided with a built-in CLI flag (e.g. `args.save`), parser injection correctly skipped adding the flag, but runtime interpolation re-derived the schema and read the base parser's `args.save` value into `{save}` placeholders. The collision-filtered schema is now stashed on the parser via `set_defaults(_playbook_args_schema=...)` and reused during interpolation.

## [0.22.6] - 2026-04-20

### Added

- **Two-level flow DAG**: `FlowAgent` (persistent Branch identity with memory) + `FlowOp` (DAG node); multiple ops share an `agent_id` so the branch accumulates message history across turns.
- **Run-scoped persistence**: branch state at `~/.lionagi/runs/{run_id}/`; artifacts at `--save <dir>` or `state_root/artifacts/`; `run.json` manifest records command, branches, agents, and ops.
- **`Middle` protocol** (`lionagi.operations.types`): formalizes the middle-stage callable contract for `operate()`; both `communicate` and `run_and_collect` satisfy it structurally.
- **Named CLI logging channels** (`lionagi/cli/_logging.py`): `progress()`, `hint()`, `warn()`, `log_error()` replace scattered `print(..., file=sys.stderr)` calls.
- **Team file locking** (POSIX `fcntl.flock`): `_locked_team` serializes concurrent `li team send` / `li team receive` calls from parallel workers.
- **`--from-op` on `li team send`**: ties a coordination signal to a specific op invocation so consumers can distinguish which turn emitted it.
- **Timestamped read receipts**: team message `read_by` is now `dict[name → ISO timestamp]`; old list format auto-normalized on read.
- **`python -m lionagi.cli`** entry point via `lionagi/cli/__main__.py`; required by `li o flow --background` subprocess respawn.
- **Per-agent artifact directories**: `{artifact_root}/{agent_id}/` — all ops on the same agent share one directory; cross-agent deps read via `{artifact_root}/{dep_agent_id}/*`.
- **`LIONAGI_RUN_ID` env var**: subprocesses inherit the parent run ID so `--background` respawns attach to the same run manifest.

### Changed

- **`branch.operate()` absorbs `branch.instruct()`** — accepts `stream_persist`, `persist_dir`, `middle` kwargs and auto-dispatches to `communicate` (API) or `run_and_collect` (CLI) based on `chat_model.is_cli`.
- **`run_and_collect()`** added to `lionagi.operations.run.run` — streams via `run()`, accumulates assistant text, optionally parses into a Pydantic model.
- **`TEAM_COORD_SECTION`** appended to the base worker prompt (was replacing it); `TEAM_WORKER_SYSTEM` preserved as a composed alias for backward compatibility.
- **DAG operation name** in `li o flow` / `li o fanout` manifests changed from `"instruct"` to `"operate"`.
- **Branch storage**: `logs/agents/{provider}/{branch_id}.json` → `runs/{run_id}/branches/{branch_id}.json`; `find_branch` falls back to the legacy layout for pre-0.22.6 branches.
- **`last_branch.json`** schema: `{provider, branch_id}` → `{run_id, branch_id}`; legacy format auto-detected on read.

### Removed

- **`branch.instruct()`** method — migrate to `branch.operate(instruction=..., stream_persist=..., persist_dir=...)`.
- **`lionagi.operations.instruct`** module — import from `lionagi.operations.operate` instead.
- **Flat artifact dumps** at `{save_dir}/{id}_{name}.md` — superseded by per-agent artifact directories.
- **Hardcoded `"codex/gpt-5.4"` fallback** in profile resolution — set `--model` or `LIONAGI_CHAT_MODEL` explicitly.

### Fixed

- **`operate()` crash on raw-text CLI output**: `str.get()` TypeError when `response_format` is absent — now checks `isinstance(result, dict)` before field extraction.
- **Flow DAG schema conflation**: `FlowAgentSpec` was simultaneously branch identity, DAG node, and instruction carrier — multi-op-per-branch was impossible; redesigned as separate `FlowAgent` + `FlowOp`.
- **Team concurrent-write race**: parallel `li team send` calls no longer clobber each other under POSIX exclusive lock.

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
