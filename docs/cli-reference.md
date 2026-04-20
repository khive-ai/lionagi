# CLI Reference

```bash
li agent MODEL PROMPT [flags]        # single-turn agent
li team SUBCMD [flags]               # persistent team messaging
li o fanout MODEL PROMPT [flags]     # parallel workers
li o flow   MODEL PROMPT [flags]     # auto-DAG pipeline
```

---

## Common flags

Available on `li agent`, `li o fanout`, `li o flow`. Source: `cli/_providers.py:263`

| Flag | Default | Notes |
|------|---------|-------|
| `--yolo` | false | Auto-approve all tool calls |
| `-v, --verbose` | false | Stream real-time output; suppresses final print |
| `--theme {light,dark}` | none | Terminal theme |
| `--effort LEVEL` | none | Override effort; claude: `low medium high xhigh max`; codex: `none minimal low medium high xhigh`; gemini: unsupported (`cli/_providers.py:24,44`) |
| `--cwd DIR` | none | Working directory for CLI endpoint |
| `--timeout SECONDS` | none | Hard timeout; partial branches saved |

**Model spec**: `provider/model[-effort]` — e.g. `claude/opus-4-7-high`, `codex/gpt-5.4-xhigh`. Bare aliases: `claude` → `claude_code/sonnet`, `codex` → `codex/gpt-5.3-codex-spark`, `gemini-code` → `gemini_code/gemini-3.1-flash-lite-preview`. Source: `cli/_providers.py:72,145`

---

## `li agent`

One-shot agent turn or resumed conversation.

```bash
li agent [model] prompt [flags]
```

| Arg/Flag | Default | Notes |
|----------|---------|-------|
| `model` | — | Spec or alias. Omit with `-r` or `-c`. `cli/agent.py:156` |
| `prompt` | — | Message to send. `cli/agent.py:165` |
| `-a, --agent NAME` | none | Profile from `.lionagi/agents/<NAME>.md`; sets model/effort/system/yolo. `cli/agent.py:167` |
| `-r, --resume BRANCH_ID` | none | Resume prior branch. `cli/agent.py:178` |
| `-c, --continue-last` | false | Resume most recent branch. `cli/agent.py:184` |

`-r` and `-c` are mutually exclusive (`cli/agent.py:49`). Common flags apply.

```bash
li agent claude/sonnet "What does Branch.operate() do?"
```

```text
# output:
Branch.operate() is the universal structured operation entry point...

[to resume] li agent -r 20260420T110143-a1b2c3 "..."
```

Python equivalent: `branch.operate(instruction="...")` → [`api/branch.md#operate`](api/branch.md#operate)

---

## `li team`

Persistent inbox messaging. Teams stored at `~/.lionagi/teams/{team_id}.json` under `fcntl.flock` (`cli/team.py:50`).

```bash
li team create NAME -m MEMBERS
li team list     [alias: ls]
li team show TEAM
li team send CONTENT -t TEAM --to RECIPIENTS [--from NAME] [--from-op OP]
li team receive  -t TEAM [--as MEMBER]   [alias: recv]
```

### `li team create`

| Arg/Flag | Required | Notes |
|----------|----------|-------|
| `name` | yes | Team name |
| `-m, --members` | yes | Comma-separated member names |

Source: `cli/team.py:284`

```bash
li team create "docs-team" -m "researcher,writer,reviewer"
```

```text
# output:
Created team 'docs-team' (7fa0d9abbf5b)
  Members: researcher, writer, reviewer
  File: ~/.lionagi/teams/7fa0d9abbf5b.json
```

**list** — sorted by mtime; shows ID, name, members, msg count (`cli/team.py:294`). **show TEAM** — all messages with timestamps and `read_by` (`cli/team.py:297`). `TEAM` = ID, prefix, or name.

### `li team send`

| Arg/Flag | Required | Default | Notes |
|----------|----------|---------|-------|
| `content` | yes | — | Message text (positional) |
| `--team, -t` | yes | — | Team ID or name |
| `--to` | yes | — | `all` or comma-separated names |
| `--from` | no | `_cli` | Sender name |
| `--from-op` | no | none | Op id; ties signal to a specific flow invocation |

Source: `cli/team.py:301`

```bash
li team send "Research done — see research.md" \
  --team 7fa0d9abbf5b --to writer --from researcher --from-op o1
```

### `li team receive`

| Flag | Required | Default | Notes |
|------|----------|---------|-------|
| `--team, -t` | yes | — | Team ID or name |
| `--as` | no | none | Mark as read for this member; omit = see all |

Source: `cli/team.py:322`

```bash
li team receive --team 7fa0d9abbf5b --as writer
```

Python equivalent: `session.send()` / `session.receive()` → [`api/team.md`](api/team.md)

---

## `li o fanout`

Three-phase: orchestrator decomposes → N workers in parallel → optional synthesis.

```bash
li o fanout [model] prompt [flags]
```

| Flag | Default | Notes |
|------|---------|-------|
| `-a, --agent NAME` | none | Orchestrator profile. `cli/orchestrate/__init__.py:49` |
| `-n, --num-workers N` | 3 | Worker count; ignored when `--workers` set |
| `--workers M1,M2,...` | none | Per-worker model specs (each can include effort suffix) |
| `--max-concurrent N` | 0 | Max concurrent (0 = all) |
| `--with-synthesis [MODEL]` | false | Enable synthesis; bare = orchestrator model |
| `--synthesis-prompt TEXT` | none | Override synthesis instruction |
| `--output {text,json}` | text | Output format |
| `--save DIR` | none | Write artifacts here |
| `--team-mode [NAME]` | none | Create persistent team; bare = `"fanout"` |

Source: `cli/orchestrate/__init__.py:29–119`. Common flags apply.

```bash
li o fanout claude/opus-high "Audit lionagi/session/ for stale API surface" \
  -n 3 --with-synthesis --save ./audit-out
```

```text
# output:
Phase 1: Orchestrator decomposing task into 3 agent requests...
Phase 1 done (3.2s): 3 requests generated.
Phase 2: Fanning out to 3 workers: [claude/opus, claude/opus, claude/opus]
Phase 2 done (14.1s).
Saved 3 worker results to /Users/ocean/audit-out
Phase 3: Synthesis [claude/opus]...
Saved to /Users/ocean/audit-out
```

Worker outputs: `worker_1.md … worker_N.md` in artifact root (`fanout.py:269`). Synthesis: `synthesis.md` (`fanout.py:317`). Resume cancelled workers with `li agent -r BRANCH_ID`.

---

## `li o flow`

Auto-DAG pipeline. Orchestrator plans a `FlowPlan` (agents + ops with `depends_on` edges); engine executes with dependency-aware parallelism. Control ops trigger re-planning up to 3 rounds (`flow.py:705`).

```bash
li o flow [model] prompt [flags]
```

| Flag | Default | Notes |
|------|---------|-------|
| `-a, --agent NAME` | none | Orchestrator profile |
| `--with-synthesis [MODEL]` | false | Final synthesis after all ops |
| `--max-concurrent N` | 0 | Max concurrent agents per phase (0 = all) |
| `--max-agents N` | 0 | Cap total ops (0 = unlimited) |
| `--dry-run` | false | Plan DAG and print; no execution |
| `--show-graph` | false | Render DAG as matplotlib PNG into `--save` dir |
| `--bare` | false | Ignore agent profiles; all workers use CLI model |
| `--background` | false | Subprocess run; requires `--save`; monitor `tail -f <save>/flow.log`; child inherits `LIONAGI_RUN_ID` (`cli/_runs.py:57`) |
| `--output {text,json}` | text | Output format |
| `--save DIR` | none | Artifact dir; required for `--background` |
| `--team-mode [NAME]` | none | Create persistent team; bare = `"flow"` |

Source: `cli/orchestrate/__init__.py:122–209`. `--background` re-invokes `python -m lionagi.cli` without itself (`cli/orchestrate/__init__.py:265`). Common flags apply.

```bash
li o flow claude/opus "Write and test a CLI arg parser for a new subcommand" \
  --save ./parser-work --with-synthesis
```

```text
# output:
Planning DAG...
Plan done (4.1s): 3 agents, 4 ops — o1:r1 | o2:i1←o1 | o3:t1←o2 | o4:r1←o3
Executing DAG: 3 agents / 4 ops...
  ▶ researcher started
  ✓ researcher done (8.2s)
  ▶ implementer started
  ✓ implementer done (22.1s)
  ▶ tester started
  ✓ tester done (18.4s)
Synthesis [claude/opus]...
Saved to ./parser-work/
Total: 55.8s
```

Use `--dry-run` to inspect the plan before running. Artifact dirs per agent: `<save>/{agent_id}/`. Python equivalent: `Builder` + `Session.flow()` → [`api/flow.md`](api/flow.md)

---

## Run-ID and persistence

Every CLI invocation allocates a run directory. Source: `cli/_runs.py:14`. Run ID format: `YYYYMMDDTHHMMSS-{6hex}` (`cli/_runs.py:61`).

```text
~/.lionagi/runs/{run_id}/
  run.json                        manifest (command, branches, artifact_root)
  branches/{branch_id}.json       branch snapshot — resumable via -r / -c
  stream/{branch_id}.buffer.jsonl live chunk buffer during streaming
```

Resume any prior branch:

```bash
li agent -r 20260420T110143-a1b2c3 "follow up"
li agent -c "continue most recent"
```

### Env Vars

| Variable | Purpose | Source |
|----------|---------|--------|
| `LIONAGI_RUN_ID` | Child inherits parent run ID (background flows) | `cli/_runs.py:57` |
| `LIONAGI_HOME` | Override `~/.lionagi/` base dir | `lionagi/utils.py` |
| `OPENAI_API_KEY` | OpenAI REST API key (for `iModel`, not for `codex` CLI alias) | `lionagi/config.py` |
| `ANTHROPIC_API_KEY` | Anthropic REST API key (for `iModel`; `claude` alias uses `claude login` instead) | `lionagi/config.py` |
| `GOOGLE_API_KEY` | Gemini key | `lionagi/config.py` |
| `GROQ_API_KEY` | Groq key | `lionagi/config.py` |

---

*Sources: `cli/agent.py` · `cli/team.py` · `cli/orchestrate/__init__.py` · `cli/orchestrate/fanout.py` · `cli/orchestrate/flow.py` · `cli/_providers.py` · `cli/_runs.py`*

Next: [Python API reference](api/index.md)
