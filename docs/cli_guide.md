# LionAGI CLI Guide

The `li` command-line tool lets you spawn agents and orchestrate multi-agent workflows directly from your terminal.

## Prerequisites

You must authenticate with each provider **before** using it through `li`:

```bash
# Claude Code — install and login
npm install -g @anthropic-ai/claude-code
claude  # opens interactive session, completes login

# OpenAI Codex — install and login
npm install -g @openai/codex
codex  # opens interactive session, completes login
```

Once logged in, `li` can spawn agents through these providers without additional setup.

> **Gemini CLI** (`gemini-code`) is currently unstable due to MCP configuration conflicts and is not recommended for use.

## Installation

```bash
uv add lionagi   # or pip install lionagi
```

The `li` command is automatically available after installation.

## Commands

### `li agent` — Single Agent

Spawn a one-shot agent and get a response.

```bash
li agent MODEL PROMPT [OPTIONS]
```

**Model spec format**: `provider/model-effort`

| Example | Provider | Model | Effort |
|---------|----------|-------|--------|
| `claude/sonnet` | claude_code | sonnet | (default) |
| `claude/opus-4-7-high` | claude_code | claude-opus-4-7 | high |
| `codex/gpt-5.3-codex-spark` | codex | gpt-5.3-codex-spark | (default) |
| `codex/gpt-5.4-xhigh` | codex | gpt-5.4 | xhigh |

Aliases: `claude` = `claude/sonnet`, `codex` = `codex/gpt-5.3-codex-spark`

**Options**:

| Flag | Description |
|------|-------------|
| `-v, --verbose` | Stream real-time output (thinking, tool use, text) |
| `--yolo` | Auto-approve all tool calls |
| `--effort LEVEL` | Override effort (claude: low/medium/high/xhigh/max, codex: none/minimal/low/medium/high/xhigh) |
| `--theme light\|dark` | Terminal display theme |
| `-r BRANCH_ID` | Resume a previous conversation |
| `-c` | Continue the most recent conversation |

**Examples**:

```bash
# Basic
li agent claude/sonnet "Explain the observer pattern in 3 sentences"

# With verbose streaming
li agent claude/opus-4-7-high "Review this code for security issues" -v

# Auto-approve tool calls (claude code will read/write files)
li agent claude/sonnet "Fix the bug in main.py" --yolo

# Using codex
li agent codex/gpt-5.4-xhigh "Audit this function for edge cases"
```

### `li o fanout` — Multi-Agent Fan-Out

Three-phase orchestration: decompose, fan out to workers, synthesize.

```bash
li o fanout MODEL PROMPT [OPTIONS]
```

The orchestrator (MODEL) decomposes your prompt into N agent requests using structured output, fans them out to workers in parallel, and optionally synthesizes the results.

**Options**:

| Flag | Description |
|------|-------------|
| `-n, --num-workers N` | Number of workers (default: 3) |
| `--workers M1,M2,...` | Explicit worker model list (overrides -n) |
| `--with-synthesis [MODEL]` | Enable synthesis. Bare flag uses orchestrator model; with MODEL uses that instead |
| `--synthesis-prompt TEXT` | Custom synthesis instruction |
| `--max-concurrent N` | Limit concurrent workers |
| `--output text\|json` | Output format |
| `--save DIR` | Save outputs to directory |

Plus all shared flags: `-v`, `--yolo`, `--effort`, `--theme`

**Examples**:

```bash
# Pattern A: Homogeneous workers, no synthesis
li o fanout claude/sonnet "What are the key design patterns here?" -n 3

# Pattern B: Heterogeneous workers + synthesis
li o fanout claude/sonnet "Analyze the error handling in this codebase" \
    --workers "claude/sonnet, codex/gpt-5.3-codex-spark" \
    --with-synthesis

# Pattern C: Cheap workers, strong synthesizer
li o fanout claude/sonnet "Compare async patterns in Python vs Rust" \
    -n 3 --with-synthesis claude/opus-4-7-high

# Save outputs + JSON format
li o fanout claude/sonnet "Review authentication flow" \
    -n 2 --with-synthesis --save ./review-output --output json
```

### Resuming Conversations

Every agent and fanout session persists its conversation state. After any command, you'll see:

```
[to resume] li agent -r 3560c288-ff93-4713-9bce-23880cbd03df "..."
```

For fanout, each branch is resumable:

```
[orchestrator] li agent -r adf15442-8d0c-4499-a88b-16d8d1adaeeb "..."
[worker-1]      li agent -r 1d63e2bd-5f86-4532-8c5e-1d2bf2340852 "..."
[worker-2]      li agent -r 95d31076-079e-48af-94b3-86ebdf417260 "..."
```

Resume continues the conversation with full context:

```bash
li agent -r adf15442-8d0c-4499-a88b-16d8d1adaeeb "expand on point 2"
```

Sessions are stored at `~/.lionagi/logs/agents/{provider}/{branch-id}`.

## Provider Support

| Provider | Status | Auth |
|----------|--------|------|
| **Claude Code** (`claude/...`) | Stable | `npm i -g @anthropic-ai/claude-code && claude` |
| **Codex** (`codex/...`) | Stable | `npm i -g @openai/codex && codex` |
| **Gemini** (`gemini-code/...`) | Unstable | Not recommended |

## Effort Levels

Effort controls reasoning depth. Embedded in model spec or via `--effort` flag:

```bash
# In spec (parsed automatically)
li agent claude/opus-4-7-high "..."

# Via flag (overrides spec)
li agent claude/opus-4-7 "..." --effort xhigh
```

| Provider | Levels |
|----------|--------|
| Claude | low, medium, high, xhigh, max |
| Codex | none, minimal, low, medium, high, xhigh |
| Gemini | Not supported (errors if specified) |
