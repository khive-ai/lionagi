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
| `-a, --agent NAME` | Load agent profile from `.lionagi/agents/<NAME>.md` |
| `-r BRANCH_ID` | Resume a previous conversation |
| `-c` | Continue the most recent conversation |
| `-v, --verbose` | Stream real-time output (thinking, tool use, text) |
| `--yolo` | Auto-approve all tool calls |
| `--effort LEVEL` | Override effort (claude: low/medium/high/xhigh/max, codex: none/minimal/low/medium/high/xhigh) |
| `--theme light\|dark` | Terminal display theme |
| `--cwd DIR` | Working directory for the agent |
| `--timeout SECONDS` | Timeout for the agent run |

**Examples**:

```bash
# Basic
li agent claude/sonnet "Explain the observer pattern in 3 sentences"

# With agent profile (model + system prompt from profile)
li agent -a implementer "Fix the bug in main.py"

# Profile + model override (CLI wins over profile default)
li agent claude/opus -a reviewer "Review this PR"

# With verbose streaming
li agent claude/opus-4-7-high "Review this code for security issues" -v

# Auto-approve tool calls
li agent claude/sonnet "Fix the bug in main.py" --yolo

# Working directory + timeout
li agent -a implementer --cwd /path/to/repo --timeout 300 "Add tests"
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

Plus all shared flags: `-v`, `--yolo`, `--effort`, `--theme`, `--cwd`, `--timeout`

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

### `li team` — Team Messaging

Persistent inbox-style messaging for coordinating agent teams.

```bash
li team COMMAND [OPTIONS]
```

**Commands**:

| Command | Description |
|---------|-------------|
| `create NAME -m "A,B,C"` | Create a named team with members |
| `list` | List all teams |
| `show TEAM_ID` | Show team details and all messages |
| `send CONTENT -t ID --to RECIPIENTS` | Send message (`all` or comma-separated names) |
| `receive -t ID --as NAME` | Read inbox (marks messages as read) |

**Examples**:

```bash
# Create a team
li team create "research" -m "analyst,writer,reviewer"

# Send to everyone
li team send "analyze the auth module" -t abc123 --to all

# Send to specific member
li team send "focus on JWT handling" -t abc123 --to writer --from analyst

# Check inbox
li team receive -t abc123 --as writer
```

Teams persist at `~/.lionagi/teams/{id}.json`. Messages track read state per member.

### `--team-mode` with Fan-Out

Add `--team-mode` to any fanout to auto-create a team. Workers get named identities and team context in their system prompt. Results are posted as team messages after completion.

```bash
# Workers get team context, results tracked as messages
li o fanout claude/sonnet "Improve test coverage" \
    -n 5 --yolo --team-mode "coverage-boost" --with-synthesis

# After completion, follow up with any worker
li team receive -t <team-id> --as orchestrator
li agent -r <worker-branch-id> "dig deeper on the auth module"
li team send "found 3 more edge cases" -t <team-id> --to all --from worker-1
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

## Agent Profiles (`.lionagi/agents/`)

Agent profiles define reusable agent configurations — system prompt, default model, effort level, and permissions. Profiles live in your repository's `.lionagi/agents/` directory.

### Profile format

```markdown
---
model: claude_code/sonnet
effort: high
yolo: true
---

You are an implementer agent. Write production code — no stubs, no
placeholders. Read existing code before writing. Match the codebase's
style and conventions. Write tests alongside implementation.
```

**Frontmatter fields** (all optional, CLI flags override):

| Field | Description |
|-------|-------------|
| `model` | Default model spec (e.g. `claude_code/opus`, `codex/gpt-5.4`) |
| `effort` | Default reasoning effort level |
| `yolo` | Auto-approve tool calls by default |

The markdown body becomes the system prompt injected into the conversation.

### Profile discovery

`li` walks up from the current directory to the git root looking for `.lionagi/`. This means profiles are project-scoped — different repos can have different agent configurations.

### Using profiles

```bash
# Use profile (model from frontmatter, system prompt injected)
li agent -a implementer "fix the auth bug"

# Profile + explicit model (CLI overrides profile default)
li agent claude/opus -a reviewer "review the PR"

# Profile with common flags
li agent -a implementer --cwd ./myrepo --timeout 300 "add integration tests"
```

### Creating profiles

```bash
mkdir -p .lionagi/agents

cat > .lionagi/agents/implementer.md << 'EOF'
---
model: claude_code/sonnet
effort: high
---

You are an implementer. Write production code, not stubs...
EOF
```

## Programmatic Usage (`branch.run()`)

The CLI is built on `branch.run()`, a streaming async generator that yields typed `Message` objects. You can use it directly in Python for more control.

```python
from lionagi import Branch
from lionagi.service.imodel import iModel

sonnet = iModel(model="claude_code/sonnet")
branch = Branch()

# Stream message objects
async for msg in branch.run("analyze this code", chat_model=sonnet):
    match type(msg).__name__:
        case "Instruction":    print(f"You: {msg.content.instruction}")
        case "AssistantResponse": print(f"AI: {msg.response}")
        case "ActionRequest":  print(f"Tool: {msg.content.function}")
        case "ActionResponse": print(f"Result: {msg.content.output[:100]}")
```

### Multi-model conversations

Switch providers mid-conversation on the same branch — context carries forward:

```python
sonnet = iModel(model="claude_code/sonnet")
gpt    = iModel(model="openai/gpt-4.1-mini")

branch = Branch()

# Step 1: Claude analyzes (CLI endpoint, streaming)
async for msg in branch.run("What is the capital of Japan?", chat_model=sonnet):
    pass

# Step 2: GPT extracts structured data (API endpoint)
from pydantic import BaseModel, Field

class CityInfo(BaseModel):
    city: str = Field(description="City name")
    country: str = Field(description="Country")
    population_millions: float = Field(description="Population in millions")

result = await branch.operate(
    "Extract the city info from our conversation",
    chat_model=gpt,
    response_format=CityInfo,
)
# CityInfo(city='Tokyo', country='Japan', population_millions=14.0)
```

### Message types

`branch.run()` yields these message types:

| Type | When | Content |
|------|------|---------|
| `Instruction` | Start of each turn | User's prompt |
| `AssistantResponse` | Model's reply | `.response` for text, `.metadata["thinking"]` for reasoning |
| `ActionRequest` | Model calls a tool | `.content.function`, `.content.arguments` |
| `ActionResponse` | Tool returns result | `.content.output` |

Thinking traces are folded into `AssistantResponse.metadata["thinking"]` — not yielded as separate messages.

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
