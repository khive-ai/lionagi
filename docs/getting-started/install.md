# Install

```bash
pip install lionagi   # or: uv add lionagi
```

## Authenticate your CLI providers

lionagi's CLI aliases (`claude`, `codex`) spawn subprocess tools, not REST API calls.
Each requires its own login step.

### `claude` (Claude Code CLI)

Option A — subscription login (recommended if you have Claude Max):

```bash
npm install -g @anthropic-ai/claude-code
claude login
```

Option B — API key (works without a subscription):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### `codex` (OpenAI Codex CLI)

Requires ChatGPT Plus or Pro subscription. No API key — uses CLI session:

```bash
npm install -g @openai/codex
codex login
```

### OpenAI API models (`gpt-4.1-mini`, etc.)

Used by `Branch(chat_model=iModel(model="gpt-4o-mini"))` in Python, not by `li agent`:

```bash
export OPENAI_API_KEY="sk-..."
```

Add any export to `~/.zshrc` or `~/.bashrc` to persist across shells.

## Verify

```bash
li --help
```

```text
# output:
usage: li [-h] {orchestrate,o,agent,team} ...

lionagi command line — spawn subagents via any CLI-backed provider.

positional arguments:
  {orchestrate,o,agent,team}
    orchestrate (o)     Multi-agent orchestration patterns.
    agent               Spawn one-shot subagent (blocking); prints final response.
    team                Team messaging — send/receive between named agents.
```

## Optional extras

| Extra | Installs | Command |
|-------|----------|---------|
| `reader` | PDF/HTML document parsing | `uv add "lionagi[reader]"` |
| `ollama` | Local model support via Ollama | `uv add "lionagi[ollama]"` |
| `rich` | Richer terminal output | `uv add "lionagi[rich]"` |

Next: [Your first flow](first-flow.md)
