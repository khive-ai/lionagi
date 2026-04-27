# Provider Reference

## API providers

Pass `provider=` to `iModel()`, or let lionagi infer from the model name.

| Provider | `provider=` string | Key env var |
|----------|-------------------|------------|
| OpenAI | `"openai"` | `OPENAI_API_KEY` |
| Anthropic | `"anthropic"` | `ANTHROPIC_API_KEY` |
| Google Gemini | `"gemini"` | `GOOGLE_API_KEY` |
| Ollama (local) | `"ollama"` | — (no key needed) |
| NVIDIA NIM | `"nvidia"` | `NVIDIA_API_KEY` |
| Perplexity | `"perplexity"` | `PERPLEXITY_API_KEY` |
| Groq | `"groq"` | `GROQ_API_KEY` |
| OpenRouter | `"openrouter"` | `OPENROUTER_API_KEY` |
| DeepSeek | `"deepseek"` | `DEEPSEEK_API_KEY` |

DeepSeek's `reasoning_effort` field maps lionagi effort levels: `low`/`medium` → `"high"`,
`xhigh` → `"max"`. DeepSeek native values (`low`, `medium`, `high`, `max`) pass through unchanged.

```python
# Explicit provider
model = li.iModel(provider="anthropic", model="claude-opus-4-7-20251001")

# Default provider inferred (openai)
model = li.iModel(model="gpt-4o")

# Slash notation — provider inferred from prefix
model = li.iModel(model="anthropic/claude-opus-4-7")

# Custom base URL (proxy, local inference)
model = li.iModel(
    provider="openai",
    base_url="http://localhost:8080/v1",
    model="my-model",
)
```

## CLI endpoint providers

CLI endpoints spawn subprocess tools instead of calling REST APIs.
Pass `endpoint=` to select one; `is_cli` is set `True` automatically.
Use with `Branch.run()` or `Branch.operate()`.

| `endpoint=` | CLI tool spawned | `provider=` |
|-------------|-----------------|------------|
| `"claude_code"` | `claude` | `"claude_code"` |
| `"codex"` | `codex` | `"codex"` |
| `"gemini_code"` | `gemini` | `"gemini_code"` |
| `"pi"` / `"pi_code"` | `pi` | `"pi"` |

CLI endpoints authenticate via their own login, not via env vars:

- `claude_code`: `claude login` (Claude Max subscription) or `ANTHROPIC_API_KEY` (API key)
- `codex`: `codex login` (requires ChatGPT Plus/Pro — no API key accepted)
- `pi` / `pi_code`: subprocess-based; uses the local `pi` binary

```python
# Claude Code CLI endpoint
claude_model = li.iModel(provider="claude_code", model="sonnet")

# Codex CLI endpoint
codex_model = li.iModel(provider="codex", model="o3")

async for msg in branch.run("Refactor this function:", chat_model=claude_model):
    print(msg.content, end="", flush=True)
```

`Branch.operate()` detects `is_cli=True` and routes to `run_and_collect` instead of `communicate`.
See [operations.md](../api/operations.md#middle-protocol) for routing details.

## Tool / search providers

These providers integrate as callable tools via `branch.connect()` — not as chat models.

| Provider | Purpose | Key env var |
|----------|---------|------------|
| Exa | Neural search | `EXA_API_KEY` |
| Firecrawl | Web scraping | `FIRECRAWL_API_KEY` |
| Tavily | Research search | `TAVILY_API_KEY` |

```python
branch.connect(provider="exa", endpoint="search", name="search")
results = await branch.operate(
    instruction="Find recent papers on diffusion models",
    actions=True,
    tools=["search"],
)
```

## Default provider config

Set environment variables to avoid repeating `provider=` on every `iModel()` call:

```bash
export LIONAGI_CHAT_PROVIDER=openai
export LIONAGI_CHAT_MODEL=gpt-4.1-mini
```

Or configure per branch:

```python
branch = li.Branch(
    chat_model=li.iModel(provider="anthropic", model="claude-opus-4-7-20251001"),
    parse_model=li.iModel(model="gpt-4o-mini"),
)
```

Next: [Troubleshooting](troubleshooting.md)
