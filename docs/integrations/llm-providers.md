# LLM Provider Integration

LionAGI supports 12 providers through a unified `iModel` interface. All providers
work with the same `Branch` API -- swap the model and your application code stays
the same.

## How Provider Selection Works

Every LLM call in LionAGI goes through `iModel`, which wraps an `Endpoint` for a
specific provider. When you create an `iModel`, the `match_endpoint` function
selects the right endpoint class based on the `provider` and `endpoint` strings
you pass in. API keys are resolved automatically from environment variables (or
`.env` / `.env.local` / `.secrets.env` files) via pydantic-settings.

```python
from lionagi import Branch, iModel

# Explicit provider + model
model = iModel(provider="openai", model="gpt-4.1-mini")

# Slash-syntax auto-detects provider from the model string
model = iModel(model="openai/gpt-4.1-mini")

# Branch uses the model for all LLM operations
branch = Branch(chat_model=model)
```

## Quick Reference

| Provider | `provider=` | Env Var for API Key | Endpoint(s) | Default Model |
|---|---|---|---|---|
| OpenAI | `"openai"` | `OPENAI_API_KEY` | `"chat"`, `"response"` | `gpt-4.1-mini` |
| Anthropic | `"anthropic"` | `ANTHROPIC_API_KEY` | `"chat"` / `"messages"` | -- |
| Google Gemini | `"gemini"` | `GEMINI_API_KEY` | `"chat"` | `gemini-2.5-flash` |
| Ollama | `"ollama"` | *(none required)* | `"chat"` | -- |
| Perplexity | `"perplexity"` | `PERPLEXITY_API_KEY` | `"chat"` | `sonar` |
| Groq | `"groq"` | `GROQ_API_KEY` | `"chat"` | `llama-3.3-70b-versatile` |
| OpenRouter | `"openrouter"` | `OPENROUTER_API_KEY` | `"chat"` | `google/gemini-2.5-flash` |
| NVIDIA NIM | `"nvidia_nim"` | `NVIDIA_NIM_API_KEY` | `"chat"`, `"embed"` | `meta/llama3-8b-instruct` |
| Exa | `"exa"` | `EXA_API_KEY` | `"search"` | -- |
| Claude Code CLI | `"claude_code"` | *(uses local CLI)* | `"query_cli"` | `sonnet` |
| Gemini CLI | `"gemini_code"` | *(uses local CLI)* | `"query_cli"` | `gemini-2.5-pro` |
| Codex CLI | `"codex"` | *(uses local CLI)* | `"query_cli"` | `gpt-5.3-codex` |

## Installation

```bash
# Core package (includes OpenAI, Anthropic, Gemini, Groq, OpenRouter,
# Perplexity, NVIDIA NIM, Exa support)
uv pip install lionagi

# For Ollama local models
uv pip install "lionagi[ollama]"
```

---

## OpenAI

### Setup

```bash
export OPENAI_API_KEY="sk-..."
```

### Basic Usage

```python
import asyncio
from lionagi import Branch, iModel

async def main():
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4.1-mini")
    )
    result = await branch.chat(
        instruction="Explain the difference between async and sync programming."
    )
    print(result)

asyncio.run(main())
```

If `OPENAI_API_KEY` is set and you do not pass a `chat_model`, Branch defaults to
`provider="openai"`, `model="gpt-4.1-mini"` automatically (configurable via the
`LIONAGI_CHAT_PROVIDER` and `LIONAGI_CHAT_MODEL` environment variables).

### Endpoints

OpenAI has two endpoint types:

- **Chat Completions** (`endpoint="chat"`, the default) -- standard
  `chat/completions` API.
- **Responses** (`endpoint="response"`) -- the newer `responses` API.

```python
# Responses endpoint
model = iModel(provider="openai", model="gpt-4.1-mini", endpoint="response")
```

### Structured Output

```python
import asyncio
from pydantic import BaseModel, Field
from lionagi import Branch, iModel

class SentimentResult(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="Confidence score 0-1")

async def main():
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4.1-mini")
    )
    result = await branch.operate(
        instruction="Analyze the sentiment of this review.",
        context="The product exceeded all my expectations!",
        operative_model=SentimentResult,
    )
    print(result)  # Operative with .output containing a SentimentResult

asyncio.run(main())
```

### Available Models

Any model available through the OpenAI API works. Common choices:

- `gpt-4.1-mini` (default, cost-effective)
- `gpt-4.1`
- `gpt-4o`
- `gpt-4o-mini`
- `o3-mini`

---

## Anthropic

### Setup

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Basic Usage

```python
import asyncio
from lionagi import Branch, iModel

async def main():
    branch = Branch(
        chat_model=iModel(provider="anthropic", model="claude-sonnet-4-20250514")
    )
    result = await branch.chat(
        instruction="Analyze this code for potential issues.",
        context="def divide(a, b): return a / b",
    )
    print(result)

asyncio.run(main())
```

The endpoint string can be `"chat"` or `"messages"` -- both resolve to the
Anthropic Messages API. System messages in the conversation are automatically
extracted and passed as the Anthropic `system` parameter.

### Prompt Caching

Anthropic supports prompt caching via `cache_control`. Pass it when invoking:

```python
result = await branch.chat(
    instruction="Summarize this long document.",
    context=long_document,
    cache_control=True,
)
```

### Available Models

- `claude-sonnet-4-20250514`
- `claude-opus-4-20250514`
- `claude-3-5-sonnet-20241022`
- `claude-3-5-haiku-20241022`

---

## Google Gemini (Native API)

### Setup

```bash
export GEMINI_API_KEY="..."
```

### Basic Usage

```python
import asyncio
from lionagi import Branch, iModel

async def main():
    branch = Branch(
        chat_model=iModel(provider="gemini", model="gemini-2.5-flash")
    )
    result = await branch.chat(
        instruction="Compare Python and Rust for systems programming."
    )
    print(result)

asyncio.run(main())
```

The Gemini provider uses Google's OpenAI-compatible endpoint at
`generativelanguage.googleapis.com/v1beta/openai`, so it accepts standard
chat-completion parameters.

### Available Models

- `gemini-2.5-flash` (default)
- `gemini-2.5-pro`
- `gemini-2.0-flash`

---

## Ollama (Local Models)

### Setup

```bash
# 1. Install Ollama (https://ollama.com)
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Install the lionagi Ollama extra
uv pip install "lionagi[ollama]"

# 3. Pull a model
ollama pull llama3.2:3b
```

No API key is needed. The endpoint defaults to `http://localhost:11434/v1`.

### Basic Usage

```python
import asyncio
from lionagi import Branch, iModel

async def main():
    branch = Branch(
        chat_model=iModel(provider="ollama", model="llama3.2:3b")
    )
    result = await branch.chat(
        instruction="Explain how transformers work in machine learning."
    )
    print(result)

asyncio.run(main())
```

### Auto-Pull

If the requested model is not available locally, LionAGI will pull it from the
Ollama registry automatically before the first call (with a progress bar via
`tqdm`).

### Custom Base URL

If Ollama runs on a different host or port:

```python
model = iModel(
    provider="ollama",
    model="llama3.2:3b",
    base_url="http://my-server:11434/v1",
)
```

---

## Perplexity

Perplexity provides real-time web search and Q&A through their Sonar API.

### Setup

```bash
export PERPLEXITY_API_KEY="pplx-..."
```

### Basic Usage

```python
import asyncio
from lionagi import Branch, iModel

async def main():
    branch = Branch(
        chat_model=iModel(provider="perplexity", model="sonar")
    )
    result = await branch.chat(
        instruction="What are the latest developments in quantum computing?"
    )
    print(result)

asyncio.run(main())
```

### Available Models

- `sonar` (default)
- `sonar-pro`
- `sonar-reasoning`
- `sonar-reasoning-pro`
- `sonar-deep-research`

### Provider-Specific Parameters

Perplexity supports additional parameters that can be passed as kwargs:

- `search_mode` -- `"default"` or `"academic"` (restricts to scholarly sources)
- `search_domain_filter` -- list of domains to include/exclude (prefix with `"-"`)
- `search_recency_filter` -- `"month"`, `"week"`, `"day"`, or `"hour"`
- `return_related_questions` -- `True`/`False`

---

## Groq

### Setup

```bash
export GROQ_API_KEY="gsk_..."
```

### Basic Usage

```python
import asyncio
from lionagi import Branch, iModel

async def main():
    branch = Branch(
        chat_model=iModel(provider="groq", model="llama-3.3-70b-versatile")
    )
    result = await branch.chat(
        instruction="Fast inference test -- explain recursion in one paragraph."
    )
    print(result)

asyncio.run(main())
```

Groq uses an OpenAI-compatible API, so standard chat-completion parameters
(`temperature`, `max_tokens`, `top_p`, etc.) are supported.

### Available Models

- `llama-3.3-70b-versatile` (default)
- `llama-3.1-8b-instant`
- `mixtral-8x7b-32768`

---

## OpenRouter

OpenRouter provides access to many models through a single API.

### Setup

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

### Basic Usage

```python
import asyncio
from lionagi import Branch, iModel

async def main():
    branch = Branch(
        chat_model=iModel(
            provider="openrouter",
            model="google/gemini-2.5-flash",
        )
    )
    result = await branch.chat(instruction="Hello from OpenRouter!")
    print(result)

asyncio.run(main())
```

OpenRouter is OpenAI-compatible. Use any model ID from the
[OpenRouter model list](https://openrouter.ai/models).

---

## NVIDIA NIM

### Setup

Get an API key from [build.nvidia.com](https://build.nvidia.com/).

```bash
export NVIDIA_NIM_API_KEY="nvapi-..."
```

### Chat Endpoint

```python
import asyncio
from lionagi import Branch, iModel

async def main():
    branch = Branch(
        chat_model=iModel(
            provider="nvidia_nim",
            model="meta/llama3-8b-instruct",
        )
    )
    result = await branch.chat(instruction="Explain GPU parallelism.")
    print(result)

asyncio.run(main())
```

### Embedding Endpoint

```python
embed_model = iModel(
    provider="nvidia_nim",
    endpoint="embed",
    model="nvidia/nv-embed-v1",
)
```

---

## Exa (Search)

Exa is a search provider, not a chat provider. It uses `endpoint="search"`.

### Setup

```bash
export EXA_API_KEY="..."
```

### Basic Usage

```python
from lionagi import iModel

search_model = iModel(provider="exa", endpoint="search")
```

Exa supports search-specific parameters including `query`, `category`,
`type` (`"keyword"`, `"neural"`, `"auto"`), `num_results`, `include_domains`,
`exclude_domains`, date filters, and content retrieval options.

---

## CLI Providers (Coding Agents)

LionAGI integrates three CLI-based coding agents as first-class iModel providers.
These wrap local CLI binaries (subprocess, not HTTP) and enable
**agent-to-agent orchestration** -- your outer agent uses lionagi to spawn and
coordinate inner coding agents.

### How CLI Endpoints Work

CLI providers inherit from `CLIEndpoint`, a subclass of `Endpoint` that replaces
HTTP transport with subprocess execution:

- **Subprocess execution** -- each call spawns the CLI binary and streams NDJSON
  from stdout, decoded incrementally (handles multibyte UTF-8 splits).
- **Session persistence** -- the endpoint stores a `session_id` from the first
  response and automatically passes `--resume` on subsequent calls.
- **Conservative concurrency** -- `DEFAULT_CONCURRENCY_LIMIT = 3`,
  `DEFAULT_QUEUE_CAPACITY = 10` (vs 100 for HTTP).
- **No API key needed** -- authentication is handled by the installed CLI tool.
- **Event handlers** -- optional callbacks (`on_text`, `on_tool_use`,
  `on_tool_result`, `on_final`) fire as the subprocess streams output.

```python
from lionagi import iModel

model = iModel(provider="claude_code", model="sonnet")

# Check if a model uses a CLI endpoint
model.is_cli  # True
```

### Prerequisites

Each CLI provider requires its binary installed and on `PATH`:

| Provider | Binary | Install Command |
|----------|--------|----------------|
| Claude Code | `claude` | `npm i -g @anthropic-ai/claude-code` |
| Gemini CLI | `gemini` | `npm i -g @anthropic-ai/gemini-cli` |
| Codex | `codex` | `npm i -g codex` |

LionAGI detects availability at import time via `shutil.which()`.

### Claude Code CLI

```python
import asyncio
from lionagi import Branch, iModel

async def main():
    model = iModel(
        provider="claude_code",
        model="sonnet",                          # "sonnet" or "opus"
        system_prompt="You are a code reviewer.", # system instructions
        permission_mode="bypassPermissions",      # skip approval prompts
        allowed_tools=["Read", "Grep", "Glob", "Bash"],
        max_turns=10,                            # conversation turn limit
    )
    branch = Branch(chat_model=model, name="reviewer")
    result = await branch.communicate("Review src/auth.py for security issues")
    print(result)

asyncio.run(main())
```

**Claude Code request parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"sonnet"` | `"sonnet"`, `"opus"`, or model ID |
| `system_prompt` | `str` | None | System instructions for the agent |
| `append_system_prompt` | `str` | None | Appended to existing system prompt |
| `permission_mode` | `str` | None | `"default"`, `"acceptEdits"`, `"bypassPermissions"` |
| `allowed_tools` | `list[str]` | None | Restrict to these tools only |
| `disallowed_tools` | `list[str]` | `[]` | Block specific tools |
| `max_turns` | `int` | None | Max conversation turns |
| `max_thinking_tokens` | `int` | None | Thinking budget |
| `continue_conversation` | `bool` | `False` | Continue previous session |
| `resume` | `str` | None | Resume a specific session ID |
| `ws` | `str` | None | Subdirectory within repo |
| `add_dir` | `str` | None | Extra read-only directory mount |
| `mcp_tools` | `list[str]` | `[]` | MCP tool names to enable |
| `mcp_servers` | `dict` | `{}` | MCP server configurations |
| `mcp_config` | `str\|Path` | None | Path to MCP config file |
| `auto_finish` | `bool` | `False` | Auto-append result extraction if agent doesn't finish |
| `verbose_output` | `bool` | `False` | Show full agent output |
| `cli_include_summary` | `bool` | `False` | Include cost/usage summary |

### Gemini CLI

```python
import asyncio
from lionagi import Branch, iModel

async def main():
    model = iModel(
        provider="gemini_code",
        model="gemini-2.5-pro",
        sandbox=True,                      # safety sandboxing (default)
        approval_mode="auto_edit",         # "suggest", "auto_edit", "full_auto"
    )
    branch = Branch(chat_model=model, name="gemini_agent")
    result = await branch.communicate("Analyze the project structure")
    print(result)

asyncio.run(main())
```

**Gemini CLI request parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"gemini-2.5-pro"` | Gemini model name |
| `system_prompt` | `str` | None | System instructions |
| `sandbox` | `bool` | `True` | Enable sandbox protection |
| `yolo` | `bool` | `False` | Auto-approve all actions (emits warning) |
| `approval_mode` | `str` | None | `"suggest"`, `"auto_edit"`, `"full_auto"` |
| `debug` | `bool` | `False` | Debug mode |
| `include_directories` | `list[str]` | `[]` | Extra directories to include |
| `ws` | `str` | None | Subdirectory within repo |
| `verbose_output` | `bool` | `False` | Show full agent output |
| `cli_include_summary` | `bool` | `False` | Include cost/usage summary |

### Codex CLI

```python
import asyncio
from lionagi import Branch, iModel

async def main():
    model = iModel(
        provider="codex",
        model="gpt-5.3-codex",
        sandbox="workspace-write",         # "read-only", "workspace-write", "danger-full-access"
        full_auto=True,                    # auto-approve with workspace-write sandbox
    )
    branch = Branch(chat_model=model, name="codex_agent")
    result = await branch.communicate("Fix the failing tests in this project")
    print(result)

asyncio.run(main())
```

**Codex CLI request parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"gpt-5.3-codex"` | Codex model name |
| `system_prompt` | `str` | None | System instructions |
| `full_auto` | `bool` | `False` | Auto-approve with workspace-write sandbox |
| `sandbox` | `str` | None | `"read-only"`, `"workspace-write"`, `"danger-full-access"` |
| `bypass_approvals` | `bool` | `False` | Skip all approvals and sandbox |
| `skip_git_repo_check` | `bool` | `False` | Don't require git repo |
| `output_schema` | `str\|Path` | None | JSON Schema file for structured output |
| `include_plan_tool` | `bool` | `False` | Enable planning tool |
| `images` | `list[str]` | `[]` | Image attachments |
| `config_overrides` | `dict` | `{}` | Custom config key-value overrides |
| `ws` | `str` | None | Subdirectory within repo |
| `verbose_output` | `bool` | `False` | Show full agent output |
| `cli_include_summary` | `bool` | `False` | Include cost/usage summary |

### Session Management

CLI providers automatically manage session state:

1. First call creates a new session.
2. The endpoint stores `session_id` from the response.
3. Subsequent calls pass `--resume` automatically.
4. `iModel.copy()` creates a fresh session; `iModel.copy(share_session=True)`
   carries over the session ID.

```python
model = iModel(provider="claude_code", model="sonnet")

# Fresh copy, independent session
new_model = model.copy()

# Copy that resumes the same CLI session
resumed = model.copy(share_session=True)
```

### Event Handlers

Set callbacks to observe the agent's streaming output:

```python
model = iModel(provider="claude_code", model="sonnet")

model.endpoint.update_handlers(
    on_text=lambda chunk: print(f"Text: {chunk.text}"),
    on_tool_use=lambda chunk: print(f"Tool: {chunk}"),
    on_thinking=lambda chunk: print(f"Thinking: {chunk.text}"),
    on_final=lambda session: print(f"Done: {session.result}"),
)
```

| Provider | Available Handlers |
|----------|-------------------|
| `claude_code` | `on_thinking`, `on_text`, `on_tool_use`, `on_tool_result`, `on_system`, `on_final` |
| `gemini_code` | `on_text`, `on_tool_use`, `on_tool_result`, `on_final` |
| `codex` | `on_text`, `on_tool_use`, `on_tool_result`, `on_final` |

For more on multi-agent orchestration patterns with CLI providers, see
[CLI Agent Providers](../for-ai-agents/claude-code-usage.md).

---

## Provider Auto-Detection

If the `model` string contains a `/`, the prefix is treated as the provider name.
This lets you skip the `provider=` parameter:

```python
from lionagi import iModel

# These two are equivalent:
m1 = iModel(provider="openai", model="gpt-4.1-mini")
m2 = iModel(model="openai/gpt-4.1-mini")

# Works with any provider
m3 = iModel(model="anthropic/claude-sonnet-4-20250514")
m4 = iModel(model="groq/llama-3.3-70b-versatile")
m5 = iModel(model="gemini/gemini-2.5-flash")
```

---

## OpenAI-Compatible Custom Providers

Any provider that implements the OpenAI chat completions API can be used with
LionAGI. When `match_endpoint` does not recognize the provider name, it falls
back to a generic OpenAI-compatible endpoint. Pass `base_url` to point at the
custom server:

```python
import asyncio
from lionagi import Branch, iModel

async def main():
    model = iModel(
        provider="together",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        base_url="https://api.together.xyz/v1",
        api_key="your-together-key",
    )
    branch = Branch(chat_model=model)
    result = await branch.chat(instruction="Hello from a custom provider!")
    print(result)

asyncio.run(main())
```

The fallback endpoint uses `chat/completions`, bearer auth, and
`application/json` content type.

---

## Async Context Manager

Both `iModel` and `Branch` support `async with` for resource cleanup:

```python
import asyncio
from lionagi import Branch, iModel

async def main():
    # iModel context manager -- stops the rate-limited executor on exit
    async with iModel(provider="openai", model="gpt-4.1-mini") as model:
        branch = Branch(chat_model=model)
        result = await branch.chat(instruction="Hello!")
        print(result)

    # Branch context manager -- flushes logs on exit
    async with Branch(
        chat_model=iModel(provider="openai", model="gpt-4.1-mini")
    ) as branch:
        result = await branch.chat(instruction="Hello again!")
        print(result)

asyncio.run(main())
```

---

## Copying Models

Use `iModel.copy()` to create an independent instance with the same
configuration but a fresh ID and executor. This is useful when you need
separate rate-limiting or session state:

```python
from lionagi import iModel

model = iModel(provider="openai", model="gpt-4.1-mini")

# Fresh copy, no shared state
model2 = model.copy()

# For CLI providers: share_session=True carries over the session ID
cli_model = iModel(provider="claude_code", model="sonnet")
cli_model2 = cli_model.copy(share_session=True)
```

---

## Rate Limiting and Concurrency

`iModel` wraps a `RateLimitedAPIExecutor` that handles queuing and throttling.
You can configure it at construction time:

```python
from lionagi import iModel

model = iModel(
    provider="openai",
    model="gpt-4.1-mini",
    queue_capacity=100,            # Max queued requests (default: 100, CLI: 10)
    capacity_refresh_time=60,      # Seconds between capacity refreshes
    limit_requests=50,             # Max requests per cycle
    limit_tokens=100_000,          # Max tokens per cycle
    concurrency_limit=10,          # Max concurrent streaming requests
)
```

CLI providers use lower defaults (`queue_capacity=10`,
`concurrency_limit=3`) because each call spawns a subprocess.

---

## Multiple Models in One Branch

A `Branch` maintains a `chat_model` and a `parse_model`. By default
`parse_model` mirrors `chat_model`, but you can set them independently:

```python
from lionagi import Branch, iModel

branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
    parse_model=iModel(provider="anthropic", model="claude-sonnet-4-20250514"),
)
```

You can also override the model per-call:

```python
fast_model = iModel(provider="groq", model="llama-3.3-70b-versatile")

# Use the fast model just for this one call
result = await branch.chat(
    instruction="Quick question.",
    imodel=fast_model,
)
```

---

## Environment Variable Reference

| Variable | Provider |
|---|---|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GEMINI_API_KEY` | Google Gemini |
| `GROQ_API_KEY` | Groq |
| `OPENROUTER_API_KEY` | OpenRouter |
| `PERPLEXITY_API_KEY` | Perplexity |
| `NVIDIA_NIM_API_KEY` | NVIDIA NIM |
| `EXA_API_KEY` | Exa |
| `LIONAGI_CHAT_PROVIDER` | Default provider (default: `openai`) |
| `LIONAGI_CHAT_MODEL` | Default model (default: `gpt-4.1-mini`) |

These can be set as environment variables or placed in `.env`, `.env.local`, or
`.secrets.env` files in your project root.

---

## Troubleshooting

**"Provider must be provided"** -- You passed a `model` string without a `/`
separator and did not set `provider=`. Either use slash syntax
(`model="openai/gpt-4.1-mini"`) or pass `provider=` explicitly.

**"API key is required for authentication"** -- The environment variable for your
provider is not set. Check the table above for the correct variable name.

**"ollama is not installed"** -- Install the Ollama extra:
`uv pip install "lionagi[ollama]"`.

**Slow CLI providers** -- CLI providers spawn subprocesses. If calls seem slow,
that is expected. Avoid high concurrency with CLI providers.

**Rate limit errors (429)** -- LionAGI retries with exponential backoff
automatically. If you still hit limits, reduce `limit_requests` or
`limit_tokens` in the `iModel` constructor.
