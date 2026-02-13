# Models and Providers

LionAGI's `iModel` provides a unified interface for working with different LLM providers. Configure once, swap providers without changing application code.

## Basic Usage

```python
from lionagi import Branch, iModel

branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1-mini")
)

response = await branch.communicate("Explain quantum computing")
```

If you do not specify a model, Branch uses the default from environment configuration: `LIONAGI_CHAT_PROVIDER` (default: `openai`) and `LIONAGI_CHAT_MODEL` (default: `gpt-4.1-mini`).

## Supported Providers

### API-Based Providers

These providers communicate with hosted APIs over HTTP.

**OpenAI** -- API key: `OPENAI_API_KEY`

```python
# Default model (gpt-4.1-mini)
branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1-mini")
)

# Other OpenAI models
gpt4 = iModel(provider="openai", model="gpt-4.1")
gpt4o = iModel(provider="openai", model="gpt-4o")
gpt4o_mini = iModel(provider="openai", model="gpt-4o-mini")
```

OpenAI also supports `endpoint="response"` for the Responses API:

```python
response_model = iModel(provider="openai", endpoint="response", model="gpt-4.1")
```

**Anthropic** -- API key: `ANTHROPIC_API_KEY`

```python
branch = Branch(
    chat_model=iModel(provider="anthropic", model="claude-sonnet-4-5-20250929")
)

# Other Anthropic models
haiku = iModel(provider="anthropic", model="claude-haiku-4-5-20251001")
opus = iModel(provider="anthropic", model="claude-opus-4-6")
```

Anthropic supports prompt caching via `cache_control=True` on individual calls.

**Gemini (Native API)** -- API key: `GEMINI_API_KEY`

Gemini uses Google's OpenAI-compatible endpoint:

```python
branch = Branch(
    chat_model=iModel(provider="gemini", model="gemini-2.5-flash")
)

# Other Gemini models
gemini_pro = iModel(provider="gemini", model="gemini-2.0-flash")
```

**Groq** -- API key: `GROQ_API_KEY`

Fast inference for open models:

```python
branch = Branch(
    chat_model=iModel(provider="groq", model="llama-3.3-70b-versatile")
)
```

**OpenRouter** -- API key: `OPENROUTER_API_KEY`

Access many models through a single API:

```python
branch = Branch(
    chat_model=iModel(provider="openrouter", model="google/gemini-2.5-flash")
)
```

**Perplexity** -- API key: `PERPLEXITY_API_KEY`

Real-time web search and Q&A via the Sonar API:

```python
branch = Branch(
    chat_model=iModel(provider="perplexity", model="sonar")
)
```

**NVIDIA NIM** -- API key: `NVIDIA_NIM_API_KEY`

Cloud-hosted models on NVIDIA infrastructure:

```python
# Chat models
branch = Branch(
    chat_model=iModel(provider="nvidia_nim", model="meta/llama3-8b-instruct")
)

# Embedding models
embed_model = iModel(provider="nvidia_nim", endpoint="embed", model="nvidia/nv-embed-v1")
```

**Exa** -- API key: `EXA_API_KEY`

Semantic search (not a chat provider):

```python
exa = iModel(provider="exa", endpoint="search")
```

**Ollama** -- Local models, no API key required

```python
branch = Branch(
    chat_model=iModel(
        provider="ollama",
        model="llama3",
        base_url="http://localhost:11434"
    )
)
```

### CLI-Based Providers

CLI providers wrap agentic coding tools that run as subprocesses. They differ from API providers in several ways:

- They spawn subprocesses instead of making HTTP requests
- They maintain sessions with resume capability
- They use NDJSON streaming over stdin/stdout
- Default concurrency is limited (3 concurrent, queue capacity of 10)

**Claude Code** -- Uses installed `claude` CLI

```python
claude_code = iModel(provider="claude_code")

branch = Branch(chat_model=claude_code)
result = await branch.communicate("Refactor the auth module")
```

**Gemini CLI** -- Uses installed `gemini` CLI

```python
gemini_cli = iModel(provider="gemini_code")

branch = Branch(chat_model=gemini_cli)
result = await branch.communicate("Review this codebase")
```

**Codex CLI** -- Uses installed `codex` CLI

```python
codex = iModel(provider="codex")

branch = Branch(chat_model=codex)
result = await branch.communicate("Write tests for the parser module")
```

### OpenAI-Compatible Providers

Any provider with an OpenAI-compatible API can be used by specifying `base_url`:

```python
custom = iModel(
    provider="custom",
    model="my-model",
    base_url="https://my-provider.example.com/v1",
    api_key="my-api-key"
)
```

## iModel Constructor

The full constructor signature with all parameters:

```python
model = iModel(
    # Provider and endpoint
    provider="openai",                  # Provider name
    model="gpt-4.1-mini",              # Model name (passed via **kwargs)
    endpoint="chat",                    # Endpoint type (default: "chat")
    base_url=None,                      # Custom base URL
    api_key=None,                       # API key (defaults to env var)

    # Rate limiting
    queue_capacity=100,                 # Max queued requests (10 for CLI)
    capacity_refresh_time=60,           # Queue refresh interval (seconds)
    interval=None,                      # Processing interval
    limit_requests=None,               # Max requests per cycle
    limit_tokens=None,                 # Max tokens per cycle
    concurrency_limit=None,            # Max concurrent requests (3 for CLI)

    # Streaming
    streaming_process_func=None,       # Custom chunk processor

    # Hooks
    hook_registry=None,                # HookRegistry for pre/post hooks
    exit_hook=False,                   # Enable exit hooks

    # Model-specific parameters (passed to the endpoint)
    temperature=0.7,
    max_tokens=2000,
)
```

## Async Context Manager

Use `iModel` as an async context manager for automatic resource cleanup:

```python
async with iModel(provider="openai", model="gpt-4.1") as model:
    branch = Branch(chat_model=model)
    result = await branch.communicate("Hello")
    # Executor is stopped and resources released on exit
```

## Copying Models

Use `copy()` to create an independent `iModel` instance with the same configuration but a fresh ID and executor:

```python
original = iModel(provider="openai", model="gpt-4.1-mini")

# Fresh instance, independent executor
clone = original.copy()

# For CLI endpoints, optionally share the session for resume
cli_clone = cli_model.copy(share_session=True)
```

This is particularly useful when creating multiple branches that need independent rate limiting.

## Branch with Separate Chat and Parse Models

Branch supports separate models for chat and structured parsing:

```python
branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1"),       # For conversations
    parse_model=iModel(provider="openai", model="gpt-4.1-mini"), # For parsing
    system="Technical assistant"
)

# communicate() uses chat_model
await branch.communicate("Explain this concept")

# parse() uses parse_model
result = await branch.parse(some_text, response_format=MyModel)
```

If `parse_model` is not specified, it defaults to the same model as `chat_model`.

## Multiple Providers in One Session

Mix different providers for different tasks:

```python
from lionagi import Session, Branch, iModel

session = Session()

fast_branch = session.new_branch(
    name="fast",
    system="Quick answers",
    imodel=iModel(provider="openai", model="gpt-4.1-mini")
)

deep_branch = session.new_branch(
    name="deep",
    system="Detailed analysis",
    imodel=iModel(provider="anthropic", model="claude-sonnet-4-5-20250929")
)

# Route tasks to the appropriate model
quick = await fast_branch.communicate("What is 2+2?")
analysis = await deep_branch.communicate("Analyze the implications of quantum computing on cryptography")
```

## Model Configuration

Configure model parameters at construction time:

```python
configured = iModel(
    provider="openai",
    model="gpt-4.1",
    temperature=0.7,
    max_tokens=2000,
    limit_requests=100,
    limit_tokens=50000
)

branch = Branch(chat_model=configured, system="Creative writer")
```

## Environment Configuration

LionAGI loads API keys and defaults from environment variables:

| Variable                   | Purpose                        | Default           |
|----------------------------|--------------------------------|-------------------|
| `OPENAI_API_KEY`           | OpenAI authentication          | --                |
| `ANTHROPIC_API_KEY`        | Anthropic authentication       | --                |
| `GEMINI_API_KEY`           | Gemini authentication          | --                |
| `GROQ_API_KEY`             | Groq authentication            | --                |
| `OPENROUTER_API_KEY`       | OpenRouter authentication      | --                |
| `PERPLEXITY_API_KEY`       | Perplexity authentication      | --                |
| `NVIDIA_NIM_API_KEY`       | NVIDIA NIM authentication      | --                |
| `EXA_API_KEY`              | Exa authentication             | --                |
| `LIONAGI_CHAT_PROVIDER`    | Default chat provider          | `openai`          |
| `LIONAGI_CHAT_MODEL`       | Default chat model             | `gpt-4.1-mini`   |

Settings are loaded from `.env`, `.env.local`, or `.secrets.env` files automatically via pydantic-settings.

## Provider Comparison

| Provider     | Type | Default Model                        | Auth Key Env Var       |
|-------------|------|--------------------------------------|------------------------|
| `openai`    | API  | `gpt-4.1-mini`                       | `OPENAI_API_KEY`       |
| `anthropic` | API  | --                                   | `ANTHROPIC_API_KEY`    |
| `gemini`    | API  | `gemini-2.5-flash`                   | `GEMINI_API_KEY`       |
| `groq`      | API  | `llama-3.3-70b-versatile`            | `GROQ_API_KEY`         |
| `openrouter`| API  | `google/gemini-2.5-flash`            | `OPENROUTER_API_KEY`   |
| `perplexity`| API  | `sonar`                              | `PERPLEXITY_API_KEY`   |
| `nvidia_nim`| API  | `meta/llama3-8b-instruct`            | `NVIDIA_NIM_API_KEY`   |
| `ollama`    | API  | --                                   | --                     |
| `exa`       | API  | -- (search only)                     | `EXA_API_KEY`          |
| `claude_code`| CLI | --                                   | -- (uses CLI auth)     |
| `gemini_code`| CLI | --                                   | -- (uses CLI auth)     |
| `codex`     | CLI  | --                                   | -- (uses CLI auth)     |
