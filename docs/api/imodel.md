# `iModel`

```python
class iModel(Element)
```

Uniform interface to any LLM provider with rate limiting, queuing, and hooks.

## Constructor

```python
model = li.iModel(
    provider="openai",
    model="gpt-4o",
    api_key=None,           # falls back to OPENAI_API_KEY env var
    limit_requests=100,
    limit_tokens=100_000,
)
```

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `provider` | `str \| None` | `None` | `"openai"`, `"anthropic"`, etc. Inferred from `model` if set |
| `base_url` | `str \| None` | `None` | Custom API URL (for proxies, local endpoints) |
| `endpoint` | `str \| Endpoint` | `"chat"` | Endpoint type (see table below) |
| `api_key` | `str \| None` | `None` | Explicit key; falls back to env var |
| `queue_capacity` | `int \| None` | auto | Max queued requests before backpressure |
| `capacity_refresh_time` | `float` | `60` | Seconds between queue capacity refreshes |
| `interval` | `float \| None` | auto | Queue processing interval in seconds |
| `limit_requests` | `int \| None` | `None` | Max requests per rate-limit cycle |
| `limit_tokens` | `int \| None` | `None` | Max tokens per rate-limit cycle |
| `concurrency_limit` | `int \| None` | `None` | Max concurrent streams |
| `hook_registry` | `HookRegistry \| dict \| None` | `HookRegistry()` | Pre/post invocation hooks |
| `**kwargs` | — | — | Provider-specific config (e.g., `model="gpt-4o"`, `temperature=0.7`) |

## Endpoint types

| Value | Transport | Used with |
|-------|-----------|----------|
| `"chat"` | REST API | OpenAI, Anthropic, Gemini, Ollama, etc. |
| `"claude_code"` | CLI subprocess | `Branch.run()` streaming |
| `"codex"` | CLI subprocess | `Branch.run()` streaming |
| `"gemini_code"` | CLI subprocess | `Branch.run()` streaming |

CLI endpoints set `is_cli = True`; `Branch.operate()` routes to `run_and_collect`
instead of `communicate`. See [operations.md#middle-protocol](operations.md#middle-protocol).

## Common construction patterns

```python
import lionagi as li

# OpenAI (default)
model = li.iModel(model="gpt-4o")

# Anthropic
model = li.iModel(provider="anthropic", model="claude-opus-4-7-20251001")

# With rate limits
model = li.iModel(model="gpt-4o", limit_requests=100, limit_tokens=100_000)

# Ollama local
model = li.iModel(
    provider="ollama",
    base_url="http://localhost:11434",
    model="llama3",
)

# NVIDIA NIM
model = li.iModel(provider="nvidia", model="meta/llama-3.1-70b-instruct")

# CLI endpoint (streaming — use with Branch.run())
model = li.iModel(provider="claude_code", model="sonnet")
```

## Public methods

### `invoke()`

```python
api_call = await model.invoke(
    messages=[{"role": "user", "content": "hello"}],
    temperature=0.7,
)
response_text = api_call.response
```

Sends a rate-limited request. Returns `APICalling` with `.response` attribute.

### `stream()`

```python
async for chunk in await model.stream(messages=[...]):
    print(chunk, end="", flush=True)
```

Streaming request. Prefer `Branch.run()` for managed streaming with message history.

### `create_api_calling()`

```python
api_call = model.create_api_calling(
    messages=[{"role": "user", "content": "hello"}],
)
# inspect before invoking
result = await model.invoke(api_call)
```

Constructs an `APICalling` object without sending the request.

### `copy()`

```python
model2 = model.copy(share_session=False)
```

Creates a fresh `iModel` with the same config but a new ID and executor.
Use when you need independent rate-limit buckets for parallel workflows.

### `close()`

```python
await model.close()
```

Stops the executor and releases resources. Not needed when using as context manager.

## Context manager

```python
async with li.iModel(model="gpt-4o") as model:
    api_call = await model.invoke(messages=[{"role": "user", "content": "hello"}])
    print(api_call.response)
# executor closed automatically
```

## Properties

| Property | Type | Notes |
|----------|------|-------|
| `model_name` | `str` | Model identifier string |
| `is_cli` | `bool` | `True` for CLI endpoints (`claude_code`, `codex`, `gemini_code`) |
| `request_options` | `type[BaseModel] \| None` | Endpoint-specific request schema |
| `provider_session_id` | `str \| None` | CLI session ID for resumption |

## Provider resolution

Provider is inferred from `model` kwarg when it contains a slash (e.g., `"anthropic/claude-opus-4-7"`).
Otherwise set `provider` explicitly.

| `provider` string | API | Key env var |
|------------------|-----|------------|
| `openai` | OpenAI | `OPENAI_API_KEY` |
| `anthropic` | Anthropic | `ANTHROPIC_API_KEY` |
| `gemini` | Google AI | `GOOGLE_API_KEY` |
| `ollama` | Ollama local | — (no key needed) |
| `nvidia` | NVIDIA NIM | `NVIDIA_API_KEY` |
| `perplexity` | Perplexity | `PERPLEXITY_API_KEY` |
| `groq` | Groq | `GROQ_API_KEY` |
| `openrouter` | OpenRouter | `OPENROUTER_API_KEY` |

## HookRegistry

Pre/post invocation hooks for logging, caching, or metrics:

```python
from lionagi.service.hooks import HookRegistry, HookEventTypes

async def log_pre(event, **kw):
    print(f"Sending: {type(event).__name__}")

async def log_post(event, **kw):
    print(f"Received: {type(event).__name__}")

hooks = HookRegistry(
    hooks={
        HookEventTypes.PreInvocation: log_pre,
        HookEventTypes.PostInvocation: log_post,
    }
)

model = li.iModel(model="gpt-4o", hook_registry=hooks)
```

## Serialization

```python
data = model.to_dict()
restored = li.iModel.from_dict(data)
```

Next: [Operations & extension](operations.md) — Middle protocol and param types
