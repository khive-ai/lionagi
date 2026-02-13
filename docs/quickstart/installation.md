# Installation

## Install LionAGI

=== "pip"

    ```bash
    pip install lionagi
    ```

=== "uv"

    ```bash
    uv add lionagi
    ```

**Requirements**: Python >= 3.10

## Configure API Keys

LionAGI loads API keys from environment variables or `.env` files automatically (via `pydantic-settings` and `python-dotenv`). Set at least one provider key.

### Environment variables

```bash
# OpenAI (default provider)
export OPENAI_API_KEY=sk-...

# Or any other provider
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
export NVIDIA_NIM_API_KEY=nvapi-...
export GROQ_API_KEY=gsk_...
export PERPLEXITY_API_KEY=pplx-...
export OPENROUTER_API_KEY=sk-or-...
```

### `.env` file

Create a `.env` file in your project root:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

LionAGI checks `.env`, `.env.local`, and `.secrets.env` in that order.

### Ollama (local models)

Ollama requires no API key. Install and run Ollama, then:

```python
from lionagi import Branch, iModel

branch = Branch(
    chat_model=iModel(provider="ollama", model="llama3.2")
)
```

## Verify Installation

```python
import lionagi

print(lionagi.__version__)
```

To verify an API connection works:

```python
import asyncio
from lionagi import Branch

async def main():
    branch = Branch()
    reply = await branch.communicate("Say hello in one word.")
    print(reply)

asyncio.run(main())
```

If this prints a response, your installation and API key are working.

## Optional Extras

LionAGI ships with optional dependencies for specific features:

```bash
# Install with specific extras
pip install "lionagi[ollama]"    # Ollama client library
pip install "lionagi[mcp]"      # MCP (Model Context Protocol) server support
pip install "lionagi[reader]"   # Document reading (docling)
pip install "lionagi[rich]"     # Rich terminal output
pip install "lionagi[schema]"   # Schema code generation
pip install "lionagi[graph]"    # Graph visualization (matplotlib, networkx)
pip install "lionagi[postgres]" # PostgreSQL storage
pip install "lionagi[sqlite]"   # SQLite storage
pip install "lionagi[xml]"      # XML parsing

# Install everything
pip install "lionagi[all]"
```

## Provider Reference

| Provider | `provider=` | Environment Variable | Notes |
| --- | --- | --- | --- |
| OpenAI | `"openai"` | `OPENAI_API_KEY` | Default provider. GPT-4.1, GPT-4o, o-series. |
| Anthropic | `"anthropic"` | `ANTHROPIC_API_KEY` | Claude models via Messages API. |
| Google Gemini | `"gemini"` | `GEMINI_API_KEY` | Uses OpenAI-compatible endpoint. |
| Ollama | `"ollama"` | *(none)* | Local models. Requires Ollama running on `localhost:11434`. |
| NVIDIA NIM | `"nvidia_nim"` | `NVIDIA_NIM_API_KEY` | NVIDIA inference microservices. |
| Groq | `"groq"` | `GROQ_API_KEY` | Fast inference on LPU hardware. |
| Perplexity | `"perplexity"` | `PERPLEXITY_API_KEY` | Search-augmented responses. |
| OpenRouter | `"openrouter"` | `OPENROUTER_API_KEY` | Access 200+ models via single API. |

### Custom / OpenAI-compatible endpoints

Any endpoint that implements the OpenAI chat completions API:

```python
from lionagi import iModel

model = iModel(
    provider="my_custom_provider",
    model="my-model-name",
    api_key="your-api-key",
    base_url="https://your-endpoint.com/v1",
)
```

## Defaults

When you create a `Branch()` without specifying a model, LionAGI uses:

- **Provider**: `openai` (configurable via `LIONAGI_CHAT_PROVIDER` env var)
- **Model**: `gpt-4.1-mini` (configurable via `LIONAGI_CHAT_MODEL` env var)

```bash
# Override defaults via environment
export LIONAGI_CHAT_PROVIDER=anthropic
export LIONAGI_CHAT_MODEL=claude-sonnet-4-20250514
```

## Next Steps

Ready to write code? Continue to [Quick Start](your-first-flow.md).
