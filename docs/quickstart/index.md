# Installation

## Install LionAGI

**Recommended**: Use `uv` for faster dependency resolution:
```bash
uv add lionagi
```

**Alternative**: Standard pip installation:
```bash
pip install lionagi
```

## Configure API Keys

Create a `.env` file in your project root:

```bash
# At minimum, add one provider
OPENAI_API_KEY=your_key_here

# Optional: Additional providers
ANTHROPIC_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
NVIDIA_NIM_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
PERPLEXITY_API_KEY=your_key_here
EXA_API_KEY=your_key_here
```

LionAGI automatically loads from `.env` - no manual configuration needed.

## Verify Installation

Run this test to confirm everything works:

```python
from lionagi import Branch, iModel

async def test():
    gpt4 = iModel(provider="openai", model="gpt-4o-mini")
    branch = Branch(chat_model=gpt4)
    reply = await branch.chat("Hello from LionAGI!")
    print(f"LionAGI says: {reply}")

if __name__ == "__main__":
    import anyio
    anyio.run(test)
```

Expected output: A conversational response from the model.

## Supported Providers

LionAGI comes pre-configured for these providers:

- **`openai`** - GPT-5, GPT-4.1, o4-mini
- **`anthropic`** - Claude 4.5 Sonnet, Claude 4.1 Opus
- **`claude_code`** - Claude Code SDK integration
- **`ollama`** - Local model hosting
- **`openrouter`** - Access 200+ models via single API
- **`nvidia_nim`** - NVIDIA inference microservices
- **`groq`** - Fast inference on LPU hardware
- **`perplexity`** - Search-augmented responses

### Custom Providers

**OpenAI-compatible endpoints**:
```python
custom = iModel(
    provider="openai_compatible",
    model="custom-model",
    api_key="your_key",
    base_url="https://custom-endpoint.com/v1"
)
```
