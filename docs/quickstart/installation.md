# Installation

```bash
uv add lionagi
```

Set your API key:

```bash
OPENAI_API_KEY=your-key-here
```

Test it works:

```python
from lionagi import Branch, iModel

agent = Branch(chat_model=iModel(provider="openai", model="gpt-4.1-mini"))
response = await agent.communicate("Hello")
```

## Provider Options

```python
# OpenAI
iModel(provider="openai", model="gpt-4.1-mini")

# Anthropic  
iModel(provider="anthropic", model="claude-3-5-sonnet-20241022")

# Local (Ollama)
iModel(provider="ollama", model="llama3")
```
