# Models and Providers

LionAGI's `iModel` provides a unified interface for working with different LLM
providers.

## Basic Usage

```python
from lionagi import Branch, iModel

assistant = Branch(
    chat_model=iModel(provider="openai", model="gpt-4")
)

response = await assistant.chat("Explain quantum computing")
```

## Supported Providers

**OpenAI**: API key from `OPENAI_API_KEY` environment variable

```python
openai_branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="Quantum computing expert"
)
```

**Anthropic**: API key from `ANTHROPIC_API_KEY` environment variable

```python
claude_branch = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-5-sonnet-20240620"),
    system="Python programming expert"
)
```

**Ollama**: Local models

```python
local_branch = Branch(
    chat_model=iModel(
        provider="ollama", 
        model="llama2",
        base_url="http://localhost:11434"
    ),
    system="Local assistant"
)
```

## Multiple Providers

Mix different providers in a single session:

```python
from lionagi import Session, Branch, iModel

session = Session()

fast_branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="Quick answers",
    name="fast"
)

analytical_branch = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-5-sonnet-20240620"),
    system="Detailed analysis",
    name="analytical"
)

session.include_branches([fast_branch, analytical_branch])

# Use different models for different tasks
quick = await fast_branch.chat("What is 2+2?")
analysis = await analytical_branch.chat("Analyze AI implications")
```

## Model Configuration

Configure model parameters:

```python
from lionagi import Branch, iModel

configured_model = iModel(
    provider="openai",
    model="gpt-4",
    temperature=0.7,        # Randomness (0.0-1.0) 
    max_tokens=2000,       # Response length limit
    limit_requests=100,    # Rate limiting
    limit_tokens=50000
)

writer = Branch(
    chat_model=configured_model,
    system="Creative writer"
)
```

## Environment Configuration

Load settings from environment variables:

```python
import os
from lionagi import Branch, iModel

model = iModel(
    provider=os.getenv("LLM_PROVIDER", "openai"),
    model=os.getenv("LLM_MODEL", "gpt-4"),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.7"))
)

assistant = Branch(chat_model=model)
```

## Model Selection

Choose models based on task requirements:

```python
from lionagi import Branch, iModel

def select_model(task_type: str):
    if task_type == "simple_qa":
        return iModel(provider="openai", model="gpt-4")
    elif task_type == "complex_reasoning": 
        return iModel(provider="anthropic", model="claude-3-5-sonnet-20240620")
    elif task_type == "private_data":
        return iModel(provider="ollama", model="llama2")
    else:
        return iModel(provider="openai", model="gpt-4")  # Default

# Use appropriate model for task
model = select_model("complex_reasoning")
assistant = Branch(chat_model=model, system="Analytical assistant")
```

## Best Practices

**Use descriptive branch names** and appropriate models:

```python
# Different models for different needs
fast_qa = Branch(
    chat_model=iModel(provider="openai", model="gpt-4", temperature=0.3),
    system="Quick, consistent answers",
    name="qa_assistant"
)

analytical = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-5-sonnet-20240620"),
    system="Detailed analysis", 
    name="analyst"
)
```

**Handle errors gracefully**:

```python
models_to_try = [
    ("openai", "gpt-4"),
    ("anthropic", "claude-3-5-sonnet-20240620"),
    ("ollama", "llama2")
]

for provider, model in models_to_try:
    try:
        assistant = Branch(chat_model=iModel(provider=provider, model=model))
        response = await assistant.chat("Hello!")
        break  # Use first successful model
    except Exception as e:
        continue
```

**Pre-configure models** for different scenarios:

```python
CONFIGS = {
    "development": iModel(provider="openai", model="gpt-4", temperature=0.3),
    "production": iModel(provider="anthropic", model="claude-3-5-sonnet-20240620"),
    "creative": iModel(provider="openai", model="gpt-4", temperature=0.9)
}

dev_assistant = Branch(chat_model=CONFIGS["development"])
```

LionAGI's iModel provides a consistent interface across providers while handling
API differences and configuration automatically.
