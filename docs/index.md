# LionAGI

**Provider-agnostic LLM orchestration for structured, multi-step AI workflows.**

LionAGI gives you typed, observable control over LLM calls. Instead of dumping a prompt into a black-box chain, you build explicit workflows from `Branch` (a single conversation thread), `iModel` (any provider behind a uniform interface), and `Session` (multi-branch orchestration with graph-based execution). Every call is async, every response can be parsed into Pydantic models, and every tool invocation is logged.

---

## Why LionAGI

- **One interface, any provider.** Switch between OpenAI, Anthropic, Gemini, Ollama, NVIDIA NIM, Groq, Perplexity, and OpenRouter without changing your workflow code. Bring your own OpenAI-compatible endpoint if none of these fit.
- **Structured output by default.** `communicate()` and `operate()` parse LLM responses directly into Pydantic models. No regex, no fragile string extraction.
- **Tool calling built in.** Register Python functions as tools, and `operate()` handles schema generation, LLM function-call requests, invocation, and result injection automatically.
- **Graph-based workflows.** `Builder` composes operations into a DAG. `Session.flow()` executes it across branches with dependency tracking and parallel execution.

---

## Installation

=== "pip"

    ```bash
    pip install lionagi
    ```

=== "uv"

    ```bash
    uv add lionagi
    ```

Set your API key (at minimum, one provider):

```bash
export OPENAI_API_KEY=sk-...
```

LionAGI defaults to `openai` with `gpt-4.1-mini`. See [Installation](quickstart/installation.md) for all providers and optional extras.

---

## Hello World

```python
import asyncio
from lionagi import Branch

async def main():
    branch = Branch(system="You are a helpful assistant.")
    result = await branch.communicate("What is the capital of France?")
    print(result)

asyncio.run(main())
```

`communicate()` sends a message, adds both the instruction and the response to the conversation history, and returns the response as a string. No iModel needed for defaults -- Branch auto-creates one from your `OPENAI_API_KEY`.

---

## What Makes It Different

### Multi-provider, same code

```python
from lionagi import Branch, iModel

openai_branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1-mini")
)

anthropic_branch = Branch(
    chat_model=iModel(provider="anthropic", model="claude-sonnet-4-20250514")
)

gemini_branch = Branch(
    chat_model=iModel(provider="gemini", model="gemini-2.5-flash")
)
```

### Structured output with `response_format`

```python
from pydantic import BaseModel
from lionagi import Branch

class City(BaseModel):
    name: str
    country: str
    population: int

async def main():
    branch = Branch()
    city = await branch.communicate(
        "Tell me about Tokyo.",
        response_format=City,
    )
    print(city.name, city.population)  # Typed Pydantic model
```

### Tool calling with `operate()`

```python
from lionagi import Branch

def get_weather(city: str, unit: str = "celsius") -> str:
    """Get current weather for a city."""
    return f"22 degrees {unit} in {city}, partly cloudy"

async def main():
    branch = Branch(tools=[get_weather])
    result = await branch.operate(
        instruction="What's the weather in Tokyo?",
        actions=True,
    )
    print(result)  # LLM called get_weather, got the result, and responded
```

### Graph workflows with `Session` and `Builder`

```python
from lionagi import Session, Builder

session = Session()
builder = Builder()

# Define a two-step workflow
step1 = builder.add_operation(
    "communicate",
    instruction="List 3 startup ideas in AI.",
)
step2 = builder.add_operation(
    "communicate",
    instruction="Pick the most feasible idea and explain why.",
    depends_on=[step1],
)

async def main():
    results = await session.flow(builder.get_graph())
    print(results)
```

---

## Supported Providers

| Provider | `provider=` | Environment Variable | Example Model |
| --- | --- | --- | --- |
| OpenAI | `"openai"` | `OPENAI_API_KEY` | `gpt-4.1-mini` |
| Anthropic | `"anthropic"` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |
| Google Gemini | `"gemini"` | `GEMINI_API_KEY` | `gemini-2.5-flash` |
| Ollama | `"ollama"` | *(none, local)* | `llama3.2` |
| NVIDIA NIM | `"nvidia_nim"` | `NVIDIA_NIM_API_KEY` | `meta/llama-3.1-70b-instruct` |
| Groq | `"groq"` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| Perplexity | `"perplexity"` | `PERPLEXITY_API_KEY` | `sonar-pro` |
| OpenRouter | `"openrouter"` | `OPENROUTER_API_KEY` | `anthropic/claude-sonnet-4` |
| OpenAI-compatible | *(any string)* | *(pass `api_key=` directly)* | *(your model)* |

```python
# OpenAI-compatible custom endpoint
custom = iModel(
    provider="my_provider",
    model="my-model",
    api_key="your-key",
    base_url="https://your-endpoint.com/v1",
)
```

---

## Learning Path

| Step | Topic | Link |
| --- | --- | --- |
| 1 | Install and verify | [Installation](quickstart/installation.md) |
| 2 | First working examples | [Quick Start](quickstart/your-first-flow.md) |
| 3 | Core abstractions | [Core Concepts](core-concepts/index.md) |
| 4 | Operations in depth | [Operations](core-concepts/operations.md) |
| 5 | Branch and Session | [Sessions & Branches](core-concepts/sessions-and-branches.md) |
| 6 | Workflow patterns | [Patterns](patterns/index.md) |
| 7 | Provider details | [LLM Providers](integrations/llm-providers.md) |
| 8 | Advanced topics | [Advanced](advanced/index.md) |
| 9 | API reference | [Reference](reference/api/index.md) |

---

## Key Concepts at a Glance

**Branch** -- A single conversation thread. Manages messages, tools, and model instances. All LLM operations (`chat`, `communicate`, `operate`, `parse`, `ReAct`) are Branch methods.

**iModel** -- Wraps any LLM provider behind a uniform async interface. Handles rate limiting, retries, and request/response translation.

**Session** -- Manages multiple Branches. Executes graph-based workflows via `session.flow()`.

**Builder** -- Constructs operation graphs (DAGs) for `Session.flow()`. Supports sequential dependencies, parallel fan-out, and dynamic expansion from results.

---

## Requirements

- Python >= 3.10
- At least one LLM provider API key (or Ollama running locally)

---

*Apache 2.0 License* | [GitHub](https://github.com/khive-ai/lionagi) | [Discord](https://discord.gg/lionagi)
