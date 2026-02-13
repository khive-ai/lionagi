# For AI Agents

This section is written for AI coding agents -- Claude Code, Codex, Gemini CLI,
and similar tools -- that need to use lionagi programmatically. The content
prioritizes machine-readable structure, lookup tables, and copy-paste code
patterns over narrative explanation.

## What lionagi Is

lionagi is a provider-agnostic LLM orchestration SDK. It lets your agent call
any LLM provider through a uniform interface, manage conversation state, invoke
tools, parse structured output, and coordinate multi-step workflows as DAGs.

## Core Objects (4 things to know)

| Object | What It Does | Import |
|--------|-------------|--------|
| `Branch` | Single conversation thread. Primary API surface. | `from lionagi import Branch` |
| `iModel` | LLM provider wrapper (API or CLI). | `from lionagi import iModel` |
| `Session` | Multi-branch orchestrator. Manages Branch lifecycle. | `from lionagi import Session` |
| `Builder` | DAG builder for multi-step workflows. | `from lionagi import Builder` |

## Branch Methods (the API you will use most)

| Method | Adds to History | Tool Calling | Structured Output | Use Case |
|--------|:-:|:-:|:-:|---------|
| `chat()` | No | No | Optional | Low-level LLM call for orchestration |
| `communicate()` | Yes | No | Optional | Conversational interaction |
| `operate()` | Yes | Yes | Optional | Tool use + structured output |
| `ReAct()` | Yes | Yes | Optional | Multi-step reasoning with tools |
| `parse()` | No | No | Yes | Extract structured data from text |
| `interpret()` | No | No | No | Rewrite/refine a prompt |

## Defaults

```python
# Default provider and model (from lionagi.config.settings)
provider = "openai"
model = "gpt-4.1-mini"

# Minimal Branch -- uses defaults above
branch = Branch()

# Explicit provider
branch = Branch(chat_model=iModel(provider="anthropic", model="claude-sonnet-4-20250514"))
```

## Quick Start

```python
from lionagi import Branch, iModel

# Simple conversation (adds to history)
async with Branch() as b:
    response = await b.communicate("Explain dependency injection")

# Structured output
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    confidence: float

result = await b.communicate(
    "Analyze this code for security issues",
    response_format=Analysis,
)
# result is an Analysis instance

# Tool use
def search_docs(query: str) -> str:
    """Search documentation."""
    return f"Results for: {query}"

b.register_tools(search_docs)
result = await b.operate(
    instruction="Find documentation about authentication",
    actions=True,
)
```

## Section Contents

| Page | When to Read |
|------|-------------|
| [Claude Code Usage](claude-code-usage.md) | Using lionagi with CLI-based agent providers |
| [Orchestration Guide](orchestration-guide.md) | Building multi-step workflows |
| [Pattern Selection](pattern-selection.md) | Choosing the right Branch method |
| [Self-Improvement](self-improvement.md) | Inspecting and debugging conversations |

## Provider Support

**API Providers:** openai, anthropic, google (Gemini API), ollama, nvidia, perplexity, groq, openrouter

**CLI Providers (for agent-to-agent):** claude_code, gemini_code, codex

## Key Behaviors

- `Branch` supports `async with` for automatic log cleanup on exit.
- `iModel` supports `async with` for automatic executor shutdown.
- `chat()` does NOT add messages to conversation history. Use it for one-off
  orchestration calls.
- `communicate()` DOES add messages to history. Use it for building conversation
  context.
- `operate()` DOES add messages and can invoke registered tools.
- Default `parse_model` is the same as `chat_model` unless explicitly set.
- CLI endpoints (claude_code, gemini_code, codex) get fresh copies on
  `Branch.clone()` to avoid session conflicts.
