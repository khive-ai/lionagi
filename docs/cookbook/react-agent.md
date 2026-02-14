# ReAct Research Agent

Build agents that reason step-by-step and use tools to gather information
before answering.

## Basic ReAct Agent

```python
import asyncio
from lionagi import Branch, iModel

def web_search(query: str) -> str:
    """Search the web for information."""
    # Replace with actual search API (e.g., SerpAPI, Tavily)
    return f"Search results for '{query}': [mock results]"

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters in expression"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

agent = Branch(
    tools=[web_search, calculate],
    chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
    system="You are a research assistant. Use tools to find facts before answering.",
)

result = asyncio.run(agent.ReAct(
    instruct={"instruction": "What is the GDP per capita of Japan in USD?"},
    max_extensions=3,
    verbose=True,
))
print(result)
```

## Multi-Tool Research Agent

An agent with a toolkit for file reading, search, and computation:

```python
import json
from pathlib import Path
from pydantic import BaseModel, Field

def read_file(path: str) -> str:
    """Read a local file and return its contents."""
    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"
    return p.read_text()[:5000]  # Cap at 5k chars

def list_files(directory: str, pattern: str = "*") -> str:
    """List files in a directory matching a glob pattern."""
    p = Path(directory)
    if not p.is_dir():
        return f"Not a directory: {directory}"
    files = sorted(p.glob(pattern))[:50]
    return json.dumps([str(f) for f in files])

def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

class ResearchReport(BaseModel):
    topic: str
    key_findings: list[str]
    sources_used: list[str]
    confidence: float = Field(ge=0, le=1)
    summary: str

async def research(question: str) -> ResearchReport:
    agent = Branch(
        tools=[read_file, list_files, web_search, calculate],
        chat_model=iModel(provider="openai", model="gpt-4.1"),
        system=(
            "You are a thorough research agent. Always use tools to verify "
            "facts. Do not guess — search or compute instead. Cite which "
            "tools you used for each finding."
        ),
    )

    result = await agent.ReAct(
        instruct={"instruction": question},
        response_format=ResearchReport,
        max_extensions=5,
        verbose=True,
    )
    return result

# Usage
report = asyncio.run(research(
    "Analyze the Python files in ./src and summarize the architecture."
))
print(f"Topic: {report.topic}")
for finding in report.key_findings:
    print(f"  - {finding}")
```

## Streaming ReAct

Process reasoning steps as they happen:

```python
async def stream_research(question: str):
    """Stream intermediate reasoning steps to the user."""
    agent = Branch(
        tools=[web_search, calculate],
        chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
        system="Research assistant. Think step by step.",
    )

    step_count = 0
    async for step in agent.ReActStream(
        instruct={"instruction": question},
        max_extensions=4,
        verbose=True,
    ):
        step_count += 1
        print(f"\n--- Step {step_count} ---")
        print(step)

    print(f"\nCompleted in {step_count} steps")

asyncio.run(stream_research("Compare Python and Rust for CLI tools."))
```

## Domain-Specific Agent with Interpret

Pre-process vague questions into precise research tasks:

```python
async def smart_research(raw_question: str, domain: str = None):
    """Interpret a vague question, then research it systematically."""
    agent = Branch(
        tools=[web_search],
        chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
        system="Precise research agent. Break vague questions into specific sub-queries.",
    )

    result = await agent.ReAct(
        instruct={"instruction": raw_question},
        interpret=True,
        interpret_domain=domain,
        interpret_style="detailed",
        max_extensions=4,
    )
    return result

# Vague input gets refined automatically
answer = asyncio.run(smart_research(
    "how do I make my api faster",
    domain="backend engineering",
))
print(answer)
```

## ReAct with Structured Intermediate Steps

Capture reasoning at each step for audit trails:

```python
class StepAnalysis(BaseModel):
    thought: str
    action_taken: str
    result_summary: str

async def auditable_research(question: str):
    """Research with full reasoning trail."""
    agent = Branch(
        tools=[web_search, calculate],
        chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
        system="Research agent. Record your reasoning at each step.",
    )

    final, analyses = await agent.ReAct(
        instruct={"instruction": question},
        max_extensions=4,
        return_analysis=True,
        verbose=True,
    )

    print("=== Reasoning Trail ===")
    for i, analysis in enumerate(analyses, 1):
        print(f"Step {i}: {analysis}")

    print(f"\n=== Final Answer ===\n{final}")
    return final, analyses

answer, trail = asyncio.run(auditable_research(
    "What is 15% of the US national debt?"
))
```

## When to Use

**Perfect for:** Question answering with fact-checking, code analysis,
data investigation, any task where the agent needs to gather information
before responding.

**Key patterns:**

- Use `ReAct()` for multi-step reasoning with tools
- Use `ReActStream()` to show progress in real-time
- Use `return_analysis=True` for audit trails
- Use `interpret=True` to refine vague user input
- Set `max_extensions` to control reasoning depth (default 3, max 5)
- Provide `response_format` to get structured final output
