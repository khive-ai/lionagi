# Quick Start

Four examples, from simplest to most powerful. Each is complete and copy-pasteable.

Prerequisites: `pip install lionagi` and `OPENAI_API_KEY` set. See [Installation](installation.md) if you haven't done this yet.

---

## 1. Simple Conversation (`communicate`)

`communicate()` sends a message, adds it and the response to the conversation history, and returns the response text.

```python
import asyncio
from lionagi import Branch

async def main():
    branch = Branch(system="You are a concise assistant. Answer in one sentence.")

    # First message
    answer = await branch.communicate("What causes rainbows?")
    print(answer)

    # Follow-up -- the branch remembers the conversation
    followup = await branch.communicate("How long do they typically last?")
    print(followup)

asyncio.run(main())
```

**What happened**: Branch created a default iModel (OpenAI `gpt-4.1-mini`), sent your message with the system prompt, and stored both sides of the conversation. The follow-up question has full context of the previous exchange.

!!! note "`chat()` vs `communicate()`"
    `chat()` does **not** add messages to the conversation history. It is a lower-level method for orchestration where you manage message flow yourself. For most use cases, use `communicate()`.

---

## 2. Structured Output (`response_format`)

Pass a Pydantic model as `response_format` to get typed, validated output instead of raw text.

```python
import asyncio
from pydantic import BaseModel, Field
from lionagi import Branch

class MovieReview(BaseModel):
    title: str
    year: int
    rating: float = Field(ge=0, le=10)
    summary: str
    pros: list[str]
    cons: list[str]

async def main():
    branch = Branch(system="You are a film critic.")
    review = await branch.communicate(
        "Review the movie Inception.",
        response_format=MovieReview,
    )

    # review is a MovieReview instance, not a string
    print(f"{review.title} ({review.year}): {review.rating}/10")
    print(f"Summary: {review.summary}")
    for pro in review.pros:
        print(f"  + {pro}")
    for con in review.cons:
        print(f"  - {con}")

asyncio.run(main())
```

**What happened**: LionAGI sent the Pydantic schema to the LLM as a response format constraint, then validated and parsed the response. If parsing fails, it retries with the parse model (up to 3 times by default). The return value is a `MovieReview` instance with full type safety.

---

## 3. Tool Calling (`operate`)

Register Python functions as tools. `operate()` lets the LLM call them and incorporates the results into its response.

```python
import asyncio
from lionagi import Branch

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        result = eval(expression)  # In production, use a safe evaluator
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def lookup_constant(name: str) -> str:
    """Look up a mathematical or physical constant by name."""
    constants = {
        "pi": "3.14159265358979",
        "e": "2.71828182845905",
        "c": "299792458 m/s",
        "golden_ratio": "1.61803398874989",
    }
    return constants.get(name.lower(), f"Unknown constant: {name}")

async def main():
    branch = Branch(
        system="You are a math assistant. Use the provided tools to compute answers.",
        tools=[calculate, lookup_constant],
    )

    result = await branch.operate(
        instruction="What is pi squared plus the golden ratio?",
        actions=True,
    )
    print(result)

asyncio.run(main())
```

**What happened**: `operate()` sent the instruction along with auto-generated JSON schemas for `calculate` and `lookup_constant`. The LLM decided which tools to call and with what arguments. LionAGI executed the function calls, fed the results back, and returned the final answer. All messages (instruction, tool calls, tool results, final response) are added to the conversation.

!!! tip "Tool schema generation"
    LionAGI generates OpenAI-compatible function schemas from your Python function signatures and docstrings automatically. Type hints and docstrings improve schema quality.

---

## 4. Graph Workflows (`Session` + `Builder`)

For multi-step workflows with dependencies between operations, use `Session` with `Builder` to define and execute a DAG.

```python
import asyncio
from pydantic import BaseModel, Field
from lionagi import Session, Builder

class StartupIdeas(BaseModel):
    ideas: list[str] = Field(description="List of startup ideas")

class Evaluation(BaseModel):
    best_idea: str
    reasoning: str
    market_size: str
    main_risk: str

async def main():
    session = Session()
    builder = Builder()

    # Step 1: Generate ideas
    step1 = builder.add_operation(
        "communicate",
        instruction="Generate 5 AI startup ideas focused on healthcare.",
        response_format=StartupIdeas,
    )

    # Step 2: Evaluate ideas (depends on step 1)
    step2 = builder.add_operation(
        "communicate",
        instruction=(
            "Evaluate the ideas from the previous step. "
            "Pick the best one and explain why."
        ),
        response_format=Evaluation,
        depends_on=[step1],
    )

    # Execute the workflow
    results = await session.flow(builder.get_graph())
    print(results)

asyncio.run(main())
```

**What happened**: `Builder` created two operation nodes in a directed graph, with step 2 depending on step 1. `session.flow()` executed the graph, running step 1 first and passing its context to step 2. Independent nodes (none in this example) execute in parallel automatically.

---

## Choosing the Right Method

| Method | Adds to history? | Tool calling? | Use when |
| --- | --- | --- | --- |
| `chat()` | No | No | Low-level orchestration; you manage messages yourself |
| `communicate()` | Yes | No | Simple conversations and structured output |
| `operate()` | Yes | Yes | LLM needs to call functions/tools |
| `parse()` | No | No | Parse arbitrary text into a Pydantic model (no LLM conversation) |
| `ReAct()` | Yes | Yes | Multi-step reasoning with tool use (think-act-observe loops) |

---

## Using a Different Provider

Every example above works with any provider. Just pass an `iModel`:

```python
from lionagi import Branch, iModel

# Anthropic
branch = Branch(
    chat_model=iModel(provider="anthropic", model="claude-sonnet-4-20250514"),
    system="You are a helpful assistant.",
)

# Gemini
branch = Branch(
    chat_model=iModel(provider="gemini", model="gemini-2.5-flash"),
    system="You are a helpful assistant.",
)

# Local Ollama
branch = Branch(
    chat_model=iModel(provider="ollama", model="llama3.2"),
    system="You are a helpful assistant.",
)
```

---

## Next Steps

- [Core Concepts](../core-concepts/index.md) -- understand Branch, Session, and iModel in depth
- [Operations](../core-concepts/operations.md) -- detailed guide to `chat`, `communicate`, `operate`, `parse`, and `ReAct`
- [Patterns](../patterns/index.md) -- production workflow patterns (fan-out/in, sequential analysis, tournaments)
- [LLM Providers](../integrations/llm-providers.md) -- provider-specific configuration and features
