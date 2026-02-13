# Pattern Selection

Decision trees and lookup tables for choosing the right lionagi API for your
task.

## Primary Decision Tree

```
What do you need?
|
+-- Single LLM call, no conversation state needed?
|   +-- chat()
|
+-- Conversation with history tracking?
|   |
|   +-- Plain text response?
|   |   +-- communicate(instruction)
|   |
|   +-- Structured output (Pydantic model)?
|   |   +-- communicate(instruction, response_format=Model)
|   |
|   +-- Tool calling needed?
|   |   |
|   |   +-- Single-step tool use?
|   |   |   +-- operate(instruction, actions=True)
|   |   |
|   |   +-- Multi-step reasoning + tools?
|   |       +-- ReAct(instruct)
|   |
|   +-- Structured output + tool calling?
|       +-- operate(instruction, response_format=Model, actions=True)
|
+-- Parse existing text into a model?
|   +-- parse(text, response_format=Model)
|
+-- Rewrite/improve a prompt?
|   +-- interpret(text)
|
+-- Multiple operations with dependencies?
|   +-- Session + Builder
|
+-- Multiple independent operations in parallel?
    +-- gather() or asyncio.gather()
```

## Method Lookup Table

| Scenario | Method | Key Parameters |
|----------|--------|---------------|
| Ask a question, get a string | `communicate(instruction)` | -- |
| Ask a question, get structured data | `communicate(instruction, response_format=Model)` | `num_parse_retries=3` |
| Call tools based on instruction | `operate(instruction, actions=True)` | `tools=[fn]`, `action_strategy="concurrent"` |
| Tools + structured output | `operate(instruction, actions=True, response_format=Model)` | `reason=True` |
| Multi-step reasoning | `ReAct(instruct)` | `max_extensions=3`, `extension_allowed=True` |
| Extract data from raw text | `parse(text, response_format=Model)` | `max_retries=3`, `fuzzy_match=True` |
| Refine a prompt | `interpret(text)` | `domain="...", style="..."` |
| Orchestration call (no history) | `chat(instruction)` | `response_format=Model` |

## Concurrency Decision Tree

```
How many operations?
|
+-- Just one?
|   +-- Use the method directly: await branch.communicate(...)
|
+-- Multiple, all independent?
|   |
|   +-- Small number (2-5)?
|   |   +-- gather(branch.communicate("A"), branch.communicate("B"))
|   |
|   +-- Large number, need concurrency limit?
|   |   +-- bounded_map(fn, items, limit=N)
|   |
|   +-- Want results as they complete?
|   |   +-- CompletionStream(tasks, limit=N)
|   |
|   +-- Want only the fastest result?
|       +-- race(branch.communicate("A"), branch.communicate("B"))
|
+-- Operations have dependencies?
|   +-- Session + Builder with depends_on
|
+-- Need retry on failure?
    +-- retry(fn, attempts=3, retry_on=(ValueError,))
```

## Workflow Complexity Decision

```
Is the task a single prompt?
|
+-- Yes -> Branch method directly
|
+-- No -> Are there dependencies between steps?
    |
    +-- No -> gather() or bounded_map()
    |
    +-- Yes -> Do results of one step determine what steps come next?
        |
        +-- No -> Builder with static depends_on
        |
        +-- Yes -> Builder + expand_from_result() (dynamic expansion)
```

## Structured Output Patterns

### Pattern 1: Simple extraction from conversation

```python
from pydantic import BaseModel

class Entities(BaseModel):
    people: list[str]
    organizations: list[str]

result = await branch.communicate(
    "Extract all entities from this text: ...",
    response_format=Entities,
)
# result: Entities(people=[...], organizations=[...])
```

### Pattern 2: Tool use with structured final output

```python
class Report(BaseModel):
    findings: list[str]
    recommendation: str

branch.register_tools([search_docs, check_status])
result = await branch.operate(
    instruction="Investigate the issue and write a report",
    actions=True,
    response_format=Report,
    reason=True,  # include chain-of-thought
)
# result: Report(findings=[...], recommendation="...")
```

### Pattern 3: Multi-step reasoning with structured output

```python
from lionagi.operations.fields import Instruct

result = await branch.ReAct(
    instruct=Instruct(
        instruction="Research and analyze the market opportunity",
        guidance="Use available tools to gather data before concluding",
    ),
    tools=[search_market_data, get_competitors],
    response_format=MarketAnalysis,
    max_extensions=3,
)
```

### Pattern 4: Parse existing text (no LLM conversation)

```python
raw_text = """
Name: John Smith
Role: Engineer
Department: Backend
"""

result = await branch.parse(
    raw_text,
    response_format=Employee,
    fuzzy_match=True,        # handle approximate field names
    max_retries=3,
)
```

## Branch Configuration Patterns

### Single-purpose branch

```python
reviewer = Branch(
    system="You are a senior code reviewer. Focus on correctness and security.",
    chat_model=iModel(provider="anthropic", model="claude-sonnet-4-20250514"),
)
```

### Multi-model branch (different models for chat vs parse)

```python
branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1"),
    parse_model=iModel(provider="openai", model="gpt-4.1-mini"),
)
```

### Branch with tools pre-registered

```python
branch = Branch(
    system="You have access to search and file tools.",
    tools=[search_docs, read_file, write_file],
)
result = await branch.operate(
    instruction="Find and fix the bug in auth.py",
    actions=True,
)
```

## Common Anti-Patterns

### Using chat() when you need history

```python
# Wrong: chat() does not add to history, second call lacks context
await branch.chat("What is X?")
await branch.chat("Tell me more about it")  # "it" has no referent

# Right: communicate() maintains conversation
await branch.communicate("What is X?")
await branch.communicate("Tell me more about it")  # sees previous exchange
```

### Using operate() when communicate() suffices

```python
# Wrong: operate() overhead when no tools are needed
await branch.operate(instruction="Summarize this text")

# Right: communicate() is simpler when no tools/actions are involved
await branch.communicate("Summarize this text")
```

### Sequential when parallel is possible

```python
# Wrong: sequential independent tasks
r1 = await branch.communicate("Analyze security")
r2 = await branch.communicate("Analyze performance")

# Right: parallel independent tasks
from lionagi.ln.concurrency import gather
r1, r2 = await gather(
    branch.communicate("Analyze security"),
    branch.communicate("Analyze performance"),
)
```

### Over-engineering with Builder

```python
# Wrong: Builder for a single operation
builder = Builder("simple")
op = builder.add_operation("communicate", branch=branch, instruction="Hello")
result = await session.flow(builder.get_graph())

# Right: direct call
result = await branch.communicate("Hello")
```

## Quick Reference Card

```python
from lionagi import Branch, Session, Builder, iModel
from lionagi.ln.concurrency import gather, race, bounded_map, retry, CompletionStream
from lionagi.operations.fields import Instruct, LIST_INSTRUCT_FIELD_MODEL
from pydantic import BaseModel

# -- Setup --
branch = Branch(chat_model=iModel(provider="openai", model="gpt-4.1-mini"))

# -- Single operations --
text     = await branch.communicate("question")
model    = await branch.communicate("question", response_format=MyModel)
tool_res = await branch.operate(instruction="do X", actions=True)
react_res = await branch.ReAct(Instruct(instruction="solve X"), tools=[fn])

# -- Parallel --
results = await gather(branch.communicate("A"), branch.communicate("B"))
fastest = await race(branch.communicate("A"), branch.communicate("B"))
mapped  = await bounded_map(lambda x: branch.communicate(x), items, limit=3)

# -- DAG workflow --
session = Session()
builder = Builder("workflow")
session.include_branches([branch])
s1 = builder.add_operation("communicate", branch=branch, instruction="step 1")
s2 = builder.add_operation("communicate", branch=branch, instruction="step 2",
                           depends_on=[s1])
result = await session.flow(builder.get_graph())
```
