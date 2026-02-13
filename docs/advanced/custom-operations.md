# Custom Operations

Branch ships with built-in operations -- `chat`, `communicate`, `operate`,
`parse`, `interpret`, `ReAct`, and `act`. You can extend this set by
registering your own async functions and composing them into graphs.

## Built-in Operations

Every Branch instance exposes these methods directly:

| Operation     | Adds messages? | Tool calling? | Structured output? |
|---------------|:--------------:|:-------------:|:-------------------:|
| `chat`        | No             | No            | No                  |
| `communicate` | Yes            | No            | Optional            |
| `operate`     | Yes            | Yes           | Yes                 |
| `parse`       | No             | No            | Yes                 |
| `interpret`   | No             | No            | No (returns string) |
| `ReAct`       | Yes            | Yes           | Optional            |
| `act`         | Yes            | Yes (only)    | N/A                 |

These are the values accepted by `OperationGraphBuilder.add_operation()`
in its `operation` parameter (see the `BranchOperations` literal type in
`lionagi.operations.node`).

## Registering Custom Operations

### Session.operation() Decorator

The cleanest way to add a custom operation is with the `Session.operation()`
decorator. Registered operations become available to every Branch in the
session, and they can be referenced by name in `OperationGraphBuilder`.

```python
from lionagi import Session, Branch

session = Session()

@session.operation()
async def summarize(branch: Branch, instruction: str, **kwargs):
    """Summarize with a structured two-part prompt."""
    guidance = "Respond with: 1) Key points  2) One-sentence summary"
    return await branch.communicate(
        instruction=instruction,
        guidance=guidance,
        **kwargs,
    )
```

The function name (`summarize`) becomes the operation name. To use a
different name, pass it as an argument:

```python
@session.operation("custom_summary")
async def my_summary_func(branch: Branch, instruction: str, **kwargs):
    ...
```

### Session.register_operation()

If you prefer not to use decorators:

```python
async def extract_entities(branch: Branch, text: str, **kwargs):
    from pydantic import BaseModel

    class Entities(BaseModel):
        people: list[str]
        organizations: list[str]

    return await branch.operate(
        instruction=f"Extract entities from: {text}",
        response_format=Entities,
        **kwargs,
    )

session.register_operation("extract_entities", extract_entities)
```

### Requirements

Custom operations must be **async functions**. Synchronous functions will
raise `ValueError` at registration time.

The first positional argument receives the `Branch` instance. Any remaining
keyword arguments come from the `parameters` dict you pass to
`OperationGraphBuilder.add_operation()`.

## Using Custom Operations in Graphs

Once registered, custom operations work exactly like built-in ones in
`OperationGraphBuilder`:

```python
from lionagi import Builder, Session, Branch, iModel

session = Session()
branch = Branch(chat_model=iModel(provider="openai", model="gpt-4.1-mini"))
session.include_branches(branch)

@session.operation()
async def research(branch: Branch, topic: str, **kwargs):
    return await branch.communicate(
        instruction=f"Research: {topic}",
        guidance="Be thorough and cite sources",
    )

@session.operation()
async def critique(branch: Branch, instruction: str, **kwargs):
    return await branch.communicate(
        instruction=instruction,
        guidance="Identify weaknesses and gaps",
    )

builder = Builder("research_pipeline")

research_id = builder.add_operation(
    "research",
    branch=branch,
    topic="Transformer architecture efficiency",
)

critique_id = builder.add_operation(
    "critique",
    branch=branch,
    instruction="Critique the research above",
    depends_on=[research_id],
)

result = await session.flow(builder.get_graph(), max_concurrent=2)
```

## Wrapping External APIs

A common pattern is wrapping external services as custom operations so
they participate in graph-based workflows:

```python
import httpx

@session.operation()
async def fetch_data(branch: Branch, url: str, **kwargs):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()

    # Feed external data into the LLM as context
    return await branch.communicate(
        instruction="Summarize this data",
        context=data,
    )
```

## Composing Built-in and Custom Operations

Custom operations mix freely with built-in ones in the same graph:

```python
builder = Builder("mixed_pipeline")

# Built-in operation
step1 = builder.add_operation(
    "communicate",
    branch=branch,
    instruction="List 5 research questions about climate change",
)

# Custom operation depending on step1
step2 = builder.add_operation(
    "research",
    branch=branch,
    topic="Use the questions above",
    depends_on=[step1],
)

# Built-in aggregation
step3 = builder.add_aggregation(
    "communicate",
    branch=branch,
    source_node_ids=[step2],
    instruction="Write an executive summary",
)

result = await session.flow(builder.get_graph())
```

## Guidelines

- Prefer composing built-in operations (`communicate`, `operate`) inside
  custom operations rather than reimplementing prompt construction.
- Keep custom operations focused on a single responsibility.
- Test operations independently with a Branch before wiring them into graphs.
- Remember that `chat` does **not** add messages to history -- use
  `communicate` or `operate` when you need conversational context to
  accumulate.
