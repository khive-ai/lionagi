# Operations

Operations are the building blocks of LionAGI workflows - tasks that branches execute.

```python
from lionagi import Builder

builder = Builder("workflow")

op = builder.add_operation(
    "chat",
    instruction="Analyze this data",
    branch=analyst_branch,
    depends_on=[previous_op]
)
```

## Core Concepts

- Operations define tasks for branches to execute
- Dependencies control execution order
- Operations without dependencies run in parallel

## Operation Types

**chat**: Basic conversation

```python
builder.add_operation("chat", branch=agent, instruction="Explain quantum computing")
```

**communicate**: Stateful conversation with context

```python
builder.add_operation("communicate", branch=agent, instruction="Continue discussion", context=data)
```

**operate**: Structured output with Pydantic models

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float

builder.add_operation("operate", branch=agent, instruction="Analyze feedback", response_format=Analysis)
```

**ReAct**: Reasoning and tool use

```python
builder.add_operation(
    "ReAct", 
    branch=agent_with_tools,
    instruct={"instruction": "Research AI advances"},
    max_extensions=3
)
```

## Common Patterns

**Sequential**: Each step depends on the previous

```python
step1 = builder.add_operation("chat", instruction="Extract data")
step2 = builder.add_operation("chat", instruction="Clean data", depends_on=[step1])
step3 = builder.add_operation("chat", instruction="Analyze data", depends_on=[step2])
```

**Parallel**: No dependencies means simultaneous execution

```python
analysis1 = builder.add_operation("chat", branch=agent1, instruction="Analyze A")
analysis2 = builder.add_operation("chat", branch=agent2, instruction="Analyze B")

# Synthesis depends on both
synthesis = builder.add_operation(
    "chat", 
    depends_on=[analysis1, analysis2],
    instruction="Combine analyses"
)
```

## Execution

```python
# Build and execute workflow
graph = builder.get_graph()
results = await session.flow(graph)

# Access operation results
result = results["operation_results"][op.id]
```

## Key Parameters

```python
builder.add_operation(
    "chat",
    branch=agent,              # Which branch executes
    instruction="Task",        # What to do
    depends_on=[],            # Dependencies
    timeout_seconds=300,      # Optional timeout
    retry_times=3             # Retry on failure
)
```

## Best Practices

- Use `chat` for simple tasks, `operate` for structured output, `ReAct` for tool use
- Minimize dependencies to maximize parallelism
- Set appropriate timeouts and retry parameters