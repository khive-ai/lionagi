# Operations

Operations are the fundamental building blocks of LionAGI workflows. They define specific tasks that Branches execute, control execution order through dependencies, and enable complex orchestration patterns.

Think of Operations as discrete work units: each Operation represents a single task (like analyzing data, writing content, or making decisions) that gets assigned to a specific Branch for execution.

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

This example shows the basic Operation structure: you specify the operation type, provide instructions, assign it to a Branch, and optionally define dependencies that control when the operation runs.

## Core Concepts

- Operations define tasks for branches to execute
- Dependencies control execution order
- Operations without dependencies run in parallel

## Operation Types

LionAGI provides three core operation types, each optimized for different use cases:

### chat: Basic Conversation

Use `chat` for simple, stateless interactions where you need natural language responses:

```python
builder.add_operation("chat", branch=agent, instruction="Explain quantum computing")
```

The `chat` operation is ideal for one-off questions, explanations, and tasks that don't require maintaining conversation context.

### communicate: Stateful Conversation

Use `communicate` when you need the Branch to maintain conversation history and context across operations:

```python
builder.add_operation("communicate", branch=agent, instruction="Continue discussion", context=data)
```

This operation type enables follow-up questions, iterative refinement, and complex dialogues where context matters.

### operate: Structured Output  

Use `operate` when you need predictable, structured responses that conform to specific data models:

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float

builder.add_operation("operate", branch=agent, instruction="Analyze feedback", response_format=Analysis)
```

The `operate` operation ensures consistent output format, making it perfect for data processing, analysis, and integration with downstream systems.

### ReAct: Reasoning and Tool Use

Use `ReAct` when you need the Branch to reason through problems and use tools to gather information or perform actions:

```python
builder.add_operation(
    "ReAct", 
    branch=agent_with_tools,
    instruct={"instruction": "Research AI advances"},
    max_extensions=3
)
```

The ReAct operation enables sophisticated reasoning patterns where the agent can think, act (use tools), and observe results in iterative cycles. The `max_extensions` parameter controls how many reasoning-action cycles are allowed.

## Common Orchestration Patterns

Understanding these fundamental patterns will help you design effective multi-agent workflows:

### Sequential Processing

Use sequential patterns when each step builds on the results of the previous step:

```python
step1 = builder.add_operation("chat", instruction="Extract data")
step2 = builder.add_operation("chat", instruction="Clean data", depends_on=[step1])
step3 = builder.add_operation("chat", instruction="Analyze data", depends_on=[step2])
```

Sequential processing ensures proper data flow and maintains logical dependencies, but executes operations one at a time.

!!! tip "Learn More"
    See [Sequential Analysis Pattern](../patterns/sequential-analysis.md) for complete examples and best practices.

### Fan-Out/Fan-In Pattern

Use this pattern to analyze data from multiple perspectives simultaneously, then synthesize the results:

```python
analysis1 = builder.add_operation("chat", branch=agent1, instruction="Analyze A")
analysis2 = builder.add_operation("chat", branch=agent2, instruction="Analyze B")

# Synthesis depends on both analyses completing
synthesis = builder.add_operation(
    "chat", 
    depends_on=[analysis1, analysis2],
    instruction="Combine analyses"
)
```

This pattern maximizes parallelism while ensuring all perspectives are considered before synthesis.

!!! tip "Learn More"
    See [Fan-Out/In Pattern](../patterns/fan-out-in.md) for production examples and performance characteristics.

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

- Use `chat` for simple tasks, `operate` for structured output, `ReAct` for tool use → See [Tools and Functions](tools-and-functions.md)
- Minimize dependencies to maximize parallelism → See [Performance Guide](../advanced/performance.md)  
- Set appropriate timeouts and retry parameters → See [Error Handling](../advanced/error-handling.md)

## Next Steps

!!! success "Ready to Apply Operations?"
    Now that you understand operations, see them in action:
    
    - [Patterns](../patterns/) - Common workflow patterns using operations
    - [Cookbook](../cookbook/) - Complete examples you can copy and modify
    - [Advanced Operations](../advanced/custom-operations.md) - Build your own operation types
