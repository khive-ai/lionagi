# Flow Composition

`OperationGraphBuilder` (imported as `Builder`) lets you define directed
acyclic graphs of Branch operations. `Session.flow()` executes them with
dependency-aware scheduling and configurable concurrency.

## Core Concepts

A flow is a DAG where each node is an **Operation** (a Branch method call
with parameters) and edges encode execution order. The executor:

1. Topologically sorts the graph.
2. Runs independent operations concurrently (up to `max_concurrent`).
3. Passes predecessor results as `context` to dependent operations.
4. Returns a dict with `completed_operations`, `operation_results`,
   `skipped_operations`, and `final_context`.

## Sequential Flows

Chain operations with `depends_on`:

```python
from lionagi import Session, Builder, Branch, iModel

session = Session()
branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
    system="You are a research analyst.",
)
session.include_branches(branch)

builder = Builder("sequential")

research = builder.add_operation(
    "communicate",
    branch=branch,
    instruction="Research AI market trends for 2025",
)

analysis = builder.add_operation(
    "communicate",
    branch=branch,
    instruction="Analyze the research findings and identify opportunities",
    depends_on=[research],
)

report = builder.add_operation(
    "communicate",
    branch=branch,
    instruction="Write an executive summary with recommendations",
    depends_on=[analysis],
)

result = await session.flow(builder.get_graph())
```

When `depends_on` is omitted, `add_operation` automatically links the new
node to the previous one (sequential by default).

## Parallel Flows

Operations without dependency relationships run concurrently:

```python
builder = Builder("parallel_analysis")

# These three operations have no dependencies on each other
market = builder.add_operation(
    "communicate",
    branch=branch,
    instruction="Analyze market conditions",
)

competitor = builder.add_operation(
    "communicate",
    branch=branch,
    instruction="Analyze top 3 competitors",
    depends_on=[],  # Explicitly no dependencies
)

tech = builder.add_operation(
    "communicate",
    branch=branch,
    instruction="Analyze technology trends",
    depends_on=[],
)

# Aggregate all three into a synthesis
synthesis = builder.add_aggregation(
    "communicate",
    branch=branch,
    source_node_ids=[market, competitor, tech],
    instruction="Synthesize all analyses into a strategic brief",
)

result = await session.flow(builder.get_graph(), max_concurrent=3)
```

## Fan-Out with expand_from_result

After executing a graph, you can expand it dynamically based on results
and continue execution:

```python
from lionagi.operations.builder import ExpansionStrategy

builder = Builder("dynamic_expansion")

# Step 1: generate sub-tasks
generate = builder.add_operation(
    "operate",
    branch=branch,
    instruction="List 3 research questions about renewable energy",
    response_format=ResearchQuestions,  # a Pydantic model
)

# Execute step 1
result = await session.flow(builder.get_graph())

# Step 2: expand -- create one operation per question
questions = result["operation_results"][generate]
if hasattr(questions, "questions"):
    builder.expand_from_result(
        items=questions.questions,
        source_node_id=generate,
        operation="communicate",
        strategy=ExpansionStrategy.CONCURRENT,
        instruction="Answer this research question in detail",
    )

# Step 3: aggregate expanded results
builder.add_aggregation(
    "communicate",
    branch=branch,
    instruction="Combine all answers into a report",
)

# Execute the expanded graph
final = await session.flow(builder.get_graph())
```

`ExpansionStrategy` options:

- `CONCURRENT` -- all expanded operations run in parallel (default).
- `SEQUENTIAL` -- expanded operations run one after another.
- `SEQUENTIAL_CONCURRENT_CHUNK` -- sequential groups of concurrent ops.
- `CONCURRENT_SEQUENTIAL_CHUNK` -- concurrent groups of sequential ops.

## Multi-Branch Flows

Assign different branches (with different models or system prompts) to
different operations:

```python
researcher = Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
    system="You are a thorough researcher.",
)
analyst = Branch(
    chat_model=iModel(provider="anthropic", model="claude-sonnet-4-20250514"),
    system="You are a critical data analyst.",
)
writer = Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
    system="You are a concise report writer.",
)
session.include_branches([researcher, analyst, writer])

builder = Builder("multi_branch")

step1 = builder.add_operation(
    "communicate",
    branch=researcher,
    instruction="Research quantum computing advances",
)

step2 = builder.add_operation(
    "communicate",
    branch=analyst,
    instruction="Critically evaluate the research",
    depends_on=[step1],
)

step3 = builder.add_operation(
    "communicate",
    branch=writer,
    instruction="Write a two-paragraph summary",
    depends_on=[step2],
)

result = await session.flow(builder.get_graph())
```

## Context Inheritance

When `inherit_context=True`, a dependent operation clones the conversation
history from its primary dependency (the first ID in `depends_on`). This
means the downstream Branch sees all the messages from the upstream Branch:

```python
parent = builder.add_operation(
    "communicate",
    branch=branch,
    instruction="Analyze business requirements",
)

child = builder.add_operation(
    "communicate",
    branch=branch,
    instruction="Based on the analysis, suggest an architecture",
    depends_on=[parent],
    inherit_context=True,
)
```

Without `inherit_context`, each operation gets a fresh branch clone.
Predecessor results are still passed as `context` data, but the
conversation history does not carry over.

## Conditional Branching

`add_conditional_branch` creates a condition-check node with true/false
paths:

```python
nodes = builder.add_conditional_branch(
    condition_check_op="communicate",
    true_op="communicate",
    false_op="communicate",
    instruction="Is the dataset large enough for ML? Answer yes or no.",
)
# nodes = {"check": id, "true": id, "false": id}
```

Edge conditions on the graph control which path executes at runtime.

## Controlling Concurrency

`Session.flow()` accepts `max_concurrent` to limit how many operations
run simultaneously:

```python
# Conservative: 2 concurrent API calls
result = await session.flow(builder.get_graph(), max_concurrent=2)

# Aggressive: 10 concurrent calls (watch your rate limits)
result = await session.flow(builder.get_graph(), max_concurrent=10)

# Sequential execution
result = await session.flow(builder.get_graph(), parallel=False)
```

The default is `max_concurrent=5`. Set `parallel=False` to force
`max_concurrent=1`.

## Inspecting Results

`Session.flow()` returns a dict:

```python
result = await session.flow(builder.get_graph(), verbose=True)

print(result["completed_operations"])   # list of operation IDs
print(result["skipped_operations"])     # list of skipped operation IDs
print(result["operation_results"])      # {op_id: response} mapping
print(result["final_context"])          # accumulated context dict
```

Use `verbose=True` during development to see execution order, dependency
waits, and timing.

## Guidelines

- Start with linear flows. Add parallelism only when you have independent
  operations that benefit from concurrent execution.
- Use `add_aggregation` to merge parallel results into a single downstream
  operation.
- Keep `max_concurrent` at or below your API provider's rate limit.
- Use multi-branch flows when operations need different models or system
  prompts, not just different instructions.
