# Orchestration Guide

How to build single-step and multi-step workflows with lionagi.

## Architecture Overview

```
Branch (single conversation)
  |- chat_model: iModel      -- LLM for conversation
  |- parse_model: iModel      -- LLM for structured parsing (defaults to chat_model)
  |- messages: Pile[RoledMessage]  -- conversation history
  |- tools: dict[str, Tool]   -- registered callable tools
  |- logs: Pile[Log]          -- activity logs

Session (multi-branch orchestrator)
  |- branches: Pile[Branch]   -- all managed branches
  |- default_branch: Branch   -- fallback branch
  |- flow(graph) -> dict      -- execute a DAG workflow

Builder (OperationGraphBuilder)
  |- add_operation(...)       -- add a node to the DAG
  |- add_aggregation(...)     -- add a node that collects from multiple sources
  |- expand_from_result(...)  -- dynamically expand based on results
  |- get_graph() -> Graph     -- return the graph for execution
```

## Branch: The Primary API Surface

Branch is a facade over four managers. All LLM operations are Branch methods.

### Method Signatures

```python
# Low-level call -- does NOT add to history
await branch.chat(
    instruction="...",          # main prompt
    guidance="...",             # system-level guidance
    context="...",              # additional context
    response_format=MyModel,   # optional structured output
    imodel=alt_model,          # optional model override
    images=[...],              # optional images
    **kwargs,                  # passed to LLM (temperature, etc.)
) -> tuple[Instruction, AssistantResponse]
# Returns (instruction_msg, response_msg) -- neither added to history

# Conversational -- DOES add to history
await branch.communicate(
    instruction="...",          # main prompt (positional OK)
    guidance="...",             # system-level guidance
    context="...",              # additional context
    response_format=MyModel,   # optional structured output
    chat_model=alt_model,      # optional model override
    parse_model=parse_model,   # optional parse model override
    num_parse_retries=3,       # retries for structured parsing
    clear_messages=False,      # clear history before this call
    **kwargs,
) -> str | BaseModel | dict | None

# Tool use + structured output -- DOES add to history
await branch.operate(
    instruction="...",          # or instruct=Instruct(...)
    guidance="...",
    context="...",
    response_format=MyModel,   # optional structured output
    actions=True,              # enable tool calling
    tools=[my_func],           # register tools for this call
    reason=True,               # request chain-of-thought
    invoke_actions=True,       # auto-invoke requested tools
    action_strategy="concurrent",  # or "sequential"
    **kwargs,
) -> list | BaseModel | None | dict | str

# Multi-step reasoning -- DOES add to history
await branch.ReAct(
    instruct=Instruct(instruction="...", guidance="..."),
    tools=[my_func],           # available tools
    response_format=MyModel,   # final output format
    extension_allowed=True,    # allow multi-step expansion
    max_extensions=3,          # max reasoning steps (capped at 5)
    verbose=False,             # print intermediate steps
    **kwargs,
) -> Any | tuple[Any, list]   # result, or (result, analyses) if return_analysis=True
```

### When to Use Which Method

```
Need simple LLM response, no history?          -> chat()
Need conversational context?                   -> communicate()
Need structured output from conversation?      -> communicate(response_format=Model)
Need tool calling?                             -> operate(actions=True)
Need structured output + tools?               -> operate(response_format=Model, actions=True)
Need multi-step reasoning with tools?          -> ReAct()
Need to parse existing text into a model?      -> parse(text, response_format=Model)
Need to refine/rewrite a prompt?               -> interpret(text)
```

## Concurrency Patterns

lionagi provides structured concurrency primitives in `lionagi.ln.concurrency`.

### gather -- Run awaitables concurrently

```python
from lionagi.ln.concurrency import gather

results = await gather(
    branch.communicate("Task A"),
    branch.communicate("Task B"),
    branch.communicate("Task C"),
    return_exceptions=False,  # True to collect errors instead of failing fast
)
# results: list in same order as input
```

### race -- First to complete wins

```python
from lionagi.ln.concurrency import race

result = await race(
    branch.communicate("Fast approach"),
    branch.communicate("Thorough approach"),
)
# Returns the first result; cancels the rest
```

### bounded_map -- Concurrency-limited parallel map

```python
from lionagi.ln.concurrency import bounded_map

tasks = ["Analyze module A", "Analyze module B", "Analyze module C"]
results = await bounded_map(
    lambda task: branch.communicate(task),
    tasks,
    limit=2,  # max 2 concurrent
)
```

### retry -- Exponential backoff

```python
from lionagi.ln.concurrency import retry

result = await retry(
    lambda: branch.communicate("Flaky task"),
    attempts=3,
    base_delay=0.5,
    retry_on=(ValueError,),
)
```

### CompletionStream -- Process results as they arrive

```python
from lionagi.ln.concurrency import CompletionStream

tasks = [branch.communicate(f"Task {i}") for i in range(10)]
async with CompletionStream(tasks, limit=3) as stream:
    async for idx, result in stream:
        print(f"Task {idx} completed: {result[:50]}")
```

### alcall / bcall -- Batch processing

```python
from lionagi.ln import alcall, bcall

# Apply function to list with concurrency control
results = await alcall(
    ["item1", "item2", "item3"],
    lambda x: branch.communicate(f"Process {x}"),
    max_concurrent=2,
    retry_default=3,
)

# Batch processing with yields per batch
async for batch_results in bcall(
    large_list,
    process_fn,
    batch_size=10,
):
    handle(batch_results)
```

## Session + Builder: DAG Workflows

For workflows with dependencies between operations, use Session + Builder.

### Basic Sequential Pipeline

```python
from lionagi import Branch, Session, Builder, iModel

session = Session()
builder = Builder("pipeline")

branch = Branch(chat_model=iModel(provider="openai", model="gpt-4.1-mini"))
session.include_branches([branch])

step1 = builder.add_operation(
    "communicate", branch=branch,
    instruction="Research the topic",
)

step2 = builder.add_operation(
    "communicate", branch=branch,
    instruction="Analyze findings",
    depends_on=[step1],
)

step3 = builder.add_operation(
    "communicate", branch=branch,
    instruction="Write report",
    depends_on=[step2],
)

result = await session.flow(builder.get_graph())
# result["operation_results"][step1]  -- result from step 1
# result["operation_results"][step2]  -- result from step 2
# result["completed_operations"]      -- list of completed node IDs
# result["skipped_operations"]        -- list of skipped node IDs
```

### Fan-Out / Fan-In

```python
builder = Builder("fan_out")

# Independent analyses run in parallel
security = builder.add_operation(
    "communicate", branch=sec_branch,
    instruction="Security analysis",
)
perf = builder.add_operation(
    "communicate", branch=perf_branch,
    instruction="Performance analysis",
)
style = builder.add_operation(
    "communicate", branch=style_branch,
    instruction="Code style analysis",
)

# Aggregation collects all results
synthesis = builder.add_aggregation(
    "communicate", branch=main_branch,
    source_node_ids=[security, perf, style],
    instruction="Synthesize all analyses into a report",
)

result = await session.flow(builder.get_graph(), max_concurrent=3)
```

### Dynamic Expansion

```python
from lionagi.operations.fields import LIST_INSTRUCT_FIELD_MODEL

# Step 1: Generate sub-tasks
root = builder.add_operation(
    "operate", branch=branch,
    instruction="Generate 5 research questions",
    field_models=[LIST_INSTRUCT_FIELD_MODEL],
)

result = await session.flow(builder.get_graph())
sub_tasks = result["operation_results"][root].instruct_models

# Step 2: Expand graph dynamically
new_ids = builder.expand_from_result(
    sub_tasks,
    source_node_id=root,
    operation="communicate",
)

# Step 3: Execute expanded graph
final = await session.flow(builder.get_graph(), max_concurrent=5)
```

### Builder API Reference

```python
builder = Builder(name="workflow_name")

# Add a single operation
node_id = builder.add_operation(
    operation,          # "communicate" | "operate" | "chat" | "ReAct"
    node_id=None,       # optional reference ID
    depends_on=None,    # list of node IDs this depends on
    inherit_context=False,  # inherit conversation from dependency
    branch=None,        # Branch to execute on
    **parameters,       # passed to the Branch method
)

# Add aggregation node
node_id = builder.add_aggregation(
    operation,
    source_node_ids=None,  # defaults to current head nodes
    inherit_context=False,
    branch=None,
    **parameters,
)

# Expand based on results
new_ids = builder.expand_from_result(
    items,              # list of items (e.g., Instruct models)
    source_node_id,     # parent node ID
    operation,          # operation for each item
    strategy=ExpansionStrategy.CONCURRENT,
    **shared_params,
)

# Get the graph for execution
graph = builder.get_graph()

# Inspect state
state = builder.visualize_state()
# {"total_nodes": N, "executed_nodes": M, "current_heads": [...]}
```

### Session.flow() Parameters

```python
result = await session.flow(
    graph,                  # Graph from builder.get_graph()
    context=None,           # dict of initial context
    parallel=True,          # enable parallel execution
    max_concurrent=5,       # max concurrent operations
    verbose=False,          # enable logging
    default_branch=None,    # override default branch
)
```

## Error Handling

### Branch-Level

```python
try:
    result = await branch.communicate("Task")
except ValueError as e:
    # API call failure, timeout, or parsing error
    print(f"Failed: {e}")
```

### Workflow-Level

```python
result = await session.flow(builder.get_graph())

# Check for skipped operations
if result.get("skipped_operations"):
    for skipped_id in result["skipped_operations"]:
        print(f"Skipped: {skipped_id}")

# Check specific operation results
for node_id, op_result in result.get("operation_results", {}).items():
    if op_result is None:
        print(f"Operation {node_id} returned None")
```

### Structured Output Validation

```python
# communicate() with response_format handles parse retries internally
result = await branch.communicate(
    "Extract entities",
    response_format=EntityList,
    num_parse_retries=3,  # retry parsing up to 3 times
)
# result is EntityList or None if all retries fail

# operate() with handle_validation control
result = await branch.operate(
    instruction="Extract entities",
    response_format=EntityList,
    handle_validation="return_none",  # "raise" | "return_value" | "return_none"
)
```

## Registering Tools

```python
# Function tools -- auto-generates schema from signature
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

async def fetch_data(url: str) -> str:
    """Fetch data from URL."""
    ...

branch.register_tools([multiply, fetch_data])

# Then use with operate()
result = await branch.operate(
    instruction="What is 6 times 7?",
    actions=True,
)
```

## System Messages

```python
# Set at construction
branch = Branch(system="You are a code reviewer.")

# With datetime
branch = Branch(
    system="You are a code reviewer.",
    system_datetime=True,  # includes current timestamp
)

# With the built-in Lion system message
branch = Branch(
    system="Additional instructions here",
    use_lion_system_message=True,
)
```
