# Observability

lionagi provides several mechanisms for inspecting runtime behavior:
`DataLogger` for activity logs, `HookRegistry` for aspect-oriented hooks
on iModel calls, message inspection on Branch, and verbose mode in
`Session.flow()`.

## DataLogger and Log

Every Branch has a `DataLogger` that stores `Log` entries -- immutable
snapshots of events (API calls, tool invocations, etc.).

### Accessing Logs

```python
from lionagi import Branch, iModel

branch = Branch(chat_model=iModel(provider="openai", model="gpt-4.1-mini"))
await branch.communicate("Explain quantum computing")

# branch.logs is a Pile[Log]
print(f"Log entries: {len(branch.logs)}")
for log in branch.logs:
    print(log.content.keys())
```

### Configuring the Logger

```python
from lionagi.protocols.generic import DataLoggerConfig

config = DataLoggerConfig(
    persist_dir="./data/logs",       # Where log files are saved
    subfolder="experiment_01",       # Subdirectory within persist_dir
    file_prefix="run",              # Filename prefix
    capacity=100,                    # Auto-dump after 100 entries
    extension=".json",               # .json or .csv
    use_timestamp=True,              # Include timestamp in filename
    hash_digits=5,                   # Random hash in filename
    auto_save_on_exit=True,          # Dump remaining logs at exit
    clear_after_dump=True,           # Clear in-memory logs after dump
)

branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
    log_config=config,
)
```

When `capacity` is set, the logger automatically dumps to disk once
that many entries accumulate. When `auto_save_on_exit=True` (the
default), remaining logs are saved when the Python process exits.

### Manual Dump

```python
# Synchronous dump
branch.dump_logs(clear=True, persist_path="./my_logs.json")

# Asynchronous dump
await branch.adump_logs(clear=True)
```

### Branch as Context Manager

Using `async with` on a Branch automatically dumps logs on exit:

```python
async with Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1-mini")
) as branch:
    await branch.communicate("Hello")
    # Logs are dumped when exiting the context
```

## Message Inspection

`branch.messages` is a `Pile[RoledMessage]` containing the full
conversation history. Use it for debugging, analysis, or export.

### Inspecting Messages

```python
branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
    system="You are a research assistant.",
)

await branch.communicate("List 3 AI trends")
await branch.communicate("Elaborate on the first trend")

# Total message count (system + instruction + response pairs)
print(f"Messages: {len(branch.messages)}")

# System message
if branch.system:
    print(f"System: {branch.system.content[:80]}...")

# Iterate messages
for msg in branch.messages:
    role = msg.role.value if hasattr(msg.role, "value") else msg.role
    content = str(msg.content)[:100]
    print(f"[{role}] {content}...")
```

### Exporting to DataFrame

```python
df = branch.to_df()
print(df[["role", "content"]].head())
```

### Clearing History

```python
# Clear all messages (keeps system message intact via MessageManager)
branch.messages.clear()
```

## HookRegistry

`HookRegistry` provides aspect-oriented hooks on iModel API calls. You
can intercept events at three points:

1. **PreEventCreate** -- before the API call event is constructed.
2. **PreInvocation** -- after the event is queued but before the HTTP
   request is sent.
3. **PostInvocation** -- after the HTTP response is received.

### Registering Hooks

```python
from lionagi import iModel, HookRegistry
from lionagi.service.hooks import HookEventTypes

async def log_before_call(event, **kwargs):
    """Called before each API request."""
    print(f"About to call API: {type(event).__name__}")
    # Return None to proceed normally
    return None

async def log_after_call(event, **kwargs):
    """Called after each API response."""
    print(f"API call completed: {event.execution.status}")
    return None

registry = HookRegistry(
    hooks={
        HookEventTypes.PreInvocation: log_before_call,
        HookEventTypes.PostInvocation: log_after_call,
    }
)

model = iModel(
    provider="openai",
    model="gpt-4.1-mini",
    hook_registry=registry,
)
```

### Exit Hooks

When `exit_hook=True` on iModel, a hook can abort the API call by
raising an exception. The exception is captured and the event is
marked as cancelled:

```python
async def permission_check(event, **kwargs):
    """Block calls that exceed a budget."""
    if over_budget():
        raise RuntimeError("API budget exceeded")
    return None

model = iModel(
    provider="openai",
    model="gpt-4.1-mini",
    hook_registry=HookRegistry(
        hooks={HookEventTypes.PreInvocation: permission_check}
    ),
    exit_hook=True,
)
```

### Stream Handlers

For streaming responses, register handlers by chunk type:

```python
async def handle_chunk(event, chunk_type, chunk, **kwargs):
    print(f"Received chunk: {chunk}")

registry = HookRegistry(
    stream_handlers={"text": handle_chunk}
)
```

## Flow Verbose Mode

`Session.flow()` accepts `verbose=True` to print execution details:

```python
from lionagi import Session, Builder, Branch, iModel

session = Session()
branch = Branch(chat_model=iModel(provider="openai", model="gpt-4.1-mini"))
session.include_branches(branch)

builder = Builder("debug_flow")
step1 = builder.add_operation(
    "communicate", branch=branch,
    instruction="Research topic A",
)
step2 = builder.add_operation(
    "communicate", branch=branch,
    instruction="Analyze findings",
    depends_on=[step1],
)

result = await session.flow(builder.get_graph(), verbose=True)
```

With `verbose=True`, the executor prints:

- When each operation starts executing.
- Dependency wait events (which operation is waiting for which).
- Completion and failure events with operation IDs.
- Context inheritance actions.
- Pre-allocation of branches.

### Result Inspection

```python
result = await session.flow(builder.get_graph())

print(f"Completed: {len(result['completed_operations'])}")
print(f"Skipped: {len(result['skipped_operations'])}")

# Check individual operation results
for op_id, response in result["operation_results"].items():
    if isinstance(response, dict) and "error" in response:
        print(f"FAILED {str(op_id)[:8]}: {response['error']}")
    else:
        print(f"OK {str(op_id)[:8]}: {str(response)[:60]}...")
```

## Graph Visualization

`OperationGraphBuilder` provides state inspection and visualization:

```python
# Text-based state summary
state = builder.visualize_state()
print(f"Total nodes: {state['total_nodes']}")
print(f"Executed: {state['executed_nodes']}")
print(f"Edges: {state['edges']}")

# Matplotlib visualization (requires matplotlib and networkx)
builder.visualize(title="My Workflow", figsize=(14, 10))
```

## Operation Timing

Each Operation node records execution duration:

```python
for node in builder.get_graph().internal_nodes.values():
    if hasattr(node, "execution") and node.execution.duration:
        print(
            f"{node.operation}: "
            f"{node.execution.duration:.2f}s "
            f"({node.execution.status})"
        )
```

## Guidelines

- Enable `verbose=True` during development, disable in production.
- Set `capacity` on `DataLoggerConfig` to prevent unbounded memory
  growth in long-running processes.
- Use `HookRegistry` for cross-cutting concerns (logging, metrics,
  access control) rather than modifying individual call sites.
- Export messages with `branch.to_df()` for post-hoc analysis of
  conversation quality and token usage.
- Check `execution.duration` on Operation nodes to identify bottlenecks
  in graph workflows.
