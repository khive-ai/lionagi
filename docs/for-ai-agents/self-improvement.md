# Self-Improvement

How to inspect, debug, and adapt your lionagi usage by examining conversation
state, logs, and serialization.

## Inspecting Conversation State

### Branch.messages -- The Conversation History

`branch.messages` is a `Pile[RoledMessage]` containing all messages in the
conversation. Access it to review what has been sent and received.

```python
# Number of messages
len(branch.messages)

# Iterate all messages
for msg in branch.messages:
    print(f"[{msg.role}] {msg.content[:100]}")

# Get the last message
last = branch.messages[-1]

# Access by UUID
msg = branch.messages[some_uuid]
```

### Message Types

```python
from lionagi.protocols.messages import (
    System,              # system prompt
    Instruction,         # user instruction
    AssistantResponse,   # LLM response
    ActionRequest,       # tool call from LLM
    ActionResponse,      # tool result back to LLM
)

# Check message type
from lionagi.protocols.messages import AssistantResponse
if isinstance(branch.messages[-1], AssistantResponse):
    response = branch.messages[-1]
    print(response.content)          # response text
    print(response.model_response)   # raw provider response dict
```

### System Message

```python
# Read the current system message
if branch.system:
    print(branch.system.content)
```

### Convert to DataFrame

```python
df = branch.to_df()
# Columns include: role, sender, recipient, content, created_at, etc.
```

## Inspecting Logs

### Branch.logs -- Activity Logs

`branch.logs` is a `Pile[Log]` containing API call logs, tool invocations, and
other activity records.

```python
# Number of log entries
len(branch.logs)

# Iterate logs
for log in branch.logs:
    print(log.content)  # dict with event details

# Dump logs to file
branch.dump_logs(persist_path="./debug_logs.json", clear=False)

# Async version
await branch.adump_logs(persist_path="./debug_logs.json", clear=False)
```

### Using Async Context Manager for Auto-Cleanup

```python
async with Branch() as b:
    await b.communicate("First question")
    await b.communicate("Follow-up question")
    # On exit, logs are automatically dumped and cleared
```

## Serialization: Save and Restore State

### Branch Serialization

```python
# Save branch state to dict
state = branch.to_dict()
# state contains: messages, logs, chat_model, parse_model, system, log_config, metadata

# Restore from dict
restored = Branch.from_dict(state)
# restored has the same messages, models, and configuration

# Save to JSON file
import json
with open("branch_state.json", "w") as f:
    json.dump(state, f, default=str)

# Load from JSON file
with open("branch_state.json") as f:
    data = json.load(f)
restored = Branch.from_dict(data)
```

### What Gets Serialized

| Field | Included | Notes |
|-------|:--------:|-------|
| messages | Yes | Full conversation history |
| logs | Yes | All activity logs |
| chat_model | Yes | Provider, model, endpoint config |
| parse_model | Yes | Provider, model, endpoint config |
| system | Yes | System message if set |
| log_config | Yes | Logger configuration |
| metadata | Yes | Including clone_from info |
| registered tools | No | Re-register after deserialization |

## Cloning: Explore Alternatives

### Branch.clone() -- Fork a Conversation

`clone()` creates a new Branch with the same messages, system prompt, tools,
and model configuration. Use it to explore alternative conversation paths
without affecting the original.

```python
# Synchronous clone
alt_branch = branch.clone()

# Async clone (acquires message lock)
alt_branch = await branch.aclone()

# Clone with a specific sender ID
alt_branch = branch.clone(sender=some_id)
```

### Clone Behavior by Endpoint Type

| Endpoint Type | Clone Behavior |
|--------------|----------------|
| API (openai, anthropic, etc.) | Shared endpoint (same connection pool) |
| CLI (claude_code, gemini_code, codex) | Fresh endpoint copy (independent session) |

### Exploring Alternatives

```python
# Original conversation
await branch.communicate("Analyze this code for bugs")

# Fork and try a different approach
alt = branch.clone()
await alt.communicate("Now focus specifically on security vulnerabilities")

# Compare results
original_response = branch.messages[-1].content
alternative_response = alt.messages[-1].content
```

### Session.split() -- Fork Within a Session

```python
session = Session()
session.include_branches([branch])

# Split creates a clone and adds it to the session
forked = session.split(branch)
# forked is now managed by the session alongside the original
```

## Debugging Patterns

### Inspect What the LLM Received

```python
# Get the full message sequence that was sent
for msg in branch.messages:
    print(f"Role: {msg.role}")
    print(f"Content: {msg.content}")
    print(f"Created: {msg.created_at}")
    print("---")
```

### Inspect API Call Details

```python
# Logs contain raw API call information
for log in branch.logs:
    content = log.content
    if isinstance(content, dict):
        # Check for API payload
        if "payload" in content:
            print(f"Request: {content['payload']}")
        # Check for response
        if "response" in content:
            print(f"Response: {content['response']}")
```

### Check Token Usage (CLI providers)

```python
from lionagi.protocols.messages import AssistantResponse

for msg in branch.messages:
    if isinstance(msg, AssistantResponse):
        resp = msg.model_response
        if isinstance(resp, dict):
            cost = resp.get("total_cost_usd")
            if cost:
                print(f"Cost: ${cost:.4f}")
```

### Validate Structured Output Quality

```python
from pydantic import BaseModel, ValidationError

class Expected(BaseModel):
    summary: str
    score: float

# Test parsing reliability
successes = 0
for i in range(5):
    alt = branch.clone()
    result = await alt.communicate(
        "Score this code quality",
        response_format=Expected,
    )
    if isinstance(result, Expected):
        successes += 1
        print(f"Trial {i}: score={result.score}")
    else:
        print(f"Trial {i}: parse failed, got {type(result)}")

print(f"Success rate: {successes}/5")
```

## Adapting Model Configuration

### Swap Models at Runtime

```python
from lionagi import iModel

# Upgrade to a more capable model for complex tasks
branch.chat_model = iModel(provider="openai", model="gpt-4.1")

# Use a faster model for simple follow-ups
branch.chat_model = iModel(provider="openai", model="gpt-4.1-mini")
```

### Use Different Models for Chat vs Parse

```python
# Expensive model for conversation, cheap model for parsing
branch = Branch(
    chat_model=iModel(provider="anthropic", model="claude-sonnet-4-20250514"),
    parse_model=iModel(provider="openai", model="gpt-4.1-mini"),
)
```

### Override Model Per Call

```python
# Use a specific model for just this call
result = await branch.communicate(
    "Complex analysis requiring high capability",
    chat_model=iModel(provider="openai", model="gpt-4.1"),
)
```

## Workflow Debugging

### Session.flow() Results

```python
result = await session.flow(builder.get_graph(), verbose=True)

# Inspect results per operation
for node_id, op_result in result.get("operation_results", {}).items():
    print(f"Node {str(node_id)[:8]}: {type(op_result).__name__}")
    if op_result is None:
        print("  -> Operation returned None (possible parse failure)")

# Check what was skipped
for skipped in result.get("skipped_operations", []):
    print(f"Skipped: {str(skipped)[:8]}")

# Check completed
for completed in result.get("completed_operations", []):
    print(f"Completed: {str(completed)[:8]}")
```

### Builder State Inspection

```python
state = builder.visualize_state()
print(f"Total nodes: {state['total_nodes']}")
print(f"Executed: {state['executed_nodes']}")
print(f"Remaining: {state['unexecuted_nodes']}")
print(f"Current heads: {state['current_heads']}")
print(f"Expansions: {state['expansions']}")
```

## Property Reference

### Branch Properties

| Property | Type | Description |
|----------|------|-------------|
| `branch.messages` | `Pile[RoledMessage]` | All conversation messages |
| `branch.logs` | `Pile[Log]` | Activity logs |
| `branch.system` | `System \| None` | System message |
| `branch.chat_model` | `iModel` | Chat model (settable) |
| `branch.parse_model` | `iModel` | Parse model (settable) |
| `branch.tools` | `dict[str, Tool]` | Registered tools |
| `branch.msgs` | `MessageManager` | Message manager (advanced) |
| `branch.acts` | `ActionManager` | Action manager (advanced) |
| `branch.mdls` | `iModelManager` | Model manager (advanced) |

### Branch Methods for State Management

| Method | Description |
|--------|-------------|
| `to_dict()` | Serialize to dict |
| `from_dict(data)` | Restore from dict (classmethod) |
| `to_df()` | Convert messages to DataFrame |
| `clone(sender=None)` | Synchronous fork |
| `aclone(sender=None)` | Async fork (with lock) |
| `dump_logs(clear, persist_path)` | Save logs to file |
| `adump_logs(clear, persist_path)` | Async save logs |
| `register_tools(tools)` | Add tools |

### iModel Properties

| Property | Type | Description |
|----------|------|-------------|
| `model.is_cli` | `bool` | Whether this is a CLI endpoint |
| `model.model_name` | `str` | Model name string |
| `model.request_options` | `type[BaseModel] \| None` | Request schema |
| `model.id` | `UUID` | Unique identifier |
| `model.created_at` | `float` | Creation timestamp |
