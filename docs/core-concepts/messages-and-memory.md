# Messages and Memory

LionAGI manages conversation state through a structured message system. Understanding which operations add to history and how messages are organized is essential for building reliable workflows.

## The Key Distinction: chat() vs communicate()

The most important thing to understand about LionAGI's memory model:

- **`chat()`** does **not** add messages to conversation history. It reads the existing history for context but leaves it unchanged.
- **`communicate()`** **does** add both the instruction and the response to history.

```python
from lionagi import Branch

branch = Branch(system="You are a helpful assistant")

# chat() -- stateless, no history changes
response = await branch.chat("What is 2 + 2?")
print(len(branch.messages))  # 1 (only the system message)

# communicate() -- adds to history
response = await branch.communicate("What is 2 + 2?")
print(len(branch.messages))  # 3 (system + instruction + response)

# Second communicate() sees the full conversation
response = await branch.communicate("What about 3 + 3?")
print(len(branch.messages))  # 5 (system + 2 instructions + 2 responses)
```

!!! warning "Common Mistake"
    If you use `chat()` expecting the branch to remember previous exchanges, it will not work. The model receives existing history as context, but the new instruction and response are never stored. Use `communicate()` for stateful conversations.

Other operations that **add to history**: `communicate()`, `operate()`, `ReAct()`, `act()`

Operations that **do not add to history**: `chat()`, `parse()`, `interpret()`

## Message Types

All messages inherit from `RoledMessage`, which extends `Node` (and therefore `Element`). Each message has a UUID, timestamp, role, structured content, sender, and recipient.

### RoledMessage Hierarchy

```
RoledMessage (base)
  |-- System          -- Sets conversation context and behavior
  |-- Instruction     -- User input (instructions, context, images)
  |-- AssistantResponse -- LLM replies
  |-- ActionRequest   -- Tool call from the LLM
  |-- ActionResponse  -- Tool execution result
```

### System

Sets the overall behavior and context for the conversation. Created when you pass `system=` to `Branch()`.

```python
branch = Branch(system="You are a financial analyst. Be precise with numbers.")

# Access the system message
print(branch.system.content.system_message)
print(branch.system.role)  # MessageRole.SYSTEM
```

System messages support optional datetime stamps:

```python
branch = Branch(
    system="You are a helpful assistant",
    system_datetime=True,  # Adds current timestamp
)
```

### Instruction

Represents user input. Contains structured fields for instruction text, guidance, context, tool schemas, response format, and images.

```python
from lionagi.protocols.messages import Instruction

# Accessing instruction content
for msg in branch.messages:
    if isinstance(msg, Instruction):
        print(msg.content.instruction)
        print(msg.content.guidance)
        print(msg.content.prompt_context)
```

### AssistantResponse

Wraps the LLM's reply. The extracted text is in `content.assistant_response`, while the raw provider response is stored in `metadata["model_response"]`.

```python
from lionagi.protocols.messages import AssistantResponse

# Get the last response
last = branch.msgs.last_response
if last:
    print(last.response)          # Extracted text
    print(last.model_response)    # Raw API response dict
```

### ActionRequest and ActionResponse

These represent tool calls. An `ActionRequest` contains the function name and arguments the LLM wants to invoke. An `ActionResponse` contains the result.

```python
from lionagi.protocols.messages import ActionRequest, ActionResponse

# ActionRequest -- created when the LLM requests a tool call
# request.function -> "search_database"
# request.arguments -> {"query": "revenue"}

# ActionResponse -- created after tool execution
# response.function -> "search_database"
# response.output -> [{"id": 1, "revenue": 50000}]
```

Action requests and responses are linked: `ActionRequest.content.action_response_id` points to the response, and `ActionResponse.content.action_request_id` points back to the request.

## The MessageManager

The `MessageManager` (accessible via `branch.msgs`) stores messages in a `Pile` (an O(1) dict-keyed collection) with a `Progression` that tracks ordering.

```python
# Access the manager
manager = branch.msgs

# Convenience properties
manager.last_response        # Most recent AssistantResponse
manager.last_instruction     # Most recent Instruction
manager.assistant_responses  # Pile of all AssistantResponses
manager.instructions         # Pile of all Instructions
manager.action_requests      # Pile of all ActionRequests
manager.action_responses     # Pile of all ActionResponses
```

### Adding Messages Manually

While operations handle this automatically, you can add messages directly:

```python
branch.msgs.add_message(
    instruction="Manual instruction",
    context=["some context"],
    sender="user",
    recipient=branch.id,
)

branch.msgs.add_message(
    assistant_response="Manual response",
    sender=branch.id,
)
```

### Clearing History

```python
# Remove all messages except the system message
branch.msgs.clear_messages()
```

### Converting to Chat Format

The `to_chat_msgs()` method converts messages to the standard `[{"role": ..., "content": ...}]` format used by LLM APIs:

```python
chat_msgs = branch.msgs.to_chat_msgs()
# [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
```

## Serialization

### Branch Serialization

Serialize an entire branch (messages, models, logs, config) to a dictionary:

```python
# Save
data = branch.to_dict()

# Restore
restored = Branch.from_dict(data)
```

### Message Serialization

Individual messages support `to_dict()` and `from_dict()`:

```python
msg_dict = message.to_dict()
restored_msg = RoledMessage.from_dict(msg_dict)
```

### DataFrame Export

Export messages as a pandas DataFrame:

```python
df = branch.to_df()
# Columns: created_at, role, content, id, sender, recipient, metadata
```

### Clone and Content Properties

Messages support cloning (new ID, reference to original) and a `chat_msg` property for API-ready format:

```python
cloned = message.clone()
print(cloned.metadata["clone_from"])  # Original message ID

api_format = message.chat_msg  # {"role": "user", "content": "..."}
```

## The DataLogger

Every branch has a `DataLogger` (accessible via `branch.logs`) that records API calls and tool invocations. This is separate from conversation messages.

```python
# Access logs
print(len(branch.logs))  # Number of logged events

# Dump logs to file
branch.dump_logs(clear=True, persist_path="./logs/session.json")

# Async variant
await branch.adump_logs(clear=True)
```

The DataLogger is configured through `DataLoggerConfig`:

```python
from lionagi.protocols.generic import DataLoggerConfig

config = DataLoggerConfig(
    persist_dir="./data/logs",
    capacity=100,        # Auto-dump after 100 entries
    extension=".json",   # .json, .csv, or .jsonl
    auto_save_on_exit=True,
)

branch = Branch(log_config=config)
```

### Async Context Manager

Branch supports async context manager usage. On exit, logs are automatically dumped:

```python
async with Branch(system="Assistant") as branch:
    await branch.communicate("Hello")
    await branch.communicate("How are you?")
# Logs are auto-dumped when exiting the context
```

## Multi-Branch Memory

Each branch maintains independent memory. Branches within a Session do not share conversation history:

```python
from lionagi import Session, Branch

session = Session()

researcher = Branch(system="Research specialist", name="researcher")
critic = Branch(system="Critical analyst", name="critic")
session.include_branches([researcher, critic])

await researcher.communicate("Research AI safety")
await critic.communicate("Analyze risks of AI")

# Each branch has its own memory
print(len(researcher.messages))  # Independent count
print(len(critic.messages))      # Independent count
```

## Best Practices

**Use `communicate()` for conversations** where context matters across exchanges. Use `chat()` for isolated queries or internal orchestration logic.

**Use descriptive system prompts** to set consistent behavior:

```python
branch = Branch(
    system="You are a senior data analyst. Always include statistical significance when reporting findings."
)
```

**Monitor message count** in long-running conversations. Large message histories increase token usage and cost:

```python
if len(branch.messages) > 50:
    # Consider summarizing or starting a new branch
    pass
```

**Use `clear_messages`** in `communicate()` when you need a fresh start without creating a new branch:

```python
await branch.communicate("Start a new topic", clear_messages=True)
```

## Next Steps

- [Operations](operations.md) -- which operations add to history
- [Tools and Functions](tools-and-functions.md) -- how ActionRequest/ActionResponse flow works
- [Sessions and Branches](sessions-and-branches.md) -- managing multiple conversation branches
