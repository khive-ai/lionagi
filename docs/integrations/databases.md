# Database Integration

lionagi does not include built-in database adapters. For storage, use `Branch.to_dict()` and `Branch.from_dict()` to serialize and restore conversation state.

For full storage adapter support (PostgreSQL, MongoDB, Redis, etc.), see the sister project [pydapter](https://github.com/khive-ai/pydapter).

## Saving and Loading Branch State

```python
import json
from lionagi import Branch

# Create and use a branch
branch = Branch(system="You are a helpful assistant.")
response = await branch.communicate("Explain quantum computing briefly.")

# Serialize to dict
state = branch.to_dict()

# Save to JSON file
with open("branch_state.json", "w") as f:
    json.dump(state, f, default=str)

# Load from JSON file
with open("branch_state.json") as f:
    data = json.load(f)

restored = Branch.from_dict(data)
```

The serialized state includes messages, logs, chat/parse model configs, system message, and metadata. Tools (callables) are not serialized.

## Async Context Manager

`Branch` supports `async with` for automatic log persistence:

```python
async with Branch(system="Assistant") as branch:
    await branch.communicate("Hello")
# Logs are automatically dumped on exit
```

## pydapter

[pydapter](https://github.com/khive-ai/pydapter) provides storage-agnostic adapters for persisting Pydantic models to various backends:

- PostgreSQL (async)
- MongoDB
- Neo4j
- Qdrant
- Redis
- JSON, CSV, TOML, Excel

Combine `Branch.to_dict()` with pydapter adapters for database persistence.
