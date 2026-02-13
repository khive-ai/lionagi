# Operations

Operations are the methods on a `Branch` that interact with LLMs. Each operation has different behavior around conversation history, structured output, and tool invocation.

## Operation Overview

| Operation | Adds to History | Structured Output | Tool Calling | Use Case |
|-----------|:-:|:-:|:-:|---|
| `chat()` | No | No | No | Stateless LLM calls, orchestration internals |
| `communicate()` | Yes | Optional | No | Conversational interactions with memory |
| `operate()` | Yes | Yes | Optional | Structured extraction with optional tools |
| `parse()` | No | Yes | No | Parse raw text into Pydantic models |
| `ReAct()` | Yes | Optional | Yes | Multi-step reasoning with tool use |
| `select()` | Yes | Yes | No | Choose from a set of options |
| `interpret()` | No | No | No | Rewrite user input into clearer prompts |
| `act()` | Yes | No | Yes | Execute tool calls directly |

## chat()

The lowest-level LLM operation. Sends an instruction to the model using the current conversation history as context, but **does not** add any messages to the branch.

```python
from lionagi import Branch

branch = Branch(system="You are a helpful assistant")
response = await branch.chat("What is 2 + 2?")
# response is a string: "4"

# Messages are NOT added to history
print(len(branch.messages))  # Only the system message
```

**Key parameters:**

| Parameter | Type | Description |
|---|---|---|
| `instruction` | `str \| dict` | Main instruction text |
| `guidance` | `str \| dict` | Additional system-level guidance |
| `context` | `str \| dict \| list` | Context data for the model |
| `response_format` | `type[BaseModel]` | Request structured JSON response |
| `imodel` | `iModel` | Override the default chat model |
| `images` | `list` | Image URLs or base64 data |
| `return_ins_res_message` | `bool` | If True, returns `(Instruction, AssistantResponse)` tuple |

When `return_ins_res_message=False` (default), returns a plain string. When True, returns the raw message objects for manual processing.

## communicate()

A higher-level wrapper around `chat()` that **adds messages to conversation history**. Use this when you want the branch to remember the exchange.

```python
branch = Branch(system="You are a tutor")

# Both instruction and response are saved to history
await branch.communicate("Explain list comprehensions")
await branch.communicate("Show a more advanced example")

# The second call has full context of the first exchange
print(len(branch.messages))  # system + 2 instructions + 2 responses
```

**Key parameters:**

| Parameter | Type | Description |
|---|---|---|
| `instruction` | `str \| dict` | Main instruction text |
| `response_format` | `type[BaseModel]` | Parse response into a Pydantic model |
| `request_fields` | `dict \| list[str]` | Extract specific fields from response |
| `skip_validation` | `bool` | Return raw string without parsing |
| `num_parse_retries` | `int` | Retry count for parsing failures (max 5) |
| `clear_messages` | `bool` | Clear history before sending |

**Structured output with `communicate()`:**

```python
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str
    score: float

result = await branch.communicate(
    "The product is excellent and well-made",
    response_format=Sentiment,
)
# result is a Sentiment instance
print(result.label, result.score)
```

## operate()

The most feature-rich operation. Combines structured output with optional tool invocation. Messages are added to conversation history.

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    key_points: list[str]
    confidence: float

result = await branch.operate(
    instruction="Analyze this quarterly report",
    context=report_data,
    response_format=Analysis,
)
# result is an Analysis instance
```

**Enabling tool calling with `actions=True`:**

```python
def search_database(query: str) -> list[dict]:
    """Search the internal database."""
    return [{"id": 1, "title": "Result"}]

branch = Branch(tools=[search_database])

result = await branch.operate(
    instruction="Find records about revenue",
    response_format=Analysis,
    actions=True,  # Enable tool invocation
)
```

**Key parameters:**

| Parameter | Type | Description |
|---|---|---|
| `instruction` | `str \| dict` | Main instruction text |
| `instruct` | `Instruct` | Structured instruction object (alternative to raw params) |
| `response_format` | `type[BaseModel]` | Expected output model |
| `actions` | `bool` | Enable tool calling |
| `reason` | `bool` | Request chain-of-thought reasoning |
| `invoke_actions` | `bool` | Execute tool calls automatically (default True) |
| `action_strategy` | `"concurrent" \| "sequential"` | How to run multiple tools (default "concurrent") |
| `tools` | `ToolRef` | Specific tools to make available |
| `skip_validation` | `bool` | Return raw response without parsing |
| `handle_validation` | `"raise" \| "return_value" \| "return_none"` | How to handle parse failures |

## parse()

Parses raw text into a structured Pydantic model using the parse model. Does **not** add messages to conversation history. Useful for post-processing LLM output or extracting structure from arbitrary text.

```python
from pydantic import BaseModel

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str | None = None

raw_text = "John Smith, john@example.com, phone: 555-1234"
result = await branch.parse(raw_text, response_format=ContactInfo)
# result is a ContactInfo instance
```

**Key parameters:**

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | Raw text to parse |
| `response_format` | `type[BaseModel]` | Target Pydantic model |
| `max_retries` | `int` | Number of retry attempts (default 3) |
| `fuzzy_match` | `bool` | Attempt fuzzy key matching (default True) |
| `handle_validation` | `"raise" \| "return_value" \| "return_none"` | Failure handling |
| `similarity_threshold` | `float` | Threshold for fuzzy matching (default 0.85) |

## ReAct()

Implements the Reasoning + Acting paradigm. The model thinks through a problem, optionally uses tools, observes results, and iterates until it reaches a conclusion. Messages are added to history.

```python
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

branch = Branch(tools=[web_search])

result = await branch.ReAct(
    instruct={"instruction": "Research the latest advances in quantum computing"},
    max_extensions=3,
    verbose=True,
)
```

**Key parameters:**

| Parameter | Type | Description |
|---|---|---|
| `instruct` | `Instruct \| dict` | Instruction with optional guidance/context |
| `tools` | `ToolRef` | Tools available for use (defaults to all registered) |
| `response_format` | `type[BaseModel]` | Structure for the final answer |
| `max_extensions` | `int` | Max reasoning-action cycles (default 3, max 5) |
| `extension_allowed` | `bool` | Allow multi-step reasoning (default True) |
| `interpret` | `bool` | Pre-process instruction through `interpret()` |
| `verbose` | `bool` | Print reasoning steps |
| `return_analysis` | `bool` | Return `(result, analyses)` tuple |

!!! note "ReActStream"
    `branch.ReActStream()` is an async generator variant that yields intermediate analysis results as they are produced, rather than waiting for the final answer.

## select()

Presents a set of choices to the model and returns a structured selection. This is a standalone function in `lionagi.operations.select` that uses `operate()` internally.

```python
from lionagi.operations.select.select import select

result = await select(
    branch=branch,
    instruct={"instruction": "Which database is best for this use case?"},
    choices=["PostgreSQL", "MongoDB", "Redis", "DynamoDB"],
    max_num_selections=2,
)
# result.selected contains the chosen items
```

Choices can be a list of strings, a dict mapping keys to descriptions, or an Enum type.

## interpret()

Rewrites a user's raw input into a clearer, more structured prompt. Does **not** add messages to history. Useful as a preprocessing step before other operations.

```python
refined = await branch.interpret(
    "how do i do marketing stuff?",
    domain="marketing",
    style="detailed",
)
# refined might be: "Explain step-by-step how to design and execute a
# marketing strategy, including target audience analysis, channel
# selection, and campaign performance tracking."
```

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | Raw user input |
| `domain` | `str \| None` | Domain hint (e.g., "finance", "devops") |
| `style` | `str \| None` | Style hint (e.g., "concise", "detailed") |

## act()

Executes tool calls directly without going through the LLM. Takes an `ActionRequest` (or equivalent dict) and invokes the matching registered tool. Messages (ActionRequest and ActionResponse) are added to history.

```python
from lionagi.protocols.messages import ActionRequest

response = await branch.act(
    action_request={"function": "search_database", "arguments": {"query": "revenue"}},
    strategy="concurrent",
)
# response is a list of ActionResponse objects
```

| Parameter | Type | Description |
|---|---|---|
| `action_request` | `list \| dict \| ActionRequest` | Tool call(s) to execute |
| `strategy` | `"concurrent" \| "sequential"` | Execution strategy (default "concurrent") |
| `suppress_errors` | `bool` | Log errors instead of raising (default True) |

## Using Operations with Builder

Operations can be composed into DAGs using `Builder` for multi-agent workflows:

```python
from lionagi import Builder, Session, Branch

session = Session()
builder = Builder("analysis_pipeline")

analyst = Branch(system="Data analyst")
writer = Branch(system="Report writer")
session.include_branches([analyst, writer])

# Operations without dependencies run in parallel
analyze = builder.add_operation(
    "operate",
    branch=analyst,
    instruction="Analyze this dataset",
    response_format=Analysis,
)

# Dependencies control execution order
report = builder.add_operation(
    "communicate",
    branch=writer,
    instruction="Write a report from the analysis",
    depends_on=[analyze],
)

results = await session.flow(builder.get_graph())
```

## Next Steps

- [Tools and Functions](tools-and-functions.md) -- registering tools for `operate()` and `ReAct()`
- [Messages and Memory](messages-and-memory.md) -- understanding conversation history
- [Sessions and Branches](sessions-and-branches.md) -- managing multiple conversations
