# Sessions and Branches

Sessions and Branches are the core abstractions that power LionAGI's orchestration engine.

- **Branch**: An individual conversation thread with its own message history, tools, and model configuration
- **Session**: A workspace that coordinates multiple branches and executes graph-based workflows

## Quick Start

```python
from lionagi import Branch, iModel

branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
    system="You are a helpful research assistant."
)

response = await branch.communicate("What are the latest trends in quantum computing?")
```

## Branch

A `Branch` is the primary interface for all LLM operations. It manages four internal components:

- **MessageManager** -- Conversation history (messages + ordering)
- **ActionManager** -- Tool registry and invocation
- **iModelManager** -- Chat and parse model instances
- **DataLogger** -- Activity logging

### Creating a Branch

```python
from lionagi import Branch, iModel

# Minimal -- uses default provider and model
branch = Branch()

# With explicit configuration
branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1"),
    parse_model=iModel(provider="openai", model="gpt-4.1-mini"),
    system="Expert financial analyst",
    name="finance_analyst",
    tools=[calculate_roi, fetch_market_data]
)
```

**Constructor parameters:**

| Parameter             | Type                      | Description                                          |
|-----------------------|---------------------------|------------------------------------------------------|
| `chat_model`          | `iModel` or `dict`        | Primary model for conversations                      |
| `parse_model`         | `iModel` or `dict`        | Model for structured parsing (defaults to chat_model) |
| `system`              | `str` or `System`         | System prompt                                        |
| `name`                | `str`                     | Human-readable branch name                           |
| `tools`               | `list` of functions       | Tools to register                                    |
| `messages`            | `Pile[RoledMessage]`      | Pre-existing messages                                |
| `system_sender`       | `SenderRecipient`         | Sender attributed to system message                  |
| `system_datetime`     | `bool` or `str`           | Include timestamp in system message                  |
| `log_config`          | `DataLoggerConfig`        | Logging configuration                                |
| `user`                | `SenderRecipient`         | Owner/sender of this branch                          |

### Branch Operations

Every Branch provides these operations:

#### chat() -- Raw LLM Call

Sends a message to the LLM and returns the response. Does **not** add messages to conversation history, giving you full control over what gets recorded.

```python
response = await branch.chat("Explain dark matter briefly")

# With structured output
from pydantic import BaseModel

class Answer(BaseModel):
    explanation: str
    confidence: float

response = await branch.chat(
    "Explain dark matter",
    response_format=Answer
)
```

#### communicate() -- Conversational Exchange

Like `chat()`, but automatically adds instruction and response to the conversation history. Supports structured output and follow-up questions.

```python
# Messages are added to history
await branch.communicate("What is reinforcement learning?")

# Follow-up references previous context
await branch.communicate("How does it compare to supervised learning?")

# With structured output
result = await branch.communicate(
    "Summarize our discussion",
    response_format=SummaryModel
)
```

**Key parameters:**

- `instruction` -- The user message
- `guidance` -- Additional system-level guidance
- `context` -- Extra context data
- `response_format` -- Pydantic model for structured output
- `clear_messages` -- Clear history before sending
- `skip_validation` -- Return raw string without parsing

#### operate() -- Tool Calling with Validation

Combines LLM interaction with automatic tool invocation and response validation. Messages are added to conversation history.

```python
def search(query: str, limit: int = 10) -> list[dict]:
    """Search the knowledge base."""
    return [{"title": "Result", "score": 0.95}]

branch = Branch(system="Research assistant", tools=[search])

result = await branch.operate(
    instruction="Find information about quantum computing",
    response_format=ResearchResult,
    reason=True,          # Include chain-of-thought
    actions=True,         # Signal tool use is expected
    invoke_actions=True   # Auto-execute tool calls (default)
)
```

**Key parameters:**

- `instruction` -- The task instruction
- `response_format` -- Pydantic model for the response
- `tools` -- Additional tools for this call
- `reason` -- Include chain-of-thought reasoning
- `actions` -- Signal that tool use is expected
- `invoke_actions` -- Automatically execute tool calls (default: `True`)
- `action_strategy` -- `"concurrent"` or `"sequential"` (default: `"concurrent"`)

#### parse() -- Structured Text Extraction

Parses raw text into a Pydantic model using the parse model. Does not add messages to conversation history.

```python
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    amount: float
    date: str
    line_items: list[str]

raw = """
Invoice from Acme Corp
Date: 2025-03-15
Amount: $1,234.56
Items: Widget A, Widget B, Service C
"""

invoice = await branch.parse(raw, response_format=Invoice)
print(invoice.vendor)   # "Acme Corp"
print(invoice.amount)   # 1234.56
```

**Key parameters:**

- `text` -- Raw text to parse
- `response_format` or `request_type` -- Target Pydantic model
- `max_retries` -- Retry count on parse failure (default: 3)
- `fuzzy_match` -- Attempt fuzzy field matching (default: `True`)
- `handle_validation` -- `"raise"`, `"return_value"`, or `"return_none"`

#### ReAct() -- Multi-Step Reasoning

Implements the ReAct paradigm: the model reasons about the task, takes actions (tool calls), observes results, and iterates until it reaches a conclusion.

```python
result = await branch.ReAct(
    instruct={
        "instruction": "Research and compare the top 3 cloud providers",
        "guidance": "Use search tools to gather current pricing data"
    },
    extension_allowed=True,  # Allow multiple reasoning steps
    max_extensions=3,        # Up to 3 additional steps
    verbose=True             # Log reasoning steps
)
```

**Key parameters:**

- `instruct` -- `Instruct` object or dict with `instruction`, `guidance`, `context`
- `tools` -- Tools available for the ReAct loop
- `response_format` -- Final output schema
- `extension_allowed` -- Allow multi-step reasoning (default: `True`)
- `max_extensions` -- Max reasoning steps (default: 3, capped at 5)
- `interpret` -- Rewrite instructions before starting
- `return_analysis` -- Return `(result, analyses)` tuple
- `verbose` -- Log intermediate reasoning steps

#### select() -- Choice Selection

Choose from a list of options using LLM reasoning.

```python
from enum import Enum

class Priority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

result = await branch.select(
    instruct={"instruction": "Classify the urgency of this bug report"},
    choices=Priority,
    max_num_selections=1
)
```

#### interpret() -- Prompt Refinement

Rewrites user input into a clearer, more structured prompt. Does not add messages to history.

```python
refined = await branch.interpret(
    text="how do I do marketing analytics",
    domain="marketing",
    style="detailed"
)
# refined: "Explain step-by-step how to set up a marketing analytics
#  pipeline to track campaign performance..."
```

#### act() -- Direct Tool Execution

Execute tool calls directly, without going through the LLM.

```python
from lionagi.protocols.messages import ActionRequest

responses = await branch.act(
    action_request=action_requests,
    strategy="concurrent"  # or "sequential"
)
```

### Async Context Manager

Branch supports `async with` for automatic log flushing on exit:

```python
async with Branch(system="Temporary assistant") as branch:
    result = await branch.communicate("Help me analyze this data")
    # Logs are automatically dumped on exit
```

### Cloning Branches

`clone()` creates a new Branch with copied messages and tools but independent state.

```python
original = Branch(
    system="Research assistant",
    chat_model=iModel(provider="openai", model="gpt-4.1"),
    tools=[search, analyze]
)

await original.communicate("Research quantum computing trends")

# Clone the branch -- independent conversation history
clone = original.clone()

# For API-based models: clone shares the same iModel instance (efficient)
# For CLI-based models: clone gets a fresh iModel copy (avoids session conflicts)

# Conversations diverge from here
await original.communicate("Focus on hardware advances")
await clone.communicate("Focus on algorithm improvements")
```

You can also use `await branch.aclone()` for async-safe cloning that locks the message pile during the copy.

### Registering Tools

```python
def calculate(expression: str) -> float:
    """Evaluate a math expression."""
    return eval(expression)

async def fetch_data(url: str) -> dict:
    """Fetch data from a URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()

# Register at construction
branch = Branch(tools=[calculate, fetch_data])

# Or register after construction
branch.register_tools([another_tool], update=True)
```

### Inspecting Branch State

```python
# Messages
for msg in branch.messages:
    print(f"[{msg.role}] {msg.content[:100]}...")

# System prompt
print(branch.system)

# Registered tools
print(list(branch.tools.keys()))

# Models
print(branch.chat_model.model_name)
print(branch.parse_model.model_name)

# Logs
print(len(branch.logs))

# Export to DataFrame
df = branch.to_df()
```

## Session

A `Session` manages multiple branches and provides workflow orchestration via `flow()`.

### Creating a Session

```python
from lionagi import Session

# Session auto-creates a default branch
session = Session(name="my_project")

# Access the default branch
branch = session.default_branch
response = await branch.communicate("Hello")
```

### Creating Branches Within a Session

```python
session = Session()

# Create and include a branch
researcher = session.new_branch(
    name="researcher",
    system="Research specialist",
    imodel=iModel(provider="anthropic", model="claude-sonnet-4-5-20250929")
)

writer = session.new_branch(
    name="writer",
    system="Technical writer"
)

# Or include existing branches
external_branch = Branch(name="external", system="External service")
session.include_branches(external_branch)
```

### Accessing Branches

```python
# By name
researcher = session.get_branch("researcher")

# By ID
branch = session.get_branch(branch_id)

# Change default branch
session.change_default_branch(writer)
```

### Splitting Branches

Split creates a clone of a branch within the session:

```python
# Clone a branch and add the clone to the session
clone = session.split(researcher)

# Async-safe version
clone = await session.asplit(researcher)
```

### Registering Custom Operations

Session supports registering custom operations that branches can execute via the Builder:

```python
session = Session()

# Register with decorator
@session.operation()
async def summarize_findings(context: dict) -> str:
    branch = session.default_branch
    return await branch.communicate(f"Summarize: {context}")

# Register explicitly
session.register_operation("custom_op", my_function)
```

### flow() -- Graph-Based Workflow Execution

Execute a DAG of operations across multiple branches:

```python
from lionagi import Session, Builder

session = Session()
analyzer = session.new_branch(name="analyzer", system="Data analyst")
reviewer = session.new_branch(name="reviewer", system="Code reviewer")

builder = Builder("review_pipeline")

# Parallel analysis steps
analysis_id = builder.add_operation(
    "communicate",
    instruction="Analyze the performance data",
    branch=analyzer
)

review_id = builder.add_operation(
    "communicate",
    instruction="Review the code changes",
    branch=reviewer
)

# Aggregation step (waits for both)
builder.add_aggregation(
    "communicate",
    instruction="Combine analysis and review into final report",
    source_node_ids=[analysis_id, review_id],
    branch=analyzer
)

# Execute
result = await session.flow(
    graph=builder.get_graph(),
    context={"project": "Project Alpha"},
    parallel=True,
    max_concurrent=5,
    verbose=True
)
```

**flow() parameters:**

| Parameter        | Type           | Description                                   |
|------------------|----------------|-----------------------------------------------|
| `graph`          | `Graph`        | Workflow graph from Builder                   |
| `context`        | `dict`         | Initial context for the workflow              |
| `parallel`       | `bool`         | Execute independent operations in parallel    |
| `max_concurrent` | `int`          | Max concurrent operations (default: 5)        |
| `verbose`        | `bool`         | Enable verbose logging                        |
| `default_branch` | `Branch`       | Branch to use as default (defaults to session's) |

## Multi-Branch Patterns

### Parallel Expert Panel

```python
import asyncio
from lionagi import Session, iModel

session = Session()

experts = {
    role: session.new_branch(
        name=f"{role}_expert",
        system=f"You are an expert in {role}."
    )
    for role in ["technical", "business", "legal"]
}

# Gather opinions in parallel
question = "What are the risks of deploying this AI system?"

results = await asyncio.gather(*[
    branch.communicate(question)
    for branch in experts.values()
])

# Synthesize with moderator
moderator = session.new_branch(name="moderator", system="Consensus builder")
synthesis = await moderator.communicate(
    f"Synthesize these expert opinions into a unified assessment:\n"
    + "\n".join(f"- {role}: {result}" for role, result in zip(experts.keys(), results))
)
```

### Sequential Pipeline

```python
session = Session()

parser = session.new_branch(name="parser", system="Extract structured data")
validator = session.new_branch(name="validator", system="Validate accuracy")
formatter = session.new_branch(name="formatter", system="Format for output")

async def process_document(document: str):
    parsed = await parser.communicate(f"Extract key data from: {document}")
    validated = await validator.communicate(f"Validate this data: {parsed}")
    formatted = await formatter.communicate(f"Format for report: {validated}")
    return formatted
```

### Fork-and-Compare

Use `clone()` to explore different approaches from the same starting point:

```python
branch = Branch(system="Strategy advisor")
await branch.communicate("Our company is considering expanding into Asia.")

# Fork the conversation
optimistic = branch.clone()
pessimistic = branch.clone()

plan_a = await optimistic.communicate("Assume best-case scenario. What's the plan?")
plan_b = await pessimistic.communicate("Assume worst-case scenario. What's the plan?")

# Compare
comparison = await branch.communicate(
    f"Compare these two strategies:\n"
    f"Optimistic: {plan_a}\n"
    f"Pessimistic: {plan_b}"
)
```

## Builder: Workflow Graphs

The `Builder` (alias for `OperationGraphBuilder`) constructs directed acyclic graphs of operations for `session.flow()`.

### Basic Usage

```python
from lionagi import Builder

builder = Builder("my_workflow")

# Add sequential operations (auto-linked)
step1 = builder.add_operation("communicate", instruction="Step 1")
step2 = builder.add_operation("communicate", instruction="Step 2")
# step2 automatically depends on step1

# Add with explicit dependencies
step3 = builder.add_operation(
    "communicate",
    instruction="Step 3",
    depends_on=[step1]  # Skip step2, depend directly on step1
)

# Get the graph for execution
graph = builder.get_graph()
```

### Parallel and Aggregation

```python
builder = Builder("parallel_flow")

# These run in parallel (no dependencies between them)
a = builder.add_operation("communicate", instruction="Task A")

# Reset heads to allow parallel operations
builder._current_heads = []
b = builder.add_operation("communicate", instruction="Task B")
builder._current_heads = []
c = builder.add_operation("communicate", instruction="Task C")

# Aggregate results
builder.add_aggregation(
    "communicate",
    instruction="Combine results from A, B, and C",
    source_node_ids=[a, b, c]
)
```

### Dynamic Expansion

Expand the graph based on execution results:

```python
from lionagi.operations.builder import ExpansionStrategy

builder = Builder("dynamic")

gen_id = builder.add_operation(
    "operate",
    instruction="Generate 5 sub-tasks",
    response_format=TaskList
)

# Execute first phase
result = await session.flow(graph=builder.get_graph())

# Expand based on results
builder.expand_from_result(
    items=result.tasks,
    source_node_id=gen_id,
    operation="communicate",
    strategy=ExpansionStrategy.CONCURRENT
)

# Execute expanded graph
final = await session.flow(graph=builder.get_graph())
```

### Assigning Branches to Operations

Direct specific operations to specific branches:

```python
builder = Builder("multi_branch")

builder.add_operation(
    "communicate",
    instruction="Research the topic",
    branch=researcher   # Runs on researcher branch
)

builder.add_operation(
    "communicate",
    instruction="Write the report",
    branch=writer,      # Runs on writer branch
    depends_on=[...]
)
```

## Best Practices

### Encapsulate in Classes

```python
from lionagi import Session, iModel

class DocumentProcessor:
    def __init__(self):
        self.session = Session(name="doc_processor")
        self.parser = self.session.new_branch(
            name="parser", system="Parse documents accurately"
        )
        self.validator = self.session.new_branch(
            name="validator", system="Validate extracted data"
        )

    async def process(self, document: str) -> dict:
        parsed = await self.parser.communicate(f"Parse: {document}")
        validated = await self.validator.communicate(f"Validate: {parsed}")
        return {"parsed": parsed, "validated": validated}
```

### Use Factory Functions

```python
from lionagi import Branch, iModel

def create_expert(domain: str, provider: str = "openai") -> Branch:
    return Branch(
        name=f"{domain}_expert",
        chat_model=iModel(provider=provider, model="gpt-4.1"),
        system=f"You are an expert in {domain}. Be precise and cite sources."
    )

finance = create_expert("finance")
tech = create_expert("technology", provider="anthropic")
```

### Per-User Branch Isolation

```python
from lionagi import Session

session = Session(name="app")
user_branches: dict[str, Branch] = {}

def get_user_branch(user_id: str) -> Branch:
    if user_id not in user_branches:
        user_branches[user_id] = session.new_branch(
            name=f"user_{user_id}",
            system="Helpful assistant"
        )
    return user_branches[user_id]

# Each user has isolated conversation history
branch = get_user_branch("user_123")
response = await branch.communicate("Hello!")
```

### Resource Cleanup with Context Managers

```python
async with Branch(system="Temporary analysis") as branch:
    result = await branch.communicate("Analyze this data")
    # Logs flushed automatically on exit
```
