# LionAGI Philosophy: Observable Workflows

## The Core Idea

LionAGI puts you in control. Instead of hiding LLM interactions behind opaque "agent" abstractions, LionAGI gives you observable, composable building blocks. You define the workflow. You control the flow. Every step is visible, testable, and reproducible.

```python
from lionagi import Branch, iModel

branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
    system="You are a financial analyst."
)

# Every operation is explicit and observable
response = await branch.communicate("Summarize Q4 earnings for NVDA")
```

## Three Layers of Abstraction

LionAGI is built on three layers, each adding structure on top of the previous one:

```text
Layer 1: iModel           -- Provider-agnostic LLM interface
Layer 2: Branch            -- Single conversation thread with tools
Layer 3: Session + Builder -- Multi-branch coordination via DAG workflows
```

You can use any layer independently. Most applications only need Layer 2.

## Branch: The Primary Cognitive Unit

A `Branch` is a single conversation thread with its own message history, tools, and model configuration. It is the primary surface for all LLM operations.

### Available Operations

Every Branch supports these operations, each with a specific purpose:

| Operation       | Purpose                                             | Messages added? |
|-----------------|-----------------------------------------------------|-----------------|
| `chat()`        | Raw LLM call; returns response without adding to history | No         |
| `communicate()` | Conversational exchange with optional structured output | Yes            |
| `operate()`     | Tool calling with structured response validation     | Yes             |
| `parse()`       | Parse raw text into a Pydantic model                 | No              |
| `ReAct()`       | Multi-step reasoning with tool use (think-act-observe) | Yes           |
| `select()`      | Choose from a set of options                         | Yes             |
| `interpret()`   | Rewrite user input into a clearer prompt             | No              |
| `act()`         | Execute tool calls directly                          | Yes             |

### chat() -- Raw LLM Interaction

The simplest operation. Sends a message and gets a response. Does not add messages to the conversation history, making it suitable for one-off queries or when you want manual control.

```python
from lionagi import Branch

branch = Branch(system="Expert physicist")

# Returns the response content directly
response = await branch.chat("Explain dark matter in one paragraph")
print(response)
```

### communicate() -- Conversational Exchange

Like `chat()`, but automatically adds messages to the branch's conversation history. Supports structured output via `response_format`.

```python
from pydantic import BaseModel
from lionagi import Branch

class Summary(BaseModel):
    key_points: list[str]
    sentiment: str

branch = Branch(system="News analyst")

# Messages are added to history automatically
result = await branch.communicate(
    "Summarize the latest AI regulation developments",
    response_format=Summary
)
print(result.key_points)

# Follow-up questions reference conversation history
follow_up = await branch.communicate("Which point is most impactful?")
```

### operate() -- Tool Calling with Structured Output

Combines LLM interaction with automatic tool invocation and response validation. This is the primary operation for building agents that use tools.

```python
from pydantic import BaseModel
from lionagi import Branch

def search_database(query: str, limit: int = 10) -> list[dict]:
    """Search the product database."""
    return [{"name": "Widget A", "price": 9.99}]

def calculate_discount(price: float, percent: float) -> float:
    """Calculate discounted price."""
    return price * (1 - percent / 100)

class PricingResult(BaseModel):
    product: str
    original_price: float
    discounted_price: float
    recommendation: str

branch = Branch(
    system="Pricing analyst with database access",
    tools=[search_database, calculate_discount]
)

result = await branch.operate(
    instruction="Find Widget A and calculate a 15% discount",
    response_format=PricingResult,
    reason=True  # Include chain-of-thought reasoning
)
print(result.recommendation)
```

### parse() -- Structured Text Extraction

Parses raw text into a Pydantic model. Does not add messages to the conversation. Useful for processing LLM output or external text into structured data.

```python
from pydantic import BaseModel
from lionagi import Branch

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str | None = None

branch = Branch()

raw_text = "John Smith, email: john@example.com, phone: 555-0123"
contact = await branch.parse(raw_text, response_format=ContactInfo)
print(contact.name)  # "John Smith"
```

### ReAct() -- Multi-Step Reasoning

Implements the ReAct (Reason + Act) paradigm. The model thinks about what to do, takes an action (tool call), observes the result, and repeats until it reaches a conclusion. Best for complex tasks that require multiple steps.

```python
from lionagi import Branch

branch = Branch(
    system="Research assistant",
    tools=[web_search, summarize_page, extract_data]
)

result = await branch.ReAct(
    instruct={
        "instruction": "Research the top 3 renewable energy trends in 2025",
        "guidance": "Use web search, then summarize findings"
    },
    extension_allowed=True,
    max_extensions=3,
    verbose=True
)
```

## iModel: Provider-Agnostic Interface

`iModel` wraps any LLM provider behind a uniform interface. You configure it once and use it everywhere.

```python
from lionagi import iModel

# API-based providers
openai_model = iModel(provider="openai", model="gpt-4.1-mini")
claude_model = iModel(provider="anthropic", model="claude-sonnet-4-5-20250929")
gemini_model = iModel(provider="gemini", model="gemini-2.5-flash")

# CLI-based providers (agentic coding tools)
claude_code = iModel(provider="claude_code")
gemini_cli = iModel(provider="gemini_code")
codex_cli = iModel(provider="codex")

# Use as async context manager for automatic cleanup
async with iModel(provider="openai", model="gpt-4.1") as model:
    branch = Branch(chat_model=model)
    result = await branch.communicate("Hello")
```

## Session: Multi-Branch Coordination

A `Session` manages multiple branches and provides `flow()` for executing DAG-based workflows.

```python
from lionagi import Session

session = Session()

# Create specialized branches
researcher = session.new_branch(name="researcher", system="Research expert")
writer = session.new_branch(name="writer", system="Technical writer")

# Coordinate between branches
findings = await researcher.communicate("Research quantum error correction")
report = await writer.communicate(f"Write a summary based on: {findings}")
```

## Builder + flow(): DAG-Based Workflows

For workflows with dependencies, parallel steps, and aggregation, use `Builder` to construct a directed acyclic graph (DAG) of operations, then execute it with `session.flow()`.

```python
from lionagi import Session, Builder

session = Session()

# Create branches for different roles
analyzer = session.new_branch(name="analyzer", system="Data analyst")
reviewer = session.new_branch(name="reviewer", system="Code reviewer")

# Build a workflow graph
builder = Builder("analysis_pipeline")

# Step 1: Two parallel analyses
step_a = builder.add_operation(
    "communicate",
    instruction="Analyze performance metrics",
    branch=analyzer
)

step_b = builder.add_operation(
    "communicate",
    instruction="Review code quality",
    branch=reviewer
)

# Step 2: Aggregate results (depends on both step_a and step_b)
builder.add_aggregation(
    "communicate",
    instruction="Synthesize analysis and review findings",
    source_node_ids=[step_a, step_b],
    branch=analyzer
)

# Execute the workflow
result = await session.flow(
    graph=builder.get_graph(),
    parallel=True,
    max_concurrent=5
)
```

### Dynamic Graph Expansion

The Builder supports incremental workflows where you expand the graph based on intermediate results:

```python
from lionagi import Session, Builder

session = Session()
builder = Builder("dynamic_workflow")

# Initial operation
gen_id = builder.add_operation(
    "operate",
    instruction="Generate 3 research questions about climate change",
    response_format=ResearchQuestions
)

# Execute initial graph
result = await session.flow(graph=builder.get_graph())

# Expand based on results
if hasattr(result, "questions"):
    builder.expand_from_result(
        items=result.questions,
        source_node_id=gen_id,
        operation="communicate",
    )

# Execute expanded graph
final = await session.flow(graph=builder.get_graph())
```

## Design Principles

### 1. You Control the Flow

LionAGI does not impose an agent loop or conversation pattern. You write the control flow. You decide when to call the LLM, which tools to invoke, and how to combine results.

```python
# You write the loop, not the framework
for document in documents:
    summary = await branch.communicate(f"Summarize: {document}")
    if "urgent" in summary.lower():
        alert = await branch.operate(
            instruction=f"Draft alert for: {summary}",
            response_format=AlertMessage
        )
        send_alert(alert)
```

### 2. Observable State

Every Branch exposes its state -- messages, logs, tools, models -- as inspectable properties.

```python
# Inspect conversation history
for msg in branch.messages:
    print(f"[{msg.role}] {msg.content[:80]}...")

# Check registered tools
print(branch.tools.keys())

# Access logs
for log in branch.logs:
    print(log.content)
```

### 3. Composition Over Configuration

Build complex systems by composing simple parts. Branches are independent units that can be combined through explicit coordination.

```python
# Clone a branch to fork a conversation
clone = branch.clone()

# Original and clone have independent histories
await branch.communicate("Explore option A")
await clone.communicate("Explore option B")

# Compare results
```

### 4. Async Context Managers for Resource Safety

Both `Branch` and `iModel` support async context managers that handle cleanup automatically.

```python
async with Branch(system="Temporary assistant") as branch:
    result = await branch.communicate("Help me with this task")
    # Logs are automatically flushed on exit
```

## Summary

| Concept   | What It Does                                              |
|-----------|-----------------------------------------------------------|
| `iModel`  | Wraps any LLM provider behind a uniform async interface   |
| `Branch`  | Single conversation thread with messages, tools, and models |
| `Session` | Manages multiple branches for coordination                |
| `Builder` | Constructs DAG workflows for `session.flow()` execution   |

LionAGI gives you the building blocks. You build the workflow.
