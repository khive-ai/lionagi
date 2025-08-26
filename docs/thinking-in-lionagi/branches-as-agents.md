# Branches as Agents

Understanding the Branch abstraction - your agents with memory and tools.

## What is a Branch?

A Branch is LionAGI's core agent abstraction with persistent memory, tools, and
specialized behavior.

```python
from lionagi import Branch, iModel

# Create a branch with model and system prompt
branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="You are a helpful research assistant"
)

# Branch maintains conversation memory
result1 = await branch.communicate("Research AI trends")
result2 = await branch.communicate("What are the key risks?")  # Remembers context

# Check conversation history
print(f"Messages in memory: {len(branch.messages)}")
```

## Key Advantage: Persistent Memory

Unlike stateless API calls, Branches maintain conversation context
automatically.

```python
# LionAGI Branch - stateful conversations
branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini")
)

result1 = await branch.communicate("What is machine learning?")
result2 = await branch.communicate("What are its applications?")  # Remembers context
```

## Branch Components

Branches combine memory, tools, and specialized behavior.

```python
def calculate(expression: str) -> float:
    """Your calculation function"""
    return eval(expression)  # Use safe evaluation in production

# Create branch with tools and system prompt
branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="You are a math tutor. Use tools for calculations.",
    tools=[calculate]  # Direct function passing
)

# Branch has persistent memory and can use tools
result = await branch.communicate("What is 15 * 23? Explain the calculation.")

# Inspect branch components
print(f"Messages: {len(branch.messages)} stored")
print(f"Tools: {len(branch.tools)} available")
print(f"Model: {branch.chat_model.model}")
```

## Specialized Branches

Design branches with focused roles and appropriate tools.

```python
def web_search(query: str) -> str:
    """Your search function"""
    return f"Search results for: {query}"

# Specialized branches
researcher = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Research specialist. Gather comprehensive information.",
    tools=[web_search]
)

analyst = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Data analyst. Provide statistical insights."
)

writer = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Technical writer. Create structured reports."
)

# Each branch specializes in its domain
research = await researcher.communicate("Research AI market trends")
analysis = await analyst.communicate("Analyze the market data")
report = await writer.communicate(f"Write summary: {analysis}")
```

## Multi-Branch Coordination

Coordinate multiple branches within a Session.

```python
from lionagi import Session

# Create session to manage branches
session = Session()

# Create specialized branches
security = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Security expert focused on identifying vulnerabilities"
)

performance = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Performance expert focused on optimization"
)

# Add branches to session
session.include_branches([security, performance])

# Coordinate parallel analysis
code_sample = "def login(user, pwd): return db.query(f'SELECT * FROM users WHERE name={user}')"

results = await asyncio.gather(
    security.communicate(f"Review security: {code_sample}"),
    performance.communicate(f"Review performance: {code_sample}")
)
```

## Memory Management

Control conversation memory for long-running workflows.

```python
branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini")
)

# Build up conversation
for i in range(5):
    await branch.communicate(f"Topic {i+1}: Brief analysis")
    print(f"Messages: {len(branch.messages)}")

# Clear memory when context gets too long
if len(branch.messages) > 8:
    branch.messages.clear()

# Continue with fresh context
result = await branch.communicate("Summarize key findings")
```

## Best Practices

### 1. Design Clear Roles

```python
# Good: Specific, focused role
researcher = Branch(
    system="Research specialist. Gather and verify information from multiple sources."
)

# Avoid: Generic, unfocused role
generic = Branch(
    system="You are a helpful assistant."
)
```

### 2. Use Appropriate Tools

```python
# Match tools to branch purpose
analyst = Branch(
    system="Data analyst",
    tools=[calculate, analyze_data]  # Direct function passing
)
```

### 3. Manage Memory Wisely

```python
# Clear memory for long workflows
if len(branch.messages) > 20:
    branch.messages.clear()

# Preserve important context
important_context = branch.messages[-2:]  # Keep last exchange
branch.messages.clear()
branch.messages.extend(important_context)
```

### 4. Coordinate Through Sessions

```python
# Use Session for multi-branch workflows
session = Session()
session.include_branches([branch1, branch2, branch3])
```

Branches provide stateful, specialized agents with persistent memory, custom
tools, and clear behavioral roles for sophisticated multi-agent workflows.
