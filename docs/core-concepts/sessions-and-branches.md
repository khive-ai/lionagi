# Sessions and Branches

Understanding the core abstractions that power LionAGI's orchestration engine.

## Core Concepts Overview

At the heart of LionAGI are two fundamental abstractions that enable powerful multi-agent coordination:

- **Session**: A workspace that coordinates multiple agents and manages their interactions
- **Branch**: An individual agent with its own memory, tools, and specialized capabilities

Think of a Session as a project workspace where multiple expert agents collaborate, while each Branch represents a distinct expert with isolated memory and context.

## The Evolution: Branch as Computational Space

A `Branch` has evolved from a simple collection of methods to a **computational space** where AI operations happen. Each Branch maintains independent state, enabling true parallel processing without interference between agents.

## Quick Example

Let's start with a simple example to see how Sessions and Branches work together:

```python
from lionagi import Session, Branch, iModel

session = Session()

# Create specialized agent
researcher = Branch(
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="Research specialist"
)

session.include_branches(researcher)
result = await researcher.communicate("Analyze quantum computing trends")
```

This example demonstrates the basic pattern: create a Session as your workspace, define a specialized Branch with specific capabilities, and coordinate their interactions through the Session.

## Key Benefits

Understanding these core concepts unlocks several powerful capabilities:

- **Memory Isolation**: Each Branch maintains separate conversation history, preventing context bleeding → See [Messages and Memory](messages-and-memory.md)
- **Parallel Processing**: Multiple Branches can work simultaneously without interference → See [Fan-Out/In Pattern](../patterns/fan-out-in.md)
- **Specialized Roles**: Each Branch can have distinct system prompts, tools, and configurations → See [Tools and Functions](tools-and-functions.md)  
- **Coordinated Workflows**: Sessions orchestrate complex multi-agent interactions → See [Operations](operations.md)

## Multi-Agent Coordination

Now let's see how multiple Branches can work together in parallel. This pattern is particularly powerful when you need different perspectives on the same problem:

```python
from lionagi import Session, Branch, iModel
import lionagi as ln

session = Session()

researcher = Branch(
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="Research specialist"
)

critic = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-sonnet-20240229"),
    system="Critical reviewer"
)

session.include_branches([researcher, critic])

# Parallel execution with TaskGroup
results = {}

async def research_task():
    results["research"] = await researcher.communicate("Research quantum computing")

async def critique_task():
    results["critique"] = await critic.communicate("Critique quantum computing risks")

async with ln.create_task_group() as tg:
    tg.start_soon(research_task)
    tg.start_soon(critique_task)

print(results)  # Both tasks complete when TaskGroup exits
```

In this example, we create two specialized agents that work simultaneously. The researcher gathers information while the critic evaluates risks, both running in parallel for faster execution. The TaskGroup ensures both operations complete before proceeding.

## Session as Workspace

Sessions act as project workspaces where you can create and manage multiple specialized Branches. This pattern is ideal for building teams of AI agents with distinct roles:

```python
from lionagi import Session

session = Session(name="project_alpha")

# Create specialized branches
data_analyst = session.new_branch(
    name="data_analyst",
    system="Data analysis specialist",
    tools=[analyze_function]
)

report_writer = session.new_branch(
    name="report_writer",
    system="Report writing specialist"
)

# Coordinate workflow
async def generate_report(data):
    analysis = await data_analyst.chat(f"Analyze: {data}")
    report = await report_writer.communicate(f"Report on: {analysis}")
    return {"analysis": analysis, "report": report}
```

Here we use `session.new_branch()` to create Branches directly within the Session context. This approach provides better organization and makes it easier to coordinate workflows between agents.

## Branch as Agent

Each Branch functions as an independent agent with its own memory, tools, and conversation history. This design enables sophisticated interactions and parallel processing:

```python
from lionagi import Branch, iModel

# Branch with tools and memory
research_agent = Branch(
    name="researcher",
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="Research specialist",
    tools=[web_search, calculator]
)

# Maintains conversation history
await research_agent.communicate("What's the latest in renewable energy?")
await research_agent.communicate("Compare that to the previous answer")  # References history

# Clone for parallel work
clone = research_agent.clone()
clone.name = "research_clone"

# Independent memory, shared configuration
original = await research_agent.chat("Research wind energy")
parallel = await clone.chat("Research solar energy")
```

Notice how the Branch maintains conversation history across multiple interactions, allowing for contextual follow-up questions. The `clone()` method creates an independent copy with the same configuration but separate memory, enabling parallel work streams.

## Multi-Branch Coordination

For complex decision-making, you can coordinate multiple expert Branches to gather diverse perspectives. This expert panel pattern is particularly effective for comprehensive analysis:

```python
from lionagi import Session, iModel
import lionagi as ln

async def expert_panel():
    session = Session()
    
    # Create expert branches
    experts = [
        session.new_branch(
            name=f"{role}_expert",
            system=f"{role.title()} expert"
        )
        for role in ["technical", "business", "regulatory"]
    ]
    
    topic = "AI regulation impact"
    
    # Parallel expert opinions
    results = {}
    
    async def get_opinion(expert):
        results[expert.name] = await expert.chat(f"Your view on: {topic}")
    
    async with ln.create_task_group() as tg:
        for expert in experts:
            tg.start_soon(get_opinion, expert)
    
    # Synthesis
    moderator = session.new_branch(name="moderator", system="Consensus builder")
    consensus = await moderator.chat(f"Synthesize: {results}")
    
    return consensus
```

This pattern demonstrates the power of coordinated multi-agent systems: we dynamically create expert Branches with different specializations, gather their opinions in parallel, then use a moderator Branch to synthesize the results into a coherent consensus.

## Builder Pattern

For complex workflows with dependencies, use the Builder pattern to define operation graphs. This approach provides clear control over execution order and data flow:

```python
from lionagi import Session, Builder

session = Session()

# Create processing branches
parser = session.new_branch(name="parser", system="Extract key information")
classifier = session.new_branch(name="classifier", system="Classify content")

# Build workflow graph
builder = Builder("document_flow")

parse_op = builder.add_operation(
    "chat",
    instruction="Extract info from: {document}",
    branch=parser
)

classify_op = builder.add_operation(
    "chat",
    instruction="Classify: {parse_op}",
    depends_on=[parse_op],
    branch=classifier
)

# Execute workflow
result = await session.flow(
    graph=builder.get_graph(),
    context={"document": "Sample document..."}
)
```

The Builder pattern excels at orchestrating sequential operations where later steps depend on earlier results. The `depends_on` parameter ensures the parser completes before the classifier begins, while the `{parse_op}` template automatically injects the parser's output into the classifier's instruction.

## Best Practices

Here are proven patterns for organizing Sessions and Branches in production applications.

### Session Management

Encapsulate Sessions within classes to create reusable, stateful processors:

```python
from lionagi import Session

class DocumentProcessor:
    def __init__(self):
        self.session = Session(name="processor")
        self.parser = self.session.new_branch(name="parser", system="Parse documents")
        self.validator = self.session.new_branch(name="validator", system="Validate data")
    
    async def process(self, document):
        parsed = await self.parser.chat(f"Parse: {document}")
        validated = await self.validator.chat(f"Validate: {parsed}")
        return validated
```

This pattern provides clean encapsulation and makes it easy to manage long-lived agent systems with consistent behavior.

### Branch Factory

Use factory functions to create consistently configured Branches with specialized roles:

```python
from lionagi import Branch, iModel

def create_specialist(role: str, domain: str) -> Branch:
    return Branch(
        name=f"{role}_{domain}",
        chat_model=iModel(provider="openai", model="gpt-4"),
        system=f"{role.title()} expert in {domain}"
    )

# Usage
researcher = create_specialist("researcher", "finance")
analyst = create_specialist("analyst", "data")
```

Factory functions ensure consistent configuration across similar agent types while making it easy to customize roles and domains.

### Memory Management

For applications serving multiple users, implement per-user Branch isolation to maintain separate conversation contexts:

```python
from lionagi import Session

session = Session(name="conversation_mgr")
user_branches = {}

async def get_user_branch(user_id: str):
    if user_id not in user_branches:
        user_branches[user_id] = session.new_branch(
            name=f"user_{user_id}",
            system="Helpful assistant"
        )
    return user_branches[user_id]

# Maintains separate conversation history per user
branch = await get_user_branch("123")
response = await branch.chat("Hello")
```

This pattern ensures each user has their own isolated agent with persistent memory, preventing cross-contamination of conversation contexts.

## Summary

Sessions and Branches form the foundation of LionAGI's orchestration capabilities. Sessions provide workspace coordination and workflow management, while Branches encapsulate individual agent capabilities with isolated memory and specialized tools. Together, they enable sophisticated multi-agent systems with clear boundaries, predictable behavior, and powerful parallel processing capabilities.

Understanding these core abstractions unlocks the full potential of LionAGI for building production-ready AI systems that scale from simple single-agent interactions to complex multi-agent orchestration patterns.
