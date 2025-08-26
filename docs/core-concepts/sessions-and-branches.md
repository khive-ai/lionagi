# Sessions and Branches

Understanding the core abstractions of LionAGI's orchestration engine.

## The Evolution

### Branch: From Toolbox to Space

A `Branch` is evolving from a collection of methods to a **computational space** where operations happen.

## Quick Example

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

## Core Concepts

- **Session**: Workspace coordinating multiple branches
- **Branch**: Agent with independent memory and tools
- **Memory isolation**: Each branch maintains separate conversation history

## Multi-Agent Coordination

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
    await tg.start_soon(research_task)
    await tg.start_soon(critique_task)

print(results)  # Both tasks complete when TaskGroup exits
```

## Session as Workspace

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
    report = await report_writer.chat(f"Report on: {analysis}")
    return {"analysis": analysis, "report": report}
```

## Branch as Agent

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
await research_agent.chat("What's the latest in renewable energy?")
await research_agent.chat("Compare that to the previous answer")  # References history

# Clone for parallel work
clone = research_agent.clone()
clone.name = "research_clone"

# Independent memory, shared configuration
original = await research_agent.chat("Research wind energy")
parallel = await clone.chat("Research solar energy")
```

## Multi-Branch Coordination

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
            await tg.start_soon(get_opinion, expert)
    
    # Synthesis
    moderator = session.new_branch(name="moderator", system="Consensus builder")
    consensus = await moderator.chat(f"Synthesize: {results}")
    
    return consensus
```

## Builder Pattern

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

## Best Practices

### Session Management

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

### Branch Factory

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

### Memory Management

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

Sessions provide workspace coordination while Branches encapsulate individual
agent capabilities and memory, enabling flexible orchestration patterns with
clear boundaries.
