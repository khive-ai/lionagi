# Migrating from CrewAI

CrewAI's verbose role-based agents → LionAGI's clean graph orchestration.

## Basic Agent Creation

**CrewAI (Verbose):**

```python
from crewai import Agent, Task, Crew

coding_agent = Agent(
    role="Senior Python Developer",
    goal="Craft well-designed and thought-out code", 
    backstory="You are a senior Python developer with extensive experience in software architecture and best practices.",
    allow_code_execution=True
)

task = Task(
    description="Create a Python function to analyze data",
    expected_output="A well-documented Python function",
    agent=coding_agent
)

crew = Crew(agents=[coding_agent], tasks=[task])
result = crew.kickoff()
```

**LionAGI (Clean):**

```python
from lionagi import Branch, iModel

coder = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Senior Python developer"
)

result = await coder.communicate("Create a Python function to analyze data")
```

## Multi-Agent Research

**CrewAI (Complex Setup):**

```python
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="You work at a leading tech think tank...",
    verbose=True
)

writer = Agent(
    role="Tech Content Strategist", 
    goal="Craft compelling content on tech advancements",
    backstory="You are a renowned Content Strategist...",
    verbose=True
)

research_task = Task(
    description="Conduct thorough research about AI Agents in 2025",
    expected_output="A list with 10 bullet points",
    agent=researcher
)

writing_task = Task(
    description="Write a compelling blog post based on research",
    expected_output="A 4 paragraph blog post",
    agent=writer
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True
)

result = crew.kickoff()
```

**LionAGI (Simple Graph):**

```python
from lionagi import Session, Builder

session = Session()
builder = Builder()

research = builder.add_operation(
    "communicate",
    instruction="Research AI Agents developments in 2025"
)

writing = builder.add_operation(
    "communicate", 
    depends_on=[research],
    instruction="Write compelling blog post based on research"
)

result = await session.flow(builder.get_graph())
```

## Parallel Research Crew

**CrewAI (Sequential by Default):**

```python
# CrewAI runs tasks sequentially unless explicitly configured
agents = [researcher1, researcher2, researcher3]
tasks = [task1, task2, task3]

crew = Crew(
    agents=agents,
    tasks=tasks,
    process=Process.sequential  # Default behavior
)

result = crew.kickoff()
```

**LionAGI (Parallel by Nature):**

```python
# Automatic parallel execution
research_nodes = []
for topic in ["transformers", "multimodal", "reasoning"]:
    node = builder.add_operation(
        "communicate",
        instruction=f"Research {topic} in 2025"
    )
    research_nodes.append(node)

synthesis = builder.add_operation(
    "communicate",
    depends_on=research_nodes,
    instruction="Synthesize findings"
)

result = await session.flow(builder.get_graph())  # Parallel execution
```

## Tool Integration

**CrewAI (Limited):**

```python
from crewai_tools import SerperDevTool

search_tool = SerperDevTool()

agent = Agent(
    role="Research Analyst",
    goal="Research topics",
    backstory="...",
    tools=[search_tool]
)
```

**LionAGI (Flexible):**

```python
def custom_search(query: str) -> str:
    # Any custom logic
    return f"Results for {query}"

researcher = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    tools=[custom_search]  # Direct function passing
)

result = await researcher.ReAct(
    instruct={"instruction": "Research AI trends"},
    max_extensions=3
)
```

## Error Handling

**CrewAI (Basic):**

```python
# Limited error handling options
try:
    result = crew.kickoff()
except Exception as e:
    print(f"Crew failed: {e}")
```

**LionAGI (Robust):**

```python
import asyncio

async def robust_workflow():
    try:
        results = await asyncio.gather(
            session.flow(research_graph),
            session.flow(analysis_graph),
            return_exceptions=True
        )
        
        successful = [r for r in results if not isinstance(r, Exception)]
        return successful
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None
```

## Cost and Performance Tracking

**CrewAI (Not Built-in):**

```python
# No built-in cost tracking
# Must implement custom solutions
```

**LionAGI (Native Support):**

```python
from lionagi.protocols.messages.assistant_response import AssistantResponse

# Built-in cost tracking during execution
costs = 0
def get_context(node_id):
    nonlocal costs
    graph = builder.get_graph()
    node = graph.internal_nodes[node_id]
    branch = session.get_branch(node.branch_id, None)
    if branch and len(branch.messages) > 0:
        if isinstance(msg := branch.messages[-1], AssistantResponse):
            costs += msg.model_response.get("total_cost_usd") or 0

# Track costs across workflow
for node in research_nodes:
    get_context(node)

print(f"Total workflow cost: ${costs:.4f}")
```

## Key Advantages

**Shorter Code**: LionAGI requires 70% less boilerplate than CrewAI **Natural
Parallelism**: Built-in parallel execution vs CrewAI's sequential default
**Flexible Tools**: Convert any function vs limited tool ecosystem **Direct
Control**: Graph-based dependencies vs rigid role assignments **Cost Tracking**:
Native monitoring vs manual implementation **Error Recovery**: Robust async
patterns vs basic exception handling

## Migration Benefits

✅ **Less Configuration**: No verbose roles, goals, backstories  
✅ **Better Performance**: Automatic parallel execution  
✅ **More Control**: Explicit dependencies and error handling  
✅ **Cost Visibility**: Built-in usage tracking  
✅ **Simpler Code**: Focus on logic, not boilerplate
