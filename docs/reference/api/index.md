# API Reference

Core classes and functions for LionAGI framework.

## Core Modules

### Session

```python
from lionagi import Session

session = Session()
```

Workspace for coordinating multiple branches and managing graph execution.

### Branch

```python
from lionagi import Branch, iModel

branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="You are a helpful assistant"
)
```

Individual agent abstraction with persistent memory, tools, and specialized
behavior.

### Builder

```python
from lionagi import Builder

builder = Builder()
node = builder.add_operation("communicate", instruction="Hello")
```

Graph construction pattern for building complex multi-agent workflows with
dependencies.

### iModel

```python
from lionagi import iModel

# OpenAI
model = iModel(provider="openai", model="gpt-4o-mini")

# Anthropic  
model = iModel(provider="anthropic", model="claude-3-5-sonnet-20241022")

# Claude Code
model = iModel(provider="claude_code", model="sonnet")
```

Universal model interface supporting OpenAI, Anthropic, Ollama, and Claude Code.

## Operations

### communicate

```python
result = await branch.communicate("What is 2 + 2?")
```

Direct conversation with a single branch.

### ReAct

```python
result = await branch.ReAct(
    instruct={"instruction": "Research AI trends"},
    tools=["search_tool"],
    max_extensions=3
)
```

Reasoning + Acting workflow for systematic problem solving with tools.

### operate

```python
result = await branch.operate(
    instruct=Instruct(instruction="Analyze data", context=data),
    response_format=AnalysisReport
)
```

Structured operation with Pydantic output validation.

## Tools

### Function Tools

```python
def multiply(x: float, y: float) -> float:
    return x * y

branch = Branch(tools=[multiply])  # Direct function passing
```

### ReaderTool

```python
from lionagi.tools.types import ReaderTool

branch = Branch(tools=[ReaderTool])
result = await branch.ReAct(
    instruct={"instruction": "Read document.pdf and summarize"},
    tools=["reader_tool"]
)
```

### Custom Tool Class

```python
from lionagi import Tool
from pydantic import BaseModel

class SearchParams(BaseModel):
    query: str
    max_results: int = 5

def web_search(query: str, max_results: int = 5) -> str:
    return f"Search results for: {query}"

search_tool = Tool(
    func_callable=web_search,
    request_options=SearchParams
)
```

## Graph Execution

### Basic Flow

```python
from lionagi import Session, Builder

session = Session()
builder = Builder()

# Sequential operations
step1 = builder.add_operation("communicate", instruction="Step 1")
step2 = builder.add_operation("communicate", depends_on=[step1], instruction="Step 2")

result = await session.flow(builder.get_graph())
```

### Parallel Flow

```python
# Parallel research
topics = ["AI", "ML", "NLP"]
research_nodes = []

for topic in topics:
    node = builder.add_operation("communicate", instruction=f"Research {topic}")
    research_nodes.append(node)

# Synthesis
synthesis = builder.add_operation(
    "communicate",
    depends_on=research_nodes,
    instruction="Synthesize research findings"
)

result = await session.flow(builder.get_graph())
```

## Data Types

### Instruct

```python
from lionagi.types import Instruct

instruct = Instruct(
    instruction="Analyze the data",
    context={"data": [1, 2, 3]},
    guidance="Focus on trends"
)
```

### Node

```python
# Access node data after execution
graph = builder.get_graph()
for node_id, node in graph.internal_nodes.items():
    branch = session.get_branch(node.branch_id)
    print(f"Node {node_id}: {len(branch.messages)} messages")
```

## Cost Tracking

```python
def get_costs(node_id, builder, session):
    graph = builder.get_graph()
    node = graph.internal_nodes[node_id]
    branch = session.get_branch(node.branch_id, None)
    
    if branch and len(branch.messages) > 0:
        msg = branch.messages[-1]
        if hasattr(msg, 'model_response'):
            return msg.model_response.get("total_cost_usd", 0)
    return 0

total_cost = sum(get_costs(node, builder, session) for node in research_nodes)
```

## Error Handling

```python
try:
    result = await session.flow(builder.get_graph())
except Exception as e:
    import traceback
    traceback.print_exc()
    
    # Access execution state for debugging
    for node_id, node in builder.get_graph().internal_nodes.items():
        branch = session.get_branch(node.branch_id, None)
        if branch:
            print(f"Node {node_id} messages: {len(branch.messages)}")
```

## Configuration

```python
import os

# API Keys
os.environ["OPENAI_API_KEY"] = "your-key"
os.environ["ANTHROPIC_API_KEY"] = "your-key"

# Model Configuration
model = iModel(
    provider="openai",
    model="gpt-4o-mini", 
    temperature=0.7,
    max_tokens=1000
)
```
