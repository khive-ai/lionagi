# Migrating from LangGraph to LionAGI

A practical guide for LangGraph users to leverage LionAGI's superior
orchestration capabilities.

## Why Consider Migration?

LangGraph is powerful but can be complex for multi-agent workflows. LionAGI
offers:

- **Simpler abstractions**: Less boilerplate, cleaner code
- **Parallel by default**: Automatic concurrency without complex state
  management
- **Better production features**: Built-in monitoring, error handling,
  performance control
- **Framework agnostic**: Orchestrate LangGraph alongside other tools

## Migration Approaches

### Option 1: Gradual Migration (Recommended)

Keep your existing LangGraph workflows and orchestrate them with LionAGI:

```python
from lionagi import Session, Builder
from your_langgraph_code import existing_workflow

# Keep your LangGraph workflow unchanged
async def langgraph_research_workflow(branch, query: str, **kwargs):
    """Wrap your existing LangGraph workflow"""
    # Your existing LangGraph code - no changes needed!
    result = await existing_workflow.invoke({"query": query})
    return result["output"]

# Orchestrate with LionAGI
session = Session()
builder = Builder("hybrid_workflow")

# Use existing LangGraph workflow as custom operation
research_op = builder.add_operation(
    operation=langgraph_research_workflow,
    query="Market analysis request"
)

# Add pure LionAGI operations
analysis_branch = session.new_branch(system="Analysis specialist")
analysis_op = builder.add_operation(
    "communicate",
    branch=analysis_branch,
    instruction="Analyze the research findings",
    depends_on=[research_op]
)

result = await session.flow(builder.get_graph())
```

### Option 2: Direct Translation

Translate LangGraph patterns to LionAGI equivalents:

#### LangGraph Supervisor Pattern

```python
# LangGraph approach
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str

def supervisor_agent(state):
    # Complex routing logic with manual state management
    response = llm.invoke(state["messages"])
    return {"next": response.content, "messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_conditional_edges("supervisor", route_function, routing_map)
```

#### LionAGI Equivalent

```python
# LionAGI approach - much simpler
session = Session()
builder = Builder("coordinated_workflow")

# Create specialized agents
researcher = session.new_branch(system="Research specialist")
analyst = session.new_branch(system="Analysis specialist") 
writer = session.new_branch(system="Report writer")

# Sequential workflow with automatic coordination
research_op = builder.add_operation(
    "communicate", branch=researcher,
    instruction="Research the topic thoroughly"
)

analysis_op = builder.add_operation(
    "communicate", branch=analyst,
    instruction="Analyze research findings", 
    depends_on=[research_op]
)

report_op = builder.add_operation(
    "communicate", branch=writer,
    instruction="Write executive summary",
    depends_on=[analysis_op]
)

result = await session.flow(builder.get_graph())
```

**Analysis**: The LionAGI version eliminates state classes, routing functions, and manual edge configuration. Instead of 15+ lines of setup code, you get a natural workflow that reads like business logic. The `depends_on` parameter automatically handles execution order, while LangGraph requires explicit edge configuration between every node.

## Common Migration Patterns

### 1. State Management → Memory Management

#### LangGraph State Handling

```python
class AgentState(TypedDict):
    messages: List[BaseMessage]
    research_data: str
    analysis_result: str

def research_node(state: AgentState):
    # Manual state updates
    result = research_function()
    return {"research_data": result, "messages": state["messages"] + [result]}
```

#### LionAGI Memory Management

```python
# Automatic memory management per branch
researcher = Branch(system="Research specialist")

# Memory is handled automatically
research_result = await researcher.communicate("Research market trends")
follow_up = await researcher.communicate("What are the key risks?")  # Has context
```

### 2. Conditional Routing → Dependencies

#### LangGraph Conditional Logic

```python
def should_continue(state):
    if state["confidence"] > 0.8:
        return "high_confidence_path"
    else:
        return "low_confidence_path"

workflow.add_conditional_edges("analysis", should_continue, {
    "high_confidence_path": "final_report",
    "low_confidence_path": "additional_research"
})
```

#### LionAGI Dependency Management

```python
# Use simple dependencies and aggregation
initial_analysis = builder.add_operation("communicate", instruction="Initial analysis")

# Alternative paths based on results  
detailed_research = builder.add_operation(
    "communicate",
    instruction="Conduct detailed research if needed",
    depends_on=[initial_analysis]
)

final_report = builder.add_aggregation(
    "communicate",
    source_node_ids=[initial_analysis, detailed_research],
    instruction="Create final report based on all analysis"
)
```

### 3. Manual Parallelism → Automatic Parallelism

#### LangGraph Parallel Execution

```python
# Complex parallel setup
def run_parallel_tasks(state):
    tasks = [research_task, analysis_task, review_task]
    # Manual coordination required
    results = asyncio.gather(*tasks)
    return {"results": results}
```

#### LionAGI Automatic Parallelism

```python
# Automatic parallel execution - no dependencies = parallel
research_op = builder.add_operation("communicate", instruction="Research")
analysis_op = builder.add_operation("communicate", instruction="Analyze") 
review_op = builder.add_operation("communicate", instruction="Review")

# These run in parallel automatically
# Synthesis waits for all to complete
synthesis = builder.add_aggregation(
    "communicate",
    source_node_ids=[research_op, analysis_op, review_op],
    instruction="Combine all findings"
)
```

## Migration Benefits

### 1. Reduced Complexity

**Before (LangGraph)**:

- Manual state management
- Complex routing logic
- Verbose graph setup

**After (LionAGI)**:

- Automatic memory management
- Simple dependencies
- Clean abstractions

### 2. Better Performance

```python
# LionAGI: Built-in performance controls
result = await session.flow(
    builder.get_graph(),
    max_concurrent=5,  # Control parallelism
    verbose=True      # Built-in monitoring
)
```

### 3. Production Features

```python
# LionAGI: Built-in error handling and monitoring
try:
    result = await session.flow(builder.get_graph())
    print(f"Completed: {len(result['completed_operations'])}")
except Exception as e:
    print(f"Workflow failed: {e}")
    # Built-in fallback handling
```

## Step-by-Step Migration Guide

### Step 1: Analyze Your LangGraph Workflow

1. Identify your agents/nodes
2. Map out state dependencies
3. Note any parallel operations
4. Identify conditional logic

### Step 2: Choose Migration Strategy

**For complex workflows**: Start with gradual migration (wrap existing code)
**For simple workflows**: Direct translation to LionAGI

### Step 3: Create LionAGI Equivalent

```python
# Template for most LangGraph migrations
session = Session()
builder = Builder("migrated_workflow")

# Create branches for each LangGraph node
agent1 = session.new_branch(system="Agent 1 role")
agent2 = session.new_branch(system="Agent 2 role")

# Convert nodes to operations with dependencies
op1 = builder.add_operation("communicate", branch=agent1, instruction="Task 1")
op2 = builder.add_operation("communicate", branch=agent2, instruction="Task 2", depends_on=[op1])

# Execute with better performance and monitoring
result = await session.flow(builder.get_graph(), max_concurrent=3)
```

### Step 4: Test and Optimize

1. Compare outputs between old and new systems
2. Optimize concurrency settings
3. Add error handling
4. Monitor performance

## Migration Checklist

- [ ] Map LangGraph nodes to LionAGI branches
- [ ] Convert state management to dependencies
- [ ] Identify parallel operations
- [ ] Add error handling
- [ ] Test output equivalence
- [ ] Optimize performance settings
- [ ] Add monitoring/observability

## When Not to Migrate

Consider keeping LangGraph if:

- You have simple, working workflows that don't need orchestration
- Your use case doesn't require parallel execution
- You're heavily invested in LangChain ecosystem features

## Best of Both Worlds

Remember: You don't have to choose! LionAGI can orchestrate your existing
LangGraph workflows alongside other tools:

```python
# Orchestrate everything together
langgraph_op = builder.add_operation(operation=existing_langgraph_workflow)
crewai_op = builder.add_operation(operation=existing_crewai_workflow)  
lionagi_op = builder.add_operation("communicate", instruction="Pure LionAGI task")

# LionAGI orchestrates all frameworks
result = await session.flow(builder.get_graph())
```

LionAGI enhances your existing investments rather than replacing them.

## Key Advantages

### 1. **Simplicity**

**LangGraph**: Complex state management, manual routing, verbose setup
**LionAGI**: Clean abstractions, automatic orchestration, minimal boilerplate

### 2. **Parallel Execution**

**LangGraph**: Sequential by design, complex to parallelize **LionAGI**:
Parallel by default, automatic dependency resolution

```python
# LionAGI: Automatic parallelization
research = builder.add_operation("communicate", branch=researcher, instruction="Research")
analysis = builder.add_operation("communicate", branch=analyst, instruction="Analyze")
# Both run in parallel automatically
```

### 3. **Memory Management**

- **LangGraph**: Manual state passing, complex message handling
- **LionAGI**: Automatic memory management per branch

```python
# LionAGI: Each branch maintains independent memory
researcher.communicate("First question")
researcher.communicate("Follow up")  # Automatically has context
```

### 4. **Framework Agnostic**

**LangGraph**: Locked into LangChain ecosystem **LionAGI**: Can orchestrate ANY
framework as custom operations

```python
# Wrap existing LangGraph workflow as LionAGI operation
async def langgraph_operation(branch, **kwargs):
    # Your existing LangGraph code here - no changes needed!
    return await your_langgraph_workflow.invoke(kwargs)

# Use in LionAGI orchestration
builder.add_operation(operation=langgraph_operation, ...)
```

### 5. **Production Ready**

**LangGraph**: Complex debugging, hard to monitor **LionAGI**: Built-in
observability, error handling, performance control

```python
# LionAGI: Built-in monitoring and control
result = await session.flow(
    builder.get_graph(),
    max_concurrent=5,  # Control parallelism
    verbose=True      # Built-in monitoring
)
```

## Migration Path

Don't throw away your LangGraph investment! LionAGI can orchestrate your
existing LangGraph workflows:

```python
from lionagi import Session, Builder

# Keep your existing LangGraph workflow
async def existing_langgraph_workflow(input_data):
    # Your current LangGraph code - unchanged!
    return langgraph_result

# Orchestrate with LionAGI
session = Session()
builder = Builder("hybrid_workflow")

# LangGraph workflow as custom operation
lg_op = builder.add_operation(
    operation=existing_langgraph_workflow,
    input_data="market research"
)

# Pure LionAGI operations
analysis_op = builder.add_operation(
    "communicate",
    branch=analyst,
    instruction="Analyze the research results",
    depends_on=[lg_op]
)

# Get best of both worlds
result = await session.flow(builder.get_graph())
```

## Performance Comparison

| Feature               | LangGraph      | LionAGI   |
| --------------------- | -------------- | --------- |
| Parallel Execution    | Manual/Complex | Automatic |
| Setup Complexity      | High           | Low       |
| State Management      | Manual         | Automatic |
| Error Handling        | Manual         | Built-in  |
| Framework Lock-in     | Yes            | No        |
| Migration Cost        | High           | Zero      |
| Debugging             | Complex        | Simple    |
| Production Monitoring | Manual         | Built-in  |

## When to Choose LionAGI

- **Parallel workflows**: LionAGI excels at concurrent execution
- **Complex orchestration**: Multiple agents, dependencies, synthesis
- **Production systems**: Built-in monitoring, error handling, performance
  control
- **Multi-framework**: Orchestrate LangChain, CrewAI, AutoGen together
- **Enterprise**: Clean architecture, maintainable code, team scalability

LionAGI doesn't replace your existing investments - it orchestrates them better.
