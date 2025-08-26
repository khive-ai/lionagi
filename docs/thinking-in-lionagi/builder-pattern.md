# The Builder Pattern

Build complex workflows incrementally with dependencies and parallel execution.

## Simple Builder Example

Create operations and define their execution order.

```python
from lionagi import Session, Builder, Branch, iModel
import asyncio

async def basic_builder():
    """Simple workflow with Builder pattern"""
    session = Session()
    builder = Builder("analysis")
    
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    session.include_branches([branch])
    
    # Add operations
    research = builder.add_operation(
        "communicate",
        branch=branch,
        instruction="Research market trends"
    )
    
    analysis = builder.add_operation(
        "communicate", 
        branch=branch,
        instruction="Analyze the research findings",
        depends_on=[research]  # Runs after research
    )
    
    # Execute workflow
    result = await session.flow(builder.get_graph())
    return result

asyncio.run(basic_builder())
```

## Parallel Operations

Execute independent operations simultaneously.

```python
async def parallel_builder():
    """Parallel operations with aggregation"""
    session = Session()
    builder = Builder("parallel_analysis")
    
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    session.include_branches([branch])
    
    # Parallel operations (no dependencies)
    market_op = builder.add_operation(
        "communicate",
        branch=branch,
        instruction="Analyze market conditions"
    )
    
    competitor_op = builder.add_operation(
        "communicate",
        branch=branch, 
        instruction="Research competitors"
    )
    
    tech_op = builder.add_operation(
        "communicate",
        branch=branch,
        instruction="Evaluate technology trends"
    )
    
    # Combine results
    synthesis = builder.add_aggregation(
        "communicate",
        branch=branch,
        source_node_ids=[market_op, competitor_op, tech_op],
        instruction="Synthesize all analyses"
    )
    
    result = await session.flow(builder.get_graph(), max_concurrent=3)
    return result

asyncio.run(parallel_builder())
```

## Multi-Phase Workflows

Combine sequential and parallel patterns.

```python
async def multi_phase_builder():
    """Multi-phase workflow with mixed patterns"""
    session = Session()
    builder = Builder("complex_analysis")
    
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    session.include_branches([branch])
    
    # Phase 1: Initial research
    initial = builder.add_operation(
        "communicate",
        branch=branch,
        instruction="Initial market research"
    )
    
    # Phase 2: Parallel analysis (depends on Phase 1)
    financial = builder.add_operation(
        "communicate",
        branch=branch,
        instruction="Financial analysis",
        depends_on=[initial]
    )
    
    technical = builder.add_operation(
        "communicate", 
        branch=branch,
        instruction="Technical analysis",
        depends_on=[initial]
    )
    
    # Phase 3: Final synthesis (depends on Phase 2)
    final = builder.add_aggregation(
        "communicate",
        branch=branch,
        source_node_ids=[financial, technical],
        instruction="Create final report"
    )
    
    result = await session.flow(builder.get_graph())
    return result

asyncio.run(multi_phase_builder())
```

## Multiple Branches

Use specialized branches for different types of work.

```python
async def multi_branch_builder():
    """Workflow with specialized branches"""
    session = Session()
    builder = Builder("specialized_workflow")
    
    # Specialized branches
    researcher = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="Research specialist"
    )
    
    analyst = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="Data analyst"
    )
    
    writer = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="Report writer"
    )
    
    session.include_branches([researcher, analyst, writer])
    
    # Research phase
    research = builder.add_operation(
        "communicate",
        branch=researcher,
        instruction="Research AI market trends"
    )
    
    # Analysis phase
    analysis = builder.add_operation(
        "communicate",
        branch=analyst,
        instruction="Analyze market data",
        depends_on=[research]
    )
    
    # Writing phase
    report = builder.add_operation(
        "communicate",
        branch=writer,
        instruction="Write executive summary",
        depends_on=[analysis]
    )
    
    result = await session.flow(builder.get_graph())
    return result

asyncio.run(multi_branch_builder())
```

## Builder vs Direct Execution

When to use Builder vs direct calls.

```python
# Use Builder for: Complex workflows with dependencies
async def use_builder():
    builder = Builder("workflow")
    # Multiple operations with dependencies
    op1 = builder.add_operation("communicate", branch=branch, instruction="Step 1")
    op2 = builder.add_operation("communicate", branch=branch, instruction="Step 2", depends_on=[op1])
    return await session.flow(builder.get_graph())

# Use direct calls for: Simple single operations
async def use_direct():
    branch = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
    result = await branch.communicate("Simple analysis")
    return result

# Use asyncio.gather for: Independent parallel operations
async def use_gather():
    results = await asyncio.gather(
        branch.communicate("Task 1"),
        branch.communicate("Task 2"),
        branch.communicate("Task 3")
    )
    return results
```

## Best Practices

### 1. Start Simple

```python
# Good: Clear linear flow
research -> analysis -> report

# Avoid: Over-complex initial design
research -> [analysis1, analysis2, analysis3] -> synthesis -> validation
```

### 2. Use Dependencies Wisely

```python
# Sequential: Each step depends on previous
step2 = builder.add_operation(..., depends_on=[step1])
step3 = builder.add_operation(..., depends_on=[step2])

# Parallel: No dependencies, runs simultaneously
op1 = builder.add_operation(...)  # No depends_on
op2 = builder.add_operation(...)  # No depends_on
```

### 3. Aggregate Parallel Results

```python
# Combine parallel operations
synthesis = builder.add_aggregation(
    "communicate",
    source_node_ids=[op1, op2, op3],
    instruction="Combine all results"
)
```

### 4. Control Concurrency

```python
# Limit parallel execution
result = await session.flow(
    builder.get_graph(),
    max_concurrent=3  # Only 3 operations at once
)
```

The Builder pattern in LionAGI enables sophisticated workflows through
incremental construction, dependency management, and controlled parallel
execution.
