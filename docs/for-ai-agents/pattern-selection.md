# Pattern Selection Guide

Choosing the right orchestration pattern for the task.

## Quick Decision Framework

```python
# Single simple task → Direct execution
if single_task and simple:
    result = await branch.communicate(instruction)

# Multiple independent tasks → asyncio.gather
if multiple_tasks and independent:
    results = await asyncio.gather(*[branch.communicate(task) for task in tasks])

# Complex workflow with dependencies → Builder
if dependencies or aggregation_needed:
    builder = Builder("workflow")
    # ... build graph
    result = await session.flow(builder.get_graph())
```

## Pattern Characteristics

### Direct Execution

Best for single, straightforward tasks.

```python
from lionagi import Branch, iModel
import asyncio

async def direct_pattern():
    """Use for: Single analysis, simple questions, quick tasks"""
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    
    # Perfect for single operations
    result = await branch.communicate("Analyze this market trend")
    return result

asyncio.run(direct_pattern())
```

**When to use:**

- Single analysis or question
- No dependencies on other operations
- Quick, standalone tasks
- Conversational interactions

### Parallel Execution (asyncio.gather)

Best for multiple independent tasks.

```python
async def parallel_pattern():
    """Use for: Independent parallel tasks, bulk processing"""
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    
    tasks = [
        "Analyze market trends",
        "Research competitors", 
        "Evaluate risks",
        "Assess opportunities"
    ]
    
    # Perfect for independent parallel work
    results = await asyncio.gather(*[
        branch.communicate(task) for task in tasks
    ])
    
    return results

asyncio.run(parallel_pattern())
```

**When to use:**

- Multiple independent tasks
- No dependencies between operations
- Bulk processing
- Maximum parallelism needed

### Builder Graphs

Best for complex workflows with dependencies.

```python
from lionagi import Session, Builder

async def builder_pattern():
    """Use for: Dependencies, aggregation, complex workflows"""
    session = Session()
    builder = Builder("analysis")
    
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    session.include_branches([branch])
    
    # Step 1: Research
    research = builder.add_operation(
        "communicate", branch=branch,
        instruction="Research market conditions"
    )
    
    # Step 2: Analysis (depends on research)
    analysis = builder.add_operation(
        "communicate", branch=branch,
        instruction="Analyze research findings",
        depends_on=[research]
    )
    
    # Step 3: Synthesis
    synthesis = builder.add_aggregation(
        "communicate", branch=branch,
        source_node_ids=[research, analysis],
        instruction="Create final report"
    )
    
    result = await session.flow(builder.get_graph())
    return result

asyncio.run(builder_pattern())
```

**When to use:**

- Operations depend on each other
- Need to aggregate/synthesize results
- Multi-phase workflows
- Complex coordination required

## Pattern Comparison

| Pattern     | Best For                        | Avoid When          | Complexity | Performance |
| ----------- | ------------------------------- | ------------------- | ---------- | ----------- |
| **Direct**  | Single tasks, conversations     | Multiple operations | Low        | Fast        |
| **Gather**  | Independent parallel tasks      | Dependencies exist  | Medium     | Very Fast   |
| **Builder** | Complex workflows, dependencies | Simple single tasks | High       | Optimized   |

## Selection Examples

### Example 1: Simple Analysis

```python
# Task: "Analyze this code for security issues"
# Pattern: Direct execution

async def security_analysis():
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="Security expert"
    )
    return await branch.communicate("Analyze this code for security issues")
```

### Example 2: Multiple Independent Reviews

```python
# Task: Review code for security, performance, style
# Pattern: Parallel execution

async def multi_review():
    security = Branch(system="Security reviewer")
    performance = Branch(system="Performance reviewer") 
    style = Branch(system="Style reviewer")
    
    results = await asyncio.gather(
        security.communicate("Review security"),
        performance.communicate("Review performance"),
        style.communicate("Review style")
    )
    return results
```

### Example 3: Research → Analysis → Report

```python
# Task: Multi-step workflow with dependencies
# Pattern: Builder graph

async def research_workflow():
    session = Session()
    builder = Builder("research")
    
    branch = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
    session.include_branches([branch])
    
    research = builder.add_operation(
        "communicate", branch=branch,
        instruction="Research topic"
    )
    
    analysis = builder.add_operation(
        "communicate", branch=branch,
        instruction="Analyze findings",
        depends_on=[research]
    )
    
    report = builder.add_operation(
        "communicate", branch=branch,
        instruction="Write final report", 
        depends_on=[analysis]
    )
    
    return await session.flow(builder.get_graph())
```

## Decision Tree

```
Is it a single operation?
├── Yes → Use Direct Execution
└── No → Are operations independent?
    ├── Yes → Use asyncio.gather
    └── No → Do you need aggregation/synthesis?
        ├── Yes → Use Builder with aggregation
        └── No → Use Builder with dependencies
```

## Common Mistakes

### Over-engineering Simple Tasks

```python
# Bad: Builder for single task
builder = Builder("simple")
op = builder.add_operation("communicate", branch=branch, instruction="Hello")
result = await session.flow(builder.get_graph())

# Good: Direct execution
result = await branch.communicate("Hello")
```

### Missing Parallelism Opportunities

```python
# Bad: Sequential when could be parallel
result1 = await branch.communicate("Task 1")
result2 = await branch.communicate("Task 2")  # Waits for Task 1

# Good: Parallel independent tasks
results = await asyncio.gather(
    branch.communicate("Task 1"),
    branch.communicate("Task 2")  # Runs in parallel
)
```

### Forcing Dependencies

```python
# Bad: Unnecessary dependencies
step2 = builder.add_operation(..., depends_on=[step1])  # Not needed

# Good: Parallel when possible
step1 = builder.add_operation(...)  # No depends_on
step2 = builder.add_operation(...)  # No depends_on
synthesis = builder.add_aggregation(..., source_node_ids=[step1, step2])
```

## Fallback Strategy

When unsure, start simple and evolve:

```python
# 1. Start with direct execution
result = await branch.communicate(instruction)

# 2. If you need multiple independent tasks, upgrade to gather
results = await asyncio.gather(*tasks)

# 3. If you need dependencies or aggregation, upgrade to Builder
builder = Builder("workflow")
# ... add operations with dependencies
result = await session.flow(builder.get_graph())
```

## Performance Guidelines

- **Direct**: Fastest for single operations
- **Gather**: Fastest for independent parallel operations
- **Builder**: Optimized for complex workflows, handles concurrency limits

Choose based on your specific coordination needs, not just performance.
