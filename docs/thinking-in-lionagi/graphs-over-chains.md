# Graphs Over Chains

Why directed acyclic graphs beat sequential chains for agent orchestration.

## The Limitation of Chains

Sequential chains force linear execution even when tasks could run in parallel.

```python
# Traditional chain approach (conceptual)
async def chain_workflow():
    """Sequential chain - inefficient for parallel tasks"""
    
    # Each step waits for previous to complete
    step1 = await research_task("market analysis")     # 30 seconds
    step2 = await research_task("competitor analysis") # 30 seconds  
    step3 = await research_task("trend analysis")      # 30 seconds
    
    # Total time: 90 seconds (sequential)
    synthesis = await synthesize_results([step1, step2, step3])
    return synthesis

# These tasks could have run in parallel!
```

## Graph-Based Execution

Graphs enable parallel execution with proper dependencies.

```python
from lionagi import Session, Builder, Branch, iModel
import asyncio

async def graph_workflow():
    """Graph-based execution - parallel where possible"""
    session = Session()
    builder = Builder("parallel_research")
    
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    session.include_branches([branch])
    
    # Independent parallel operations
    market = builder.add_operation(
        "communicate",
        branch=branch,
        instruction="Research market analysis"
    )
    
    competitor = builder.add_operation(
        "communicate",
        branch=branch, 
        instruction="Research competitor analysis"
    )
    
    trends = builder.add_operation(
        "communicate",
        branch=branch,
        instruction="Research trend analysis"  
    )
    
    # Synthesis depends on all three (proper dependency)
    synthesis = builder.add_aggregation(
        "communicate",
        branch=branch,
        source_node_ids=[market, competitor, trends],
        instruction="Synthesize all research findings"
    )
    
    # Execute with parallelism - total time: ~30 seconds
    result = await session.flow(builder.get_graph(), max_concurrent=3)
    return result

asyncio.run(graph_workflow())
```

## Complex Dependencies

Graphs handle complex dependency patterns naturally.

```python
async def complex_graph():
    """Complex dependency graph"""
    session = Session()
    builder = Builder("complex_analysis")
    
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    session.include_branches([branch])
    
    # Phase 1: Initial research (parallel)
    market = builder.add_operation(
        "communicate", branch=branch,
        instruction="Market research"
    )
    
    tech = builder.add_operation(
        "communicate", branch=branch,
        instruction="Technology research"
    )
    
    # Phase 2: Analysis (depends on Phase 1)
    market_analysis = builder.add_operation(
        "communicate", branch=branch,
        instruction="Analyze market data",
        depends_on=[market]  # Waits for market research
    )
    
    tech_analysis = builder.add_operation(
        "communicate", branch=branch,
        instruction="Analyze technology trends", 
        depends_on=[tech]  # Waits for tech research
    )
    
    # Phase 3: Risk assessment (depends on both analyses)
    risk = builder.add_operation(
        "communicate", branch=branch,
        instruction="Assess combined risks",
        depends_on=[market_analysis, tech_analysis]  # Waits for both
    )
    
    # Phase 4: Final strategy (depends on everything)
    strategy = builder.add_aggregation(
        "communicate", branch=branch,
        source_node_ids=[market_analysis, tech_analysis, risk],
        instruction="Create final strategy"
    )
    
    result = await session.flow(builder.get_graph())
    return result

asyncio.run(complex_graph())
```

## Conditional Graph Paths

Graphs can represent conditional execution paths.

```python
async def conditional_graph():
    """Graph with conditional branches"""
    session = Session()
    builder = Builder("conditional_workflow")
    
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    session.include_branches([branch])
    
    # Initial assessment
    assessment = builder.add_operation(
        "communicate", branch=branch,
        instruction="Assess project complexity (simple/complex)"
    )
    
    # Simple path
    simple_plan = builder.add_operation(
        "communicate", branch=branch,
        instruction="Create simple implementation plan",
        depends_on=[assessment]
    )
    
    # Complex path  
    detailed_research = builder.add_operation(
        "communicate", branch=branch,
        instruction="Detailed technical research",
        depends_on=[assessment]
    )
    
    complex_plan = builder.add_operation(
        "communicate", branch=branch,
        instruction="Create complex implementation plan",
        depends_on=[detailed_research]
    )
    
    # Both paths can execute - actual execution depends on conditions
    result = await session.flow(builder.get_graph())
    return result

asyncio.run(conditional_graph())
```

## Fan-Out/Fan-In Pattern

Common graph pattern for parallel processing and aggregation.

```python
async def fan_out_fan_in():
    """Fan-out to parallel processing, fan-in to aggregation"""
    session = Session()
    builder = Builder("fan_pattern")
    
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    session.include_branches([branch])
    
    # Fan-out: Single input spawns multiple parallel tasks
    initial = builder.add_operation(
        "communicate", branch=branch,
        instruction="Define research scope"
    )
    
    # Parallel analysis tasks (fan-out)
    analyses = []
    topics = ["market", "technical", "financial", "legal", "competitive"]
    
    for topic in topics:
        analysis = builder.add_operation(
            "communicate", branch=branch,
            instruction=f"Analyze {topic} aspects",
            depends_on=[initial]  # All depend on scope definition
        )
        analyses.append(analysis)
    
    # Fan-in: Aggregate all parallel results
    final_report = builder.add_aggregation(
        "communicate", branch=branch,
        source_node_ids=analyses,  # Collect all analyses
        instruction="Create comprehensive final report"
    )
    
    result = await session.flow(builder.get_graph(), max_concurrent=5)
    return result

asyncio.run(fan_out_fan_in())
```

## Graph vs Chain Comparison

When to use graphs vs direct execution.

```python
# Use graphs for: Complex workflows with dependencies
async def use_graphs():
    # Multiple phases with mixed parallel/sequential execution
    # Dependencies between operations
    # Need for aggregation or synthesis
    builder = Builder("complex")
    # ... build graph with dependencies
    return await session.flow(builder.get_graph())

# Use direct execution for: Simple single operations  
async def use_direct():
    branch = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
    return await branch.communicate("Simple task")

# Use asyncio.gather for: Independent parallel operations
async def use_gather():
    return await asyncio.gather(
        branch.communicate("Task 1"),
        branch.communicate("Task 2"), 
        branch.communicate("Task 3")
    )
```

## Best Practices

### 1. Design for Parallelism

```python
# Good: Parallel where possible
market_op = builder.add_operation(...)    # No depends_on
tech_op = builder.add_operation(...)      # No depends_on  
synthesis = builder.add_aggregation(..., source_node_ids=[market_op, tech_op])

# Avoid: Unnecessary sequential dependencies
step2 = builder.add_operation(..., depends_on=[step1])  # Only if truly needed
```

### 2. Use Aggregation for Synthesis

```python
# Combine multiple parallel results
synthesis = builder.add_aggregation(
    "communicate",
    source_node_ids=[op1, op2, op3],
    instruction="Synthesize all findings"
)
```

### 3. Control Concurrency

```python
# Limit parallel execution to avoid overwhelming APIs
result = await session.flow(
    builder.get_graph(),
    max_concurrent=3  # Reasonable limit
)
```

### 4. Keep Dependencies Simple

```python
# Good: Clear, necessary dependencies
analysis = builder.add_operation(..., depends_on=[research])

# Avoid: Complex circular or unnecessary dependencies
# operation = builder.add_operation(..., depends_on=[many, complex, deps])
```

Graphs in LionAGI enable sophisticated execution patterns through parallel
processing, proper dependency management, and flexible workflow topologies that
sequential chains cannot achieve.
