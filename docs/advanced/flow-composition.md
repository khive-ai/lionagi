# Flow Composition

Building complex multi-phase workflows with LionAGI.

## Sequential Flows

Chain operations with dependencies:

```python
from lionagi import Session, Builder, Branch, iModel

session = Session()
builder = Builder("sequential_analysis")

branch = Branch(chat_model=iModel(provider="openai", model="gpt-4"))
session.include_branches([branch])

# Sequential steps
research = builder.add_operation(
    "communicate",
    branch=branch,
    instruction="Research AI market trends"
)

analysis = builder.add_operation(
    "communicate", 
    branch=branch,
    instruction="Analyze the research findings",
    depends_on=[research]
)

recommendations = builder.add_operation(
    "communicate",
    branch=branch,
    instruction="Provide recommendations",
    depends_on=[analysis]
)

result = await session.flow(builder.get_graph())
```

## Parallel Flows

Execute independent operations simultaneously:

```python
session = Session()
builder = Builder("parallel_analysis")

branch = Branch(chat_model=iModel(provider="openai", model="gpt-4"))
session.include_branches([branch])

# Parallel operations
market_op = builder.add_operation(
    "communicate",
    branch=branch,
    instruction="Analyze market conditions"
)

competitor_op = builder.add_operation(
    "communicate",
    branch=branch, 
    instruction="Analyze competitors"
)

tech_op = builder.add_operation(
    "communicate",
    branch=branch,
    instruction="Analyze technology trends"
)

# Aggregate results
synthesis = builder.add_aggregation(
    "communicate",
    branch=branch,
    source_node_ids=[market_op, competitor_op, tech_op],
    instruction="Synthesize all analyses"
)

result = await session.flow(builder.get_graph(), max_concurrent=3)
```

## Multi-Phase Workflows

Mix sequential and parallel patterns:

```python
# Phase 1: Initial research
initial_research = builder.add_operation(
    "communicate",
    branch=branch,
    instruction="Initial market research"
)

# Phase 2: Parallel analysis (depends on Phase 1)
financial_analysis = builder.add_operation(
    "communicate",
    branch=branch,
    instruction="Financial analysis based on research",
    depends_on=[initial_research]
)

market_analysis = builder.add_operation(
    "communicate", 
    branch=branch,
    instruction="Market analysis based on research",
    depends_on=[initial_research]
)

# Phase 3: Synthesis (depends on Phase 2)
final_report = builder.add_aggregation(
    "communicate",
    branch=branch,
    source_node_ids=[financial_analysis, market_analysis],
    instruction="Create comprehensive report"
)

result = await session.flow(builder.get_graph())
```

## Context Inheritance

Pass context between operations:

```python
# Parent operation
parent_op = builder.add_operation(
    "communicate",
    branch=branch,
    instruction="Analyze business requirements"
)

# Child operation inherits context
child_op = builder.add_operation(
    "communicate",
    branch=branch,
    instruction="Provide implementation recommendations",
    depends_on=[parent_op],
    inherit_context=True
)

result = await session.flow(builder.get_graph())
```

## Multi-Branch Flows

Use specialized branches for different tasks:

```python
# Specialized branches
researcher = Branch(
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="Research specialist"
)

analyst = Branch(
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="Data analyst"
)

writer = Branch(
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="Report writer"
)

session.include_branches([researcher, analyst, writer])

# Sequential workflow with different branches
research_op = builder.add_operation(
    "communicate",
    branch=researcher,
    instruction="Research market trends"
)

analysis_op = builder.add_operation(
    "communicate",
    branch=analyst,
    instruction="Analyze research data",
    depends_on=[research_op]
)

report_op = builder.add_operation(
    "communicate",
    branch=writer,
    instruction="Write executive summary",
    depends_on=[analysis_op]
)

result = await session.flow(builder.get_graph())
```

## Best Practices

**Start simple** with linear flows before adding complexity:

```python
# Good: Clear progression
research -> analysis -> report

# Avoid: Over-complex initial design
research -> [analysis1, analysis2, analysis3] -> synthesis -> validation -> report
```

**Use aggregation** to combine parallel results:

```python
synthesis = builder.add_aggregation(
    "communicate",
    source_node_ids=[op1, op2, op3],
    instruction="Combine all results"
)
```

**Control concurrency** to manage resource usage:

```python
result = await session.flow(
    builder.get_graph(),
    max_concurrent=3
)
```

Flow composition enables sophisticated workflows by combining sequential
dependencies, parallel execution, and multi-branch coordination.
