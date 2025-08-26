# Custom Operations

Creating specialized operations for LionAGI workflows.

## Core Concept

Operations are building blocks that branches execute. LionAGI provides built-in
operations like `chat`, `communicate`, `operate`, and `ReAct`, but you can
create custom ones for specialized tasks.

## Built-in Operations

- **chat**: Basic conversation
- **communicate**: Stateful conversation
- **operate**: Structured output with Pydantic
- **ReAct**: Reasoning with tools

## Creating Custom Operations

### Function-Based Operations

Define async functions for custom behavior:

```python
from lionagi import Branch, Builder, Session

async def summarize_with_keywords(branch: Branch, instruction: str, keywords: list = None, **kwargs):
    """Custom operation that emphasizes specific keywords."""
    keyword_text = f"Focus on: {', '.join(keywords or [])}"
    enhanced_instruction = f"{instruction}\n\n{keyword_text}"
    
    return await branch.chat(enhanced_instruction, **kwargs)

# Usage in workflow
session = Session()
builder = Builder("custom_workflow")

node = builder.add_operation(
    operation=summarize_with_keywords,
    instruction="Summarize this research paper",
    keywords=["machine learning", "performance"]
)

result = await session.flow(builder.get_graph())
```

### Class-Based Operations

For stateful operations, use classes:

```python
class DataAnalysisOperation:
    def __init__(self, analysis_type: str = "descriptive"):
        self.analysis_type = analysis_type
    
    async def __call__(self, branch: Branch, data_context: str, **kwargs):
        instruction = f"""
        Perform {self.analysis_type} analysis on: {data_context}
        
        Provide:
        1. Key findings
        2. Statistical insights  
        3. Recommendations
        """
        
        return await branch.chat(instruction, **kwargs)

# Usage
analyzer = DataAnalysisOperation("predictive")

builder.add_operation(
    operation=analyzer,
    data_context="Sales data Q1-Q3 2024"
)
```

## Integration Patterns

### Sequential Dependencies

```python
# Operations run in order
data_load = builder.add_operation(
    operation="communicate",
    instruction="Load and validate dataset"
)

analyze = builder.add_operation(
    operation=analyzer,
    depends_on=[data_load],
    data_context="Use loaded data"
)
```

### Parallel with Aggregation

```python
# Multiple analyses run in parallel
analysis_nodes = []
for analysis_type in ["descriptive", "predictive"]:
    node = builder.add_operation(
        operation=DataAnalysisOperation(analysis_type),
        data_context="Sales data"
    )
    analysis_nodes.append(node)

# Combine results
summary = builder.add_aggregation(
    operation="communicate",
    source_node_ids=analysis_nodes,
    instruction="Combine analyses into executive summary"
)
```

## Best Practices

- **Keep operations focused** on single responsibilities
- **Use existing LionAGI operations** (`chat`, `communicate`) when possible
- **Handle errors gracefully** with try/catch in custom logic
- **Test operations independently** before integrating into workflows
