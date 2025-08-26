# Performance Optimization

Making LionAGI workflows fast and efficient.

## Parallel Execution

Execute multiple branches simultaneously:

```python
from lionagi import Session, Branch, iModel
import asyncio

session = Session()

# Multiple branches for parallel processing
researcher = Branch(
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="Research specialist"
)
analyst = Branch(
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="Data analyst"
)

session.include_branches([researcher, analyst])

# Execute in parallel
results = await asyncio.gather(
    researcher.communicate("Research market trends"),
    analyst.communicate("Analyze competitive landscape")
)
```

## Concurrency Control

Control parallel execution with `max_concurrent`:

```python
from lionagi import Session, Builder, Branch, iModel

session = Session()
builder = Builder("performance_test")

branch = Branch(chat_model=iModel(provider="openai", model="gpt-4"))
session.include_branches([branch])

# Create multiple operations
topics = ["AI trends", "market analysis", "competition"]
for topic in topics:
    builder.add_operation(
        "communicate",
        branch=branch,
        instruction=f"Brief analysis of {topic}"
    )

# Execute with controlled concurrency
result = await session.flow(
    builder.get_graph(),
    max_concurrent=2  # Only 2 operations at once
)
```

## Token Efficiency

Use appropriate models for different tasks:

```python
# Light model for simple tasks
classifier = Branch(
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="Classify content briefly"
)

# Powerful model for complex analysis
analyzer = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-5-sonnet-20240620"),
    system="Provide detailed analysis"
)

content = "Sample content to process"

# Step 1: Quick classification
category = await classifier.communicate(f"Category (simple/complex): {content}")

# Step 2: Use appropriate model based on complexity
if "complex" in category.lower():
    analysis = await analyzer.communicate(f"Detailed analysis: {content}")
else:
    analysis = await classifier.communicate(f"Brief analysis: {content}")
```

## Batch Processing

Process multiple items efficiently:

```python
branch = Branch(chat_model=iModel(provider="openai", model="gpt-4"))

items = [f"item_{i}" for i in range(20)]
batch_size = 5
results = []

for i in range(0, len(items), batch_size):
    batch = items[i:i + batch_size]
    
    # Process batch in parallel
    batch_results = await asyncio.gather(*[
        branch.communicate(f"Process: {item}")
        for item in batch
    ])
    
    results.extend(batch_results)
```

## Memory Management

Clear message history in long workflows:

```python
branch = Branch(chat_model=iModel(provider="openai", model="gpt-4"))

large_dataset = [f"document_{i}" for i in range(100)]
chunk_size = 10
results = []

for i in range(0, len(large_dataset), chunk_size):
    chunk = large_dataset[i:i + chunk_size]
    
    # Process chunk
    chunk_result = await branch.communicate(
        f"Summarize these {len(chunk)} documents: {chunk}"
    )
    results.append(chunk_result)
    
    # Clear message history to free memory
    branch.messages.clear()
```

## Best Practices

**Choose appropriate patterns** for your use case:

```python
# Simple parallel tasks: Use asyncio.gather()
results = await asyncio.gather(*[branch.communicate(task) for task in tasks])

# Complex workflows: Use Builder + session.flow()
result = await session.flow(builder.get_graph(), max_concurrent=5)
```

**Control concurrency** based on your needs:

- Start with `max_concurrent=3-5`
- Adjust based on API rate limits
- Monitor for optimal settings

**Optimize token usage**:

- Use appropriate models for task complexity
- Clear message history when context not needed
- Batch similar operations together

**Monitor performance**:

```python
# Use verbose mode for insights
result = await session.flow(graph, verbose=True)

# Track metrics
print(f"Completed: {len(result['completed_operations'])}")
```

Performance optimization focuses on parallel execution, concurrency control, and
efficient resource usage.
