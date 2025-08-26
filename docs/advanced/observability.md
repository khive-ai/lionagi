# Observability

Monitor and debug LionAGI workflows.

## Verbose Mode

Enable detailed logging during development:

```python
from lionagi import Session, Builder, Branch, iModel

session = Session()
builder = Builder("debug_test")

branch = Branch(chat_model=iModel(provider="openai", model="gpt-4"))
session.include_branches([branch])

# Build workflow
for topic in ["research", "analysis", "synthesis"]:
    builder.add_operation(
        "communicate",
        branch=branch,
        instruction=f"Brief {topic} on AI trends"
    )

# Execute with verbose logging
result = await session.flow(
    builder.get_graph(),
    verbose=True  # Shows execution details
)

print(f"Completed: {len(result['completed_operations'])}")
print(f"Skipped: {len(result['skipped_operations'])}")
```

## Message Inspection

Track conversation history:

```python
branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="Helpful assistant"
)

topics = ["AI trends", "market analysis", "predictions"]

for topic in topics:
    response = await branch.communicate(f"Brief analysis of {topic}")
    
    print(f"After {topic}:")
    print(f"Total messages: {len(branch.messages)}")
    
    # Show recent exchange
    if len(branch.messages) >= 2:
        user_msg = branch.messages[-2]
        assistant_msg = branch.messages[-1]
        print(f"User: {user_msg.content[:50]}...")
        print(f"Assistant: {assistant_msg.content[:50]}...")
```

## Performance Tracking

Monitor execution time:

```python
import time

session = Session()
builder = Builder("perf_test")

branch = Branch(chat_model=iModel(provider="openai", model="gpt-4"))
session.include_branches([branch])

# Build operations
for i in range(3):
    builder.add_operation(
        "communicate",
        branch=branch,
        instruction=f"Quick analysis {i+1}"
    )

# Execute with timing
start_time = time.time()
result = await session.flow(builder.get_graph(), verbose=True)
execution_time = time.time() - start_time

print(f"Execution time: {execution_time:.2f}s")
print(f"Completed: {len(result['completed_operations'])}")
```

## Error Monitoring

Track failures during execution:

```python
branch = Branch(chat_model=iModel(provider="openai", model="gpt-4"))

test_prompts = ["Normal request", "", "Another request"]
results = {"success": 0, "failed": 0}

for i, prompt in enumerate(test_prompts):
    try:
        result = await branch.communicate(prompt)
        results["success"] += 1
        print(f"✓ Request {i+1} succeeded")
    except Exception as e:
        results["failed"] += 1
        print(f"✗ Request {i+1} failed: {e}")

print(f"Summary: {results['success']} success, {results['failed']} failed")
```

## Best Practices

**Use verbose mode** during development:

```python
# Development
result = await session.flow(graph, verbose=True)

# Production  
result = await session.flow(graph, verbose=False)
```

**Track key metrics**:

- Execution times and success rates
- Message history and memory usage
- Error patterns and frequencies

**Simple monitoring**:

```python
print(f"Completed: {len(result['completed_operations'])}")
print(f"Skipped: {len(result['skipped_operations'])}")
```

LionAGI provides observability through verbose logging, message inspection,
performance tracking, and error monitoring.
