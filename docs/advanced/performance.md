# Performance

lionagi provides structured concurrency primitives in
`lionagi.ln.concurrency` and built-in rate limiting in `iModel` to help
you maximize throughput without overwhelming API providers.

## Concurrency Primitives

All primitives are built on [AnyIO](https://anyio.readthedocs.io/) and
work with both asyncio and trio backends.

### gather -- Concurrent Execution

Run multiple awaitables concurrently and collect results in input order:

```python
from lionagi.ln.concurrency import gather

results = await gather(
    branch.communicate("Analyze market trends"),
    branch.communicate("Analyze competitor landscape"),
    branch.communicate("Analyze technology adoption"),
)
# results[0], results[1], results[2] match input order
```

With `return_exceptions=True`, failures are returned as exception objects
instead of propagating:

```python
results = await gather(
    branch.communicate("Task A"),
    branch.communicate("Task B"),
    return_exceptions=True,
)
for r in results:
    if isinstance(r, Exception):
        print(f"Failed: {r}")
    else:
        print(f"Success: {r[:50]}...")
```

### race -- First-to-Complete

Run multiple awaitables and return the first result. All other tasks are
cancelled:

```python
from lionagi.ln.concurrency import race

# Try multiple providers, use whichever responds first
fastest = await race(
    openai_branch.communicate("Summarize this paper"),
    anthropic_branch.communicate("Summarize this paper"),
)
```

### bounded_map -- Concurrent Mapping with Limit

Apply an async function to a sequence of items with a concurrency limit:

```python
from lionagi.ln.concurrency import bounded_map

documents = ["doc1.txt", "doc2.txt", "doc3.txt", "doc4.txt", "doc5.txt"]

async def summarize(doc: str):
    return await branch.communicate(f"Summarize: {doc}")

# Process all documents, at most 3 at a time
summaries = await bounded_map(summarize, documents, limit=3)
```

Like `gather`, it supports `return_exceptions=True` for partial failure
tolerance.

### CompletionStream -- Results As They Arrive

Iterate over results in completion order (first-finished, not input order):

```python
from lionagi.ln.concurrency import CompletionStream

tasks = [
    branch.communicate(f"Analyze topic {i}")
    for i in range(10)
]

async with CompletionStream(tasks, limit=5) as stream:
    async for idx, result in stream:
        print(f"Task {idx} finished: {result[:80]}...")
        # Process each result as soon as it's available
```

The `limit` parameter controls how many tasks run concurrently. Without
it, all tasks start immediately.

### retry -- Exponential Backoff

Retry an async callable with exponential backoff and deadline awareness:

```python
from lionagi.ln.concurrency import retry

result = await retry(
    lambda: branch.communicate("Flaky request"),
    attempts=3,
    base_delay=0.5,
    max_delay=5.0,
    retry_on=(ValueError, ConnectionError),
    jitter=0.1,
)
```

`retry` respects structured concurrency: cancellation exceptions are
never retried, and delays are capped to any ambient deadline from a
parent `CancelScope`.

## iModel Rate Limiting

Every `iModel` instance uses a `RateLimitedAPIExecutor` that queues
requests and enforces rate limits automatically.

### Configuration

```python
from lionagi import iModel

model = iModel(
    provider="openai",
    model="gpt-4.1-mini",
    # Rate limiting
    limit_requests=60,          # Max requests per cycle
    limit_tokens=100_000,       # Max tokens per cycle
    capacity_refresh_time=60,   # Cycle duration in seconds
    # Queue
    queue_capacity=100,         # Max queued requests
    # Streaming concurrency
    concurrency_limit=5,        # Max concurrent streaming requests
)
```

The executor maintains token and request budgets that replenish every
`capacity_refresh_time` seconds. When limits are exhausted, requests
queue until capacity is available.

### Per-Task Model Selection

Use lighter models for simple tasks and heavier models for complex ones:

```python
from lionagi import Branch, iModel

fast = Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
    system="Classify briefly.",
)
powerful = Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1"),
    system="Provide detailed analysis.",
)

# Quick classification
category = await fast.communicate("Classify: complex or simple?")

# Route to appropriate model
if "complex" in str(category).lower():
    analysis = await powerful.communicate("Detailed analysis of...")
else:
    analysis = await fast.communicate("Brief analysis of...")
```

## Flow-Level Concurrency

`Session.flow()` controls how many operations in a graph run
simultaneously:

```python
from lionagi import Session, Builder

# Run at most 3 operations concurrently
result = await session.flow(builder.get_graph(), max_concurrent=3)

# Sequential execution (useful for debugging)
result = await session.flow(builder.get_graph(), parallel=False)
```

The `max_concurrent` parameter maps directly to a `CapacityLimiter` in
the `DependencyAwareExecutor`. The default is 5.

## Memory Management

### Clearing Message History

Long-running branches accumulate messages. Clear them when context is
no longer needed:

```python
branch = Branch(chat_model=iModel(provider="openai", model="gpt-4.1-mini"))

for chunk in data_chunks:
    result = await branch.communicate(f"Process: {chunk}")
    results.append(result)
    branch.messages.clear()  # Free memory, reset context
```

### Branch as Context Manager

Branch supports `async with` for automatic log cleanup:

```python
async with Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1-mini")
) as branch:
    result = await branch.communicate("Analyze this data")
    # Logs are automatically dumped on exit
```

### Flow Cleanup

For large graphs, use `flow_with_cleanup` to free operation results
after execution:

```python
from lionagi.operations.flow import flow_with_cleanup

result = await flow_with_cleanup(
    session=session,
    graph=builder.get_graph(),
    cleanup_results=True,
    keep_only=[final_op_id],  # Only keep the final result
)
```

## Structured Concurrency Patterns

### Task Groups

For fine-grained control, use `TaskGroup` directly:

```python
from lionagi.ln.concurrency import create_task_group

results = {}

async with create_task_group() as tg:
    async def worker(name, prompt):
        results[name] = await branch.communicate(prompt)

    tg.start_soon(worker, "market", "Analyze market")
    tg.start_soon(worker, "tech", "Analyze technology")
    # All tasks complete before exiting the context
```

### Cancel Scopes

Set timeouts on operations:

```python
from lionagi.ln.concurrency import fail_after, move_on_after

# Hard timeout: raises TimeoutError after 30 seconds
with fail_after(30):
    result = await branch.communicate("Complex analysis...")

# Soft timeout: continues execution, result may be None
with move_on_after(10) as scope:
    result = await branch.communicate("Quick check...")
if scope.cancelled_caught:
    result = "Timed out, using fallback"
```

## Guidelines

- Use `bounded_map` instead of manual batching loops -- it handles
  concurrency limiting and error propagation correctly.
- Set `limit_requests` and `limit_tokens` on `iModel` to match your
  API provider's rate limits.
- Use `CompletionStream` when you need to process results as they arrive
  rather than waiting for all to complete.
- Prefer `gather` over `asyncio.gather` -- lionagi's version uses
  structured concurrency (AnyIO TaskGroups) which provides proper
  cancellation semantics.
