# Error Handling

lionagi provides multiple layers of resilience: retry with backoff in the
concurrency layer, circuit breakers and rate limiting in the service layer,
and structured error propagation in operation flows.

## iModel Built-in Resilience

### Rate Limiting

Every `iModel` wraps a `RateLimitedAPIExecutor` that automatically queues
and throttles requests. You configure limits at construction time:

```python
from lionagi import iModel

model = iModel(
    provider="openai",
    model="gpt-4.1-mini",
    limit_requests=60,          # Requests per cycle
    limit_tokens=100_000,       # Tokens per cycle
    capacity_refresh_time=60,   # Cycle length in seconds
    queue_capacity=100,         # Max queued requests
)
```

When the request or token budget is exhausted, the executor holds incoming
requests until the next replenishment cycle. This happens transparently --
you do not need to add manual delays.

### Circuit Breaker

Endpoints can be configured with a `CircuitBreaker` that prevents
repeated calls to a failing service:

```python
from lionagi.service.resilience import CircuitBreaker, RetryConfig

breaker = CircuitBreaker(
    failure_threshold=5,     # Open after 5 consecutive failures
    recovery_time=30.0,      # Wait 30s before testing recovery
    half_open_max_calls=1,   # Allow 1 test call in half-open state
    name="openai_chat",
)
```

Circuit states:

- **CLOSED** -- normal operation, requests pass through.
- **OPEN** -- too many failures, requests are rejected immediately with
  `CircuitBreakerOpenError`.
- **HALF_OPEN** -- recovery time elapsed, allowing a limited number of
  test calls. Success closes the circuit; failure reopens it.

### Retry with Backoff (Service Layer)

The service layer provides `retry_with_backoff` for retrying API calls:

```python
from lionagi.service.resilience import retry_with_backoff

result = await retry_with_backoff(
    some_api_call,
    arg1, arg2,
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0,
    jitter=True,
    retry_exceptions=(ConnectionError, TimeoutError),
    exclude_exceptions=(AuthenticationError,),
)
```

There is also a `@with_retry` decorator for applying retry logic to
async functions declaratively:

```python
from lionagi.service.resilience import with_retry

@with_retry(max_retries=3, base_delay=1.0)
async def call_external_api():
    ...
```

## Concurrency-Layer Retry

The `retry` function in `lionagi.ln.concurrency` provides structured-
concurrency-aware retry with deadline support:

```python
from lionagi.ln.concurrency import retry, fail_after

# Retry with ambient deadline awareness
with fail_after(30):
    result = await retry(
        lambda: branch.communicate("Analyze this"),
        attempts=3,
        base_delay=0.5,
        max_delay=5.0,
        retry_on=(ValueError,),
    )
```

Key differences from the service-layer retry:

- Uses AnyIO structured concurrency (cancellation is never retried).
- Respects parent `CancelScope` deadlines -- delays are capped so they
  do not exceed the ambient deadline.
- Takes a zero-argument async callable (use `lambda` or `functools.partial`).

## Flow-Level Error Handling

### Operation Status Tracking

Each `Operation` node in a flow graph tracks its execution status:

```python
from lionagi.protocols.generic import EventStatus

# After session.flow(), check individual operation status
for node in builder.get_graph().internal_nodes.values():
    print(f"{node.operation}: {node.execution.status}")
    if node.execution.status == EventStatus.FAILED:
        print(f"  Error: {node.execution.error}")
```

Possible statuses: `PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`,
`CANCELLED`, `SKIPPED`, `ABORTED`.

### Partial Success in Flows

When an operation in a graph fails, the error is captured and dependent
operations may still proceed (they receive the error as context). The
flow result separates completed and skipped operations:

```python
result = await session.flow(builder.get_graph(), verbose=True)

completed = result["completed_operations"]
skipped = result["skipped_operations"]
errors = {
    op_id: res
    for op_id, res in result["operation_results"].items()
    if isinstance(res, dict) and "error" in res
}

print(f"Completed: {len(completed)}, Skipped: {len(skipped)}")
for op_id, err in errors.items():
    print(f"  {str(op_id)[:8]}: {err['error']}")
```

### Edge Conditions

Graph edges can have conditions that control whether dependent operations
execute. When all incoming edges fail their conditions, the operation is
skipped rather than failed:

```python
from lionagi.protocols.graph.edge import Edge, EdgeCondition
```

### Waiting for Event Completion

Every `Event` exposes a `completion_event` (`asyncio.Event`) that is
automatically set when the event reaches a terminal status. Use this
instead of polling:

```python
import asyncio

# Wait for an event to finish (with timeout)
try:
    await asyncio.wait_for(event.completion_event.wait(), timeout=30.0)
except asyncio.TimeoutError:
    print("Event did not complete in time")

# Assert success after waiting
event.assert_completed()  # Raises RuntimeError if not COMPLETED
```

The `completion_event` is lazily created on first access and is signalled
by the `status` property setter whenever the status transitions to
`COMPLETED`, `FAILED`, `CANCELLED`, `ABORTED`, or `SKIPPED`.

### Error Accumulation

When multiple errors occur during event processing, they are accumulated
into an `ExceptionGroup` (Python 3.11+) via `execution.add_error()`:

```python
exec = event.execution
exec.add_error(ValueError("first"))
exec.add_error(TypeError("second"))
# exec.error is now an ExceptionGroup with both exceptions
```

Error accumulation is capped at 100 errors to prevent unbounded memory
growth.

## Provider Fallback Pattern

Try multiple providers in sequence:

```python
from lionagi import Branch, iModel

configs = [
    {"provider": "openai", "model": "gpt-4.1-mini"},
    {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
]

async def resilient_call(prompt: str) -> str:
    for i, config in enumerate(configs):
        try:
            branch = Branch(chat_model=iModel(**config))
            return await branch.communicate(prompt)
        except Exception as e:
            if i == len(configs) - 1:
                raise
            print(f"Provider {config['provider']} failed: {e}, trying next")
```

## Structured Output Validation

`branch.operate()` and `branch.parse()` have built-in retry logic for
structured output parsing. When the LLM returns malformed JSON,
lionagi retries the parse (not the API call) using fuzzy matching:

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float
    key_points: list[str]

result = await branch.operate(
    instruction="Analyze this review",
    response_format=Analysis,
    handle_validation="return_value",  # Return best-effort on failure
    # Other options: "raise" (raise on failure), "return_none"
)
```

The `handle_validation` parameter controls behavior when parsing fails
after all retries:

- `"raise"` -- raise an exception.
- `"return_value"` -- return whatever was parsed (may be partial).
- `"return_none"` -- return `None`.

## gather with return_exceptions

For batch operations where partial failure is acceptable:

```python
from lionagi.ln.concurrency import gather

results = await gather(
    branch.communicate("Task 1"),
    branch.communicate("Task 2"),
    branch.communicate("Task 3"),
    return_exceptions=True,
)

successes = [r for r in results if not isinstance(r, BaseException)]
failures = [r for r in results if isinstance(r, BaseException)]
print(f"{len(successes)} succeeded, {len(failures)} failed")
```

## Guidelines

- Let iModel's built-in rate limiting handle API throttling -- do not
  add manual `sleep()` calls.
- Use `CircuitBreaker` for endpoints that may go down entirely, not for
  transient rate limit errors (rate limiting handles those).
- Use `handle_validation="return_value"` in `operate()` for best-effort
  structured output rather than failing the entire pipeline.
- In flows, use `verbose=True` to diagnose which operations failed and
  why, then add targeted error handling.
- Prefer `gather(return_exceptions=True)` over try/except loops for
  batch operations.
