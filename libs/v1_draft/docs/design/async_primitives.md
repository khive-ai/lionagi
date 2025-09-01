# Lion Async Primitives Design

## Purpose

Lion's `ln.concurrency` module provides thin, opinionated wrappers over AnyIO primitives to serve as foundational building blocks for Lion's orchestration layer.

## Architecture Position

```
High Level:    Flow Operations, Brainstorming, Session Management
                         |
Orchestration:  alcall/bcall/lcall (Sophisticated Coordination)  
                         |
Foundation:     ln.concurrency primitives (This Module)
                         | 
Base Layer:     AnyIO primitives
```

## Design Decisions

### Why Thin Wrappers?

1. **Backend Neutrality**: AnyIO supports both asyncio and trio - our wrappers maintain this flexibility
2. **API Stability**: Buffer against AnyIO API changes while maintaining compatibility  
3. **Ecosystem Consistency**: Single import path and consistent patterns across Lion
4. **Foundation Role**: Enable sophisticated patterns without being opinionated about their implementation

### What We Wrap

| Primitive | Purpose | Why Wrapped |
|-----------|---------|-------------|
| `Lock`, `Semaphore` | Mutual exclusion, resource limiting | Type safety, consistent async context management |
| `CapacityLimiter` | Dynamic resource control | Core to Lion's concurrency patterns in `alcall` |
| `Queue[T]` | Typed channels | Memory stream wrapper with proper cleanup, generic typing |
| `TaskGroup` | Structured concurrency | Integration point for Lion's supervision patterns |
| `fail_after/move_on_after` | Timeout patterns | AnyIO-specific semantics not in Python 3.11+ stdlib |

### What We Add (Patterns Module)

- `gather`: Fail-fast concurrent execution with result ordering
- `race`: First-completion semantics with automatic cancellation
- `bounded_map`: Concurrency-limited mapping with backpressure
- `retry`: Exponential backoff with jitter
- `as_completed`: Streaming completion processing

## Real Usage in Lion

### In `alcall` (Core Orchestration)

```python
# Concurrency limiting
semaphore = Semaphore(max_concurrent)

# Result collection safety  
results_lock = ConcurrencyLock()
async with results_lock:
    results.append(result)

# Structured concurrency
async with create_task_group() as tg:
    for item in items:
        tg.start_soon(process_item, item)
```

### In Flow Operations

```python
# Dynamic resource management
limiter = CapacityLimiter(capacity)
await self._alcall(nodes, self._execute_operation, limiter=limiter)
```

## Design Principles

1. **Minimal Surface Area**: 95% pass-through to AnyIO with focused additions
2. **Type Safety**: Proper generic typing and async context management
3. **Resource Safety**: All primitives support proper cleanup patterns
4. **Composability**: Enable higher-level patterns without constraining them
5. **Performance**: Minimal overhead over direct AnyIO usage

## Decision Framework

**Keep these primitives because:**
- Lion targets both asyncio and trio via AnyIO
- We use AnyIO-specific patterns (`move_on_after`, memory streams, capacity limiters)
- They enable Lion's sophisticated orchestration layer (`alcall`, flow operations)
- Consistent vocabulary across the Lion ecosystem

**Alternative considered:** Direct stdlib usage (Python 3.11+ `asyncio.TaskGroup`, `asyncio.timeout`) 
**Rejected because:** Loses backend neutrality and AnyIO-specific semantics that Lion's orchestration depends on

## Recent Enhancements (v1.1)

**Added features:**
1. **Absolute deadlines**: `fail_at(deadline)` and `move_on_at(deadline)` for better deadline propagation
2. **Deadline awareness**: `effective_deadline()` helper and deadline-aware `retry()`
3. **Improved gather**: Event-based cancellation instead of busy-wait polling
4. **Cancellation helper**: `is_cancelled(exc)` for clean exception handling

## Structural Concurrency Limitations

**Known Anti-Patterns:**
Our research revealed fundamental limitations with async generators and TaskGroups:

### `as_completed` Early Break Issue
```python
# This pattern violates structured concurrency:
async for idx, result in as_completed(awaitables):
    if some_condition:
        break  # Does NOT cancel remaining tasks!
```

**Root Cause:** Yielding from async generators while inside TaskGroup contexts violates structured concurrency principles. This is a well-documented limitation in AnyIO and Trio.

**Documentation:** [AnyIO Cancellation Docs](https://anyio.readthedocs.io/en/stable/cancellation.html)

**Our Approach:** 
- Document the limitation clearly in API docs
- Acknowledge that early break doesn't cancel tasks
- Recommend alternative patterns for cancellation-sensitive use cases

### Other Structural Issues
- `bounded_map` with exceptions may not cancel peers immediately
- Complex async generator patterns may violate structured concurrency

## Future Considerations

**Not planned:**
- Complex observability/logging (foundational layer should stay minimal)
- Workarounds for fundamental structured concurrency limitations
- Feature creep beyond core vocabulary

## Conclusion

These primitives provide the async vocabulary that enables Lion's orchestration patterns while maintaining simplicity and composability. The thin wrapper design is intentional - sophistication belongs in the orchestration layer, not the primitives.
