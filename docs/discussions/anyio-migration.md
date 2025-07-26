# Migration to AnyIO: A Technical Discussion

## Overview

In our latest release (v0.14.5), we've made a strategic decision to migrate our asynchronous primitives from raw `asyncio` to `anyio`. This document discusses the rationale, implications, and future benefits of this architectural change.

## Why AnyIO?

### 1. Backend Agnosticism

AnyIO provides a unified interface that works with multiple async backends:
- **asyncio** (current Python standard)
- **trio** (structured concurrency pioneer)
- **curio** (high-performance alternative)

This abstraction layer means our codebase is no longer tightly coupled to asyncio's implementation details. As the Python async ecosystem evolves, we can adapt without major rewrites.

### 2. Structured Concurrency by Default

```python
# Before (asyncio)
tasks = []
for item in items:
    task = asyncio.create_task(process(item))
    tasks.append(task)
results = await asyncio.gather(*tasks)  # What if a task fails?

# After (anyio)
async with create_task_group() as tg:
    for item in items:
        await tg.start_soon(process, item)
# All tasks are guaranteed to complete or cancel properly
```

Task groups enforce a fundamental principle: **no task outlives its parent scope**. This eliminates entire classes of bugs related to orphaned tasks and resource leaks.

### 3. Better Cancellation Semantics

AnyIO's cancellation model is more predictable:
- Cancellation is always delivered at checkpoint
- Cancel scopes provide fine-grained control
- No more mysterious "Task was destroyed but it is pending!" errors

```python
# Timeout with proper cancellation
with anyio.move_on_after(5.0) as cancel_scope:
    result = await long_running_operation()
if cancel_scope.cancelled_caught:
    print("Operation timed out cleanly")
```

## Implementation Highlights

### Thread-to-Async Bridge

One subtle but important change is how we handle sync functions in async contexts:

```python
# Before
result = await asyncio.to_thread(sync_func, arg)

# After  
result = await anyio.to_thread.run_sync(sync_func, arg)
```

AnyIO's approach provides:
- Better thread pool management
- Proper cancellation propagation to threads
- Consistent behavior across different async backends

### Unified Lock Semantics

Our `ConcurrencyLock` now wraps AnyIO's primitives, providing:
- Same lock works in both sync and async contexts
- No more deadlocks from mixing lock types
- Better debugging with lock ownership tracking

### Sleep and Timing

```python
# Before
await asyncio.sleep(1.0)

# After
await anyio.sleep(1.0)
```

While seemingly trivial, AnyIO's sleep:
- Respects cancellation properly
- Works consistently across backends
- Integrates with structured concurrency

## Performance Implications

### The Good
1. **Better CPU utilization**: Task groups reduce overhead compared to gather/wait
2. **Memory efficiency**: Structured concurrency prevents task accumulation
3. **Predictable performance**: Consistent behavior reduces edge-case slowdowns

### The Trade-offs
1. **Slight overhead**: Abstraction layer adds minimal overhead (~5-10%)
2. **Learning curve**: Developers need to understand structured concurrency
3. **Ecosystem compatibility**: Some asyncio-specific libraries need adapters

## Future Benefits

### 1. Trio Compatibility
We can now experiment with trio as a backend for specific use cases:
```python
# Run lionagi with trio backend
import trio
import anyio

async def main():
    async with anyio.from_thread.start_blocking_portal(
        backend="trio", backend_options={"trio_token": trio.lowlevel.current_trio_token()}
    ) as portal:
        await portal.call(run_lionagi_operation)
```

### 2. Better Testing
AnyIO provides excellent testing utilities:
```python
async def test_timeout_handling():
    with anyio.move_on_after(0.1):
        await anyio.sleep(1.0)  # Will be cancelled
    # Test continues normally
```

### 3. WebAssembly Ready
As Python moves toward WASM support, AnyIO's abstraction will help us adapt to environments where traditional threading doesn't exist.

## Migration Patterns

### For Library Users
Most changes are transparent, but be aware of:
- Different exception types (use `anyio.get_cancelled_exc_class()`)
- Import changes if you were using our internal async utilities
- Slightly different timeout behavior (more predictable)

### for Library Developers
Key patterns to adopt:
```python
# Always use task groups for concurrent operations
async with create_task_group() as tg:
    await tg.start_soon(task1)
    await tg.start_soon(task2)

# Use cancel scopes for timeouts
with anyio.move_on_after(timeout):
    result = await operation()

# Handle backend-specific exceptions
try:
    await operation()
except anyio.get_cancelled_exc_class():
    # Properly handle cancellation
    raise
```

## Philosophical Alignment

This migration aligns with our core principles:

1. **Reliability First**: Structured concurrency eliminates entire bug categories
2. **Future-Proof Design**: Backend agnosticism protects against ecosystem changes  
3. **Developer Experience**: Clearer mental models reduce cognitive load
4. **Performance with Correctness**: We optimize without sacrificing safety

## Conclusion

The migration to AnyIO represents a maturation of our async architecture. While the immediate benefits include better resource management and fewer concurrency bugs, the long-term value lies in our ability to evolve with the Python async ecosystem.

This change positions lionagi to take advantage of future innovations in async Python while providing our users with a more reliable and predictable experience today.

---

*For specific migration examples and API changes, see our [v0.14.6 Release Notes](./RELEASE_NOTES_v0.14.5.md).*