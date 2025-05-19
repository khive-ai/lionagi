# Structured Concurrency

The Structured Concurrency module is a core component of Pynector that provides
a robust foundation for managing concurrent operations in Python. It leverages
AnyIO to provide a consistent interface for structured concurrency across both
asyncio and trio backends, focusing on task groups, cancellation scopes, and
resource management primitives.

## Table of Contents

- [Design Philosophy](#design-philosophy)
- [Components](#components)
  - [Task Groups](#task-groups)
  - [Cancellation Scopes](#cancellation-scopes)
  - [Resource Management Primitives](#resource-management-primitives)
  - [Error Handling](#error-handling)
  - [Concurrency Patterns](#concurrency-patterns)
- [Usage Examples](#usage-examples)
  - [Basic Task Group Usage](#basic-task-group-usage)
  - [Cancellation and Timeouts](#cancellation-and-timeouts)
  - [Resource Management](#resource-management)
  - [Using Concurrency Patterns](#using-concurrency-patterns)
  - [Error Handling](#error-handling-examples)

## Design Philosophy

The Structured Concurrency module is designed with the following principles in
mind:

### Structured Concurrency Pattern

Structured concurrency is a programming pattern that ensures that concurrent
operations have well-defined lifetimes that are bound to a scope. This means
that:

- **Scope-Bound Lifetimes**: All tasks started in a scope are guaranteed to
  complete before the scope exits.
- **Automatic Cleanup**: Resources are automatically cleaned up when the scope
  exits.
- **Error Propagation**: Errors from any task are propagated to the parent
  scope.

This pattern makes concurrent code more predictable, easier to reason about, and
less prone to resource leaks and race conditions.

### AnyIO Integration

The module uses AnyIO as its foundation, which provides a consistent interface
for both asyncio and trio backends. This allows developers to write code that
works with either backend without modification.

### Context Management

The module uses async context managers extensively for resource handling. This
ensures that resources are properly acquired and released, even in the presence
of exceptions.

### Comprehensive Error Handling

The module provides a comprehensive error handling system that makes it easier
to handle specific error conditions and propagate errors correctly.

## Components

### Task Groups

Task groups provide a way to spawn and manage multiple concurrent tasks while
ensuring proper cleanup and error propagation.

```python
from pynector.concurrency import TaskGroup, create_task_group

async def main():
    async with create_task_group() as tg:
        await tg.start_soon(task1)
        await tg.start_soon(task2)
        # All tasks will complete before exiting the context
```

The `TaskGroup` class provides the following methods:

- `start_soon(func, *args, name=None)`: Start a new task in the task group
  without waiting for it to initialize.
- `start(func, *args, name=None)`: Start a new task and wait for it to
  initialize.

### Cancellation Scopes

Cancellation scopes provide fine-grained control over task cancellation and
timeouts.

```python
from pynector.concurrency import CancelScope, move_on_after, fail_after

async def main():
    # Using CancelScope directly
    with CancelScope() as scope:
        # Do something
        if condition:
            scope.cancel()  # Cancel all tasks in this scope

    # Using timeout utilities
    with move_on_after(5) as scope:  # Continue after 5 seconds
        await long_running_operation()
        if scope.cancelled_caught:
            print("Operation timed out")

    # Using fail_after to raise TimeoutError
    try:
        with fail_after(5):  # Raise TimeoutError after 5 seconds
            await long_running_operation()
    except TimeoutError:
        print("Operation timed out")
```

### Resource Management Primitives

The module provides several primitives for managing concurrent access to
resources:

#### Lock

A mutex lock for controlling access to a shared resource.

```python
from pynector.concurrency import Lock

async def main():
    lock = Lock()

    async with lock:
        # Critical section
        # Only one task can execute this at a time
```

#### Semaphore

A semaphore for limiting concurrent access to a resource.

```python
from pynector.concurrency import Semaphore

async def main():
    semaphore = Semaphore(3)  # Allow up to 3 concurrent accesses

    async with semaphore:
        # Up to 3 tasks can execute this concurrently
```

#### CapacityLimiter

A context manager for limiting the number of concurrent operations.

```python
from pynector.concurrency import CapacityLimiter

async def main():
    limiter = CapacityLimiter(10)  # Allow up to 10 concurrent operations

    async with limiter:
        # Up to 10 tasks can execute this concurrently
```

#### Event

An event object for task synchronization.

```python
from pynector.concurrency import Event

async def main():
    event = Event()

    # In one task
    await event.wait()  # Wait until the event is set

    # In another task
    event.set()  # Allow waiting tasks to proceed
```

#### Condition

A condition variable for task synchronization.

```python
from pynector.concurrency import Condition, Lock

async def main():
    condition = Condition(Lock())

    async with condition:
        await condition.wait()  # Wait for a notification

    async with condition:
        await condition.notify()  # Notify one waiting task
        # or
        await condition.notify_all()  # Notify all waiting tasks
```

### Error Handling

The module provides utilities for handling cancellation and shielding tasks from
cancellation.

```python
from pynector.concurrency import get_cancelled_exc_class, shield

async def main():
    # Get the exception class used for cancellation
    cancelled_exc_class = get_cancelled_exc_class()

    try:
        # Do something
        pass
    except cancelled_exc_class:
        print("Task was cancelled")

    # Shield a task from cancellation
    result = await shield(critical_operation)
```

### Concurrency Patterns

The module implements several common concurrency patterns:

#### ConnectionPool

A pool of reusable connections.

```python
from pynector.concurrency.patterns import ConnectionPool

async def create_connection():
    # Create and return a connection
    pass

async def main():
    pool = ConnectionPool(max_connections=10, connection_factory=create_connection)

    async with pool as p:
        # Acquire a connection
        conn = await p.acquire()

        try:
            # Use the connection
            pass
        finally:
            # Release the connection back to the pool
            await p.release(conn)
```

#### Parallel Requests

Fetch multiple URLs in parallel with limited concurrency.

```python
from pynector.concurrency.patterns import parallel_requests

async def fetch(url):
    # Fetch and return the response
    pass

async def main():
    urls = ["https://example.com", "https://example.org", "https://example.net"]
    responses = await parallel_requests(urls, fetch, max_concurrency=5)

    for response in responses:
        print(response)
```

#### Retry with Timeout

Execute a function with retry logic and timeout.

```python
from pynector.concurrency.patterns import retry_with_timeout

async def flaky_operation():
    # An operation that might fail or time out
    pass

async def main():
    try:
        result = await retry_with_timeout(
            flaky_operation,
            max_retries=3,
            timeout=5.0,
            retry_exceptions=[ConnectionError, TimeoutError]
        )
    except TimeoutError:
        print("Operation timed out after all retries")
    except Exception as e:
        print(f"Operation failed: {e}")
```

#### Worker Pool

A pool of worker tasks that process items from a queue.

```python
from pynector.concurrency.patterns import WorkerPool

async def process_item(item):
    # Process the item
    pass

async def main():
    pool = WorkerPool(num_workers=5, worker_func=process_item)

    await pool.start()

    try:
        # Submit items for processing
        for item in items:
            await pool.submit(item)
    finally:
        # Stop the worker pool
        await pool.stop()
```

## Usage Examples

### Basic Task Group Usage

Here's a basic example of how to use task groups:

```python
import asyncio
from pynector.concurrency import create_task_group

async def task1():
    print("Task 1 started")
    await asyncio.sleep(1)
    print("Task 1 completed")

async def task2():
    print("Task 2 started")
    await asyncio.sleep(2)
    print("Task 2 completed")

async def main():
    async with create_task_group() as tg:
        await tg.start_soon(task1)
        await tg.start_soon(task2)
        print("All tasks started")

    print("All tasks completed")

asyncio.run(main())
```

### Cancellation and Timeouts

Here's an example of how to use cancellation scopes and timeouts:

```python
import asyncio
from pynector.concurrency import move_on_after, fail_after

async def long_running_task():
    print("Long-running task started")
    await asyncio.sleep(10)
    print("Long-running task completed")

async def main():
    # Using move_on_after
    print("Using move_on_after:")
    with move_on_after(2) as scope:
        await long_running_task()
        if scope.cancelled_caught:
            print("Task was cancelled after timeout")

    print("Continued execution after timeout")

    # Using fail_after
    print("\nUsing fail_after:")
    try:
        with fail_after(2):
            await long_running_task()
    except TimeoutError:
        print("TimeoutError was raised")

    print("Continued execution after timeout exception")

asyncio.run(main())
```

### Resource Management

Here's an example of how to use resource management primitives:

```python
import asyncio
from pynector.concurrency import Lock, Semaphore, CapacityLimiter

shared_resource = []

async def task_with_lock(lock, item):
    async with lock:
        print(f"Adding {item} to shared resource")
        shared_resource.append(item)
        await asyncio.sleep(1)  # Simulate work
        print(f"Finished processing {item}")

async def task_with_semaphore(semaphore, item):
    async with semaphore:
        print(f"Processing {item} with semaphore")
        await asyncio.sleep(1)  # Simulate work
        print(f"Finished processing {item}")

async def task_with_limiter(limiter, item):
    async with limiter:
        print(f"Processing {item} with limiter")
        await asyncio.sleep(1)  # Simulate work
        print(f"Finished processing {item}")

async def main():
    # Using Lock
    lock = Lock()
    tasks = [task_with_lock(lock, i) for i in range(5)]
    await asyncio.gather(*tasks)
    print(f"Shared resource: {shared_resource}\n")

    # Using Semaphore
    semaphore = Semaphore(3)  # Allow up to 3 concurrent tasks
    tasks = [task_with_semaphore(semaphore, i) for i in range(5)]
    await asyncio.gather(*tasks)
    print()

    # Using CapacityLimiter
    limiter = CapacityLimiter(2)  # Allow up to 2 concurrent tasks
    tasks = [task_with_limiter(limiter, i) for i in range(5)]
    await asyncio.gather(*tasks)

asyncio.run(main())
```

### Using Concurrency Patterns

Here's an example of how to use the concurrency patterns:

```python
import asyncio
from pynector.concurrency.patterns import parallel_requests, retry_with_timeout

async def fetch(url):
    print(f"Fetching {url}")
    await asyncio.sleep(1)  # Simulate network delay
    return f"Response from {url}"

async def flaky_operation():
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("Simulated connection error")
    return "Success"

async def main():
    # Using parallel_requests
    urls = ["https://example.com", "https://example.org", "https://example.net"]
    responses = await parallel_requests(urls, fetch, max_concurrency=2)

    for url, response in zip(urls, responses):
        print(f"{url} -> {response}")

    print()

    # Using retry_with_timeout
    try:
        result = await retry_with_timeout(
            flaky_operation,
            max_retries=5,
            timeout=1.0,
            retry_exceptions=[ConnectionError]
        )
        print(f"Operation succeeded: {result}")
    except TimeoutError:
        print("Operation timed out after all retries")
    except Exception as e:
        print(f"Operation failed: {e}")

asyncio.run(main())
```

### Error Handling Examples

Here's an example of how to handle errors in structured concurrency:

```python
import asyncio
from pynector.concurrency import create_task_group, get_cancelled_exc_class, shield

async def task_that_fails():
    await asyncio.sleep(1)
    raise ValueError("Simulated error")

async def task_that_gets_cancelled():
    try:
        await asyncio.sleep(10)
    except get_cancelled_exc_class():
        print("Task was cancelled")
        raise

async def critical_operation():
    print("Starting critical operation")
    await asyncio.sleep(1)
    print("Critical operation completed")
    return "Critical result"

async def main():
    # Handling errors in task groups
    try:
        async with create_task_group() as tg:
            await tg.start_soon(task_that_fails)
            await tg.start_soon(asyncio.sleep, 2)
    except ValueError as e:
        print(f"Caught error from task: {e}")

    # Handling cancellation
    try:
        async with create_task_group() as tg:
            await tg.start_soon(task_that_gets_cancelled)
            await asyncio.sleep(0.5)
            # The task group will be cancelled when we exit the context
    except Exception as e:
        print(f"Caught exception: {e}")

    # Shielding from cancellation
    try:
        async with create_task_group() as tg:
            await tg.start_soon(task_that_gets_cancelled)
            # Shield the critical operation from cancellation
            result = await shield(critical_operation)
            print(f"Got result: {result}")
            await asyncio.sleep(0.5)
            # The task group will be cancelled when we exit the context
    except Exception as e:
        print(f"Caught exception: {e}")

asyncio.run(main())
```
