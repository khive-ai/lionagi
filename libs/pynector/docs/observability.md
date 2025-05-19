# Optional Observability

Pynector provides optional observability features through its telemetry module,
including distributed tracing and structured logging. These features are
designed to be optional dependencies, meaning Pynector will work correctly even
if the observability libraries are not installed.

## Key Features

- **Optional Dependencies**: OpenTelemetry for tracing and structlog for logging
  are optional dependencies.
- **No-op Fallbacks**: Graceful degradation when dependencies are not available.
- **Context Propagation**: Proper propagation of trace context across async
  boundaries.
- **Flexible Configuration**: Configuration via environment variables or
  programmatic API.
- **Unified API**: Consistent API regardless of whether dependencies are
  available.

## Components

The telemetry module consists of the following components:

1. **Telemetry Facade**: Provides a unified interface for tracing and logging
   operations, abstracting away the details of the underlying implementations.

2. **No-op Implementations**: Provide fallbacks when dependencies are not
   available, ensuring that the library works correctly even without the
   optional dependencies.

3. **Context Propagation**: Ensures trace context is properly maintained across
   async boundaries, allowing for accurate tracing of asynchronous operations.

4. **Configuration**: Provides flexible configuration options via environment
   variables and programmatic APIs.

5. **Dependency Detection**: Detects whether optional dependencies are available
   and sets appropriate flags.

## Installation

To use the observability features, you need to install Pynector with the
optional dependencies:

```bash
# Install with all observability dependencies
pip install pynector[observability]

# Or install individual dependencies
pip install pynector opentelemetry-api opentelemetry-sdk structlog
```

## Configuration

### Environment Variables

The telemetry module can be configured using environment variables:

| Variable                        | Description                                                            | Default               |
| ------------------------------- | ---------------------------------------------------------------------- | --------------------- |
| `OTEL_SDK_DISABLED`             | Disable OpenTelemetry tracing                                          | `false`               |
| `OTEL_SERVICE_NAME`             | Service name for traces                                                | `"unknown_service"`   |
| `OTEL_RESOURCE_ATTRIBUTES`      | Comma-separated key-value pairs for resource attributes                | `{}`                  |
| `OTEL_TRACES_EXPORTER`          | Comma-separated list of exporters to use (`otlp`, `console`, `zipkin`) | `"otlp"`              |
| `OTEL_EXPORTER_OTLP_ENDPOINT`   | Endpoint for OTLP exporter                                             | OpenTelemetry default |
| `OTEL_EXPORTER_ZIPKIN_ENDPOINT` | Endpoint for Zipkin exporter                                           | Zipkin default        |

### Programmatic Configuration

You can also configure the telemetry module programmatically:

```python
from pynector.telemetry import configure_telemetry

# Configure with defaults
configure_telemetry()

# Or with custom settings
configure_telemetry(
    service_name="my-service",
    resource_attributes={"deployment.environment": "production"},
    trace_enabled=True,
    log_level="INFO",
    trace_exporters=["console", "otlp"],
)
```

## Usage

### Basic Usage

The simplest way to use the telemetry module is through the `get_telemetry`
function:

```python
from pynector.telemetry import get_telemetry

# Get tracer and logger for a component
tracer, logger = get_telemetry("my_component")

# Use the logger
logger.info("Operation started", operation="process_data")

# Use the tracer
with tracer.start_as_current_span("process_data") as span:
    # Add attributes to the span
    span.set_attribute("data.size", 100)

    # Do some work...

    # Log within the span context (trace_id and span_id will be included)
    logger.info("Processing data", items=100)

    # Record events
    span.add_event("data_validated", {"valid_items": 95})

    # Handle errors
    try:
        # Do something that might fail
        result = process_data()
    except Exception as e:
        # Record the exception and set error status
        span.record_exception(e)
        logger.error("Processing failed", error=str(e))
        raise
```

### Async Usage

For async code, use the async span methods:

```python
import asyncio
from pynector.telemetry import get_telemetry

tracer, logger = get_telemetry("my_async_component")

async def process_item(item):
    async with tracer.start_as_current_async_span(f"process_{item}") as span:
        span.set_attribute("item.id", item)
        logger.info(f"Processing item {item}")
        await asyncio.sleep(0.1)  # Simulate work
        return item * 2

async def main():
    async with tracer.start_as_current_async_span("main_process") as span:
        logger.info("Starting batch processing")

        # Process items in parallel while maintaining trace context
        from pynector.telemetry.context import traced_gather

        items = [1, 2, 3, 4, 5]
        results = await traced_gather(
            tracer,
            [process_item(item) for item in items],
            name="parallel_processing"
        )

        logger.info("Batch processing complete", results=results)
        return results

# Run the async code
asyncio.run(main())
```

### Context Propagation Utilities

The telemetry module provides utilities for propagating context across async
boundaries:

```python
from pynector.telemetry import get_telemetry
from pynector.telemetry.context import traced_async_operation, traced_gather, traced_task_group

tracer, logger = get_telemetry("context_example")

# Use traced_async_operation for simple async operations
async def example_1():
    async with traced_async_operation(tracer, "my_operation") as span:
        # Do some work
        span.set_attribute("example", "value")
        logger.info("Operation in progress")

# Use traced_gather for parallel operations
async def example_2():
    async def task(i):
        logger.info(f"Task {i} started")
        return i * 2

    results = await traced_gather(
        tracer,
        [task(i) for i in range(5)],
        name="parallel_tasks"
    )
    logger.info("All tasks completed", results=results)

# Use traced_task_group for more complex task management (requires anyio)
async def example_3():
    task_group = await traced_task_group(tracer, "task_group_example")

    async def worker(name):
        logger.info(f"Worker {name} started")
        # Trace context is propagated automatically

    async with task_group:
        task_group.start_soon(worker, "A")
        task_group.start_soon(worker, "B")
```

## Behavior When Dependencies Are Missing

When the optional dependencies (OpenTelemetry and/or structlog) are not
available, the telemetry module provides no-op implementations that maintain the
same API but do nothing. This ensures that your code will work correctly even if
the dependencies are not installed.

### Tracing Without OpenTelemetry

If OpenTelemetry is not available:

- `TracingFacade` will use `NoOpSpan` implementations
- Spans will be created but will not record any data
- All span methods (`set_attribute`, `add_event`, etc.) will be available but
  will do nothing
- Context propagation utilities will fall back to simpler implementations

### Logging Without structlog

If structlog is not available:

- `LoggingFacade` will use `NoOpLogger` implementations
- Logger methods (`info`, `error`, etc.) will be available but will do nothing

## Integration with Other Systems

### OpenTelemetry Exporters

The telemetry module supports multiple OpenTelemetry exporters:

- **OTLP**: Send traces to an OpenTelemetry collector
- **Console**: Print traces to the console (useful for debugging)
- **Zipkin**: Send traces to a Zipkin server

You can configure which exporters to use via the `OTEL_TRACES_EXPORTER`
environment variable or the `trace_exporters` parameter in
`configure_telemetry()`.

### Structured Logging

The telemetry module uses structlog for structured logging, which provides:

- JSON-formatted logs
- Automatic inclusion of trace context in logs
- Contextual information in all log entries

## Best Practices

1. **Initialize Early**: Call `configure_telemetry()` early in your application
   startup.

2. **Use Descriptive Names**: Use descriptive names for spans and log events.

3. **Add Context to Logs**: Include relevant context in log entries using
   keyword arguments.

4. **Propagate Context**: Use the context propagation utilities for async code.

5. **Handle Errors**: Record exceptions in spans and set appropriate status.

6. **Clean Up Resources**: Use context managers (`with` and `async with`) to
   ensure proper cleanup.

7. **Consider Performance**: Be mindful of the performance impact of tracing and
   logging in hot paths.
