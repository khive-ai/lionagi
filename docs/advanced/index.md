# Advanced Topics

This section covers production-oriented features of lionagi: concurrency
primitives, resilience patterns, workflow composition, and runtime
observability.

## What's Here

- **[Custom Operations](custom-operations.md)** -- Register custom async
  functions as Branch operations, compose them into graphs with
  `OperationGraphBuilder`, and use `Session.operation()` as a decorator.

- **[Flow Composition](flow-composition.md)** -- Build multi-step graphs with
  `OperationGraphBuilder`, control dependencies, fan-out with `expand_from_result`,
  and aggregate results.

- **[Performance](performance.md)** -- Use lionagi's structured concurrency
  utilities (`gather`, `race`, `bounded_map`, `CompletionStream`, `retry`)
  and iModel rate limiting to maximize throughput.

- **[Error Handling](error-handling.md)** -- Built-in retry logic, circuit
  breakers, rate limiting, and structured error propagation in iModel and
  operation flows.

- **[Observability](observability.md)** -- DataLogger and Log objects,
  HookRegistry for pre/post-invoke hooks on iModel, message inspection,
  and flow-level verbose mode.

## Prerequisites

Before reading these pages you should be comfortable with:

- Branch operations (`chat`, `communicate`, `operate`, `ReAct`)
- iModel configuration and provider setup
- Python `async`/`await` patterns
