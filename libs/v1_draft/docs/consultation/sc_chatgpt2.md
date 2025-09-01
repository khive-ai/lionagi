Ocean — here’s a pragmatic path that keeps v1’s guarantees without paying an endless complexity tax.

---

## Executive summary

* **Keep structured concurrency for lifecycle only** (one long‑lived nursery per executor/component). Spawn work **into that single nursery**; avoid persistent nested groups. This preserves the “children finish or are cancelled together” invariant without surprising cross‑group cancellations. ([AnyIO][1])
* **Re‑introduce your `Pile` as a *thin* façade** over that nursery: it handles “include / next‑completed / drain” semantics and progress tracking, but delegates cancellation and shutdown to the nursery. Use **memory object streams** for the work queue and for the completed‑results stream (backpressure + clean close). ([AnyIO][2])
* **Bound concurrency** with `CapacityLimiter` (preferred over semaphores in AnyIO). ([AnyIO][3])
* Put “math/correctness” **above** the code: specify executor invariants (no lost completions, bounded in‑flight ≤ N, graceful shutdown) in **TLA+** or similar and keep the implementation small and obvious. This layering is common and language‑agnostic (see Datadog’s write‑up). ([Datadog][4])

This is essentially option **(b) Hybrid**: **Pile for work management**, **structured concurrency for lifecycle**.

---

## Why your nested groups hurt

AnyIO’s task groups (Trio’s “nurseries”) guarantee that when a group exits, *all* children are finished—success or cancellation—and exceptions in a child cancel siblings. It’s easy to accidentally create a short‑lived inner group whose exit cancels still‑running call tasks. Flattening to one owned nursery removes that accidental cancellation boundary. ([AnyIO][1])
Trio’s design notes explain why tasks are always tied to a parent scope (nursery) so errors and lifetimes are handled predictably. ([Vorpus][5], [Trio forum][6])

---

## Answers to your questions

### 1) **Hybrid architecture?**

Yes. Keep v1’s **formal lifecycle** (one nursery per executor + cancel scopes), and re‑use v0’s **Pile** at the API layer. Pile becomes a *policy object* that:

* enqueues work (stream send),
* fans out into the single nursery (start\_soon),
* yields completions as they arrive (completed stream recv),
* tracks progression states.

You can then model the above state machine independently in TLA+/PlusCal to prove properties like **no lost completion**, **bounded concurrency**, **eventual completion or cancellation**, while leaving the implementation simple. This “spec‑above‑code” approach is broadly adopted (e.g., Datadog’s modeling practice). ([Datadog][4])

### 2) **Complexity budget**

Spend complexity where production failures are expensive:

* **Lifecycle, backpressure, shutdown** (nursery + stream close + cancel scopes). ([AnyIO][2])
* **Admission control** (CapacityLimiter). ([AnyIO][3])
* **Error propagation** (let exceptions bubble to the parent scope; don’t swallow). The whole point of the nursery structure is reliable error surfacing. ([Vorpus][5])
* **Observability** (progression/metrics) and **idempotent retries** in orchestration.
  De‑prioritize theoretical purity inside every abstraction; keep invariants at the boundaries.

### 3) **The Pile pattern’s precedent**

The pattern you describe—“include tasks, yield completions as they finish”—is recognized in other ecosystems (e.g., `asyncio.as_completed`), but SC frameworks keep the lifecycle in a scoped group rather than global tasks. Your Pile can mimic `as_completed` while remaining inside a **single nursery**. The key difference from “pure SC” is ergonomic: you expose a *collection* view over many child tasks while nursery handles lifetimes. ([Python documentation][7])

### 4) **Minimal useful subset of structured concurrency**

The 80/20 set:

* **One nursery per component** (executor) – created at `__aenter__`, closed at `__aexit__`. No persistent nested groups. ([AnyIO][1])
* **Memory object streams** for queues/completions; close the send side to end loops cleanly. ([AnyIO][2])
* **Cancel scopes** for timeouts/shutdown and per‑call cancellation. ([AnyIO][8])
* **CapacityLimiter** to cap in‑flight calls. ([AnyIO][3])

That delivers most of SC’s safety with little surface area.

### 5) **Production call: a, b, c, or d?**

**(b) Hybrid**. Keep SC for lifecycle; re‑introduce Pile for day‑to‑day work management. You’ll ship faster while retaining predictable shutdown, error propagation, and concurrency bounds. (If you later want to go “purer”, you can collapse Pile’s surface into direct stream/nursery calls without changing invariants.)

---

## Concrete design (drop‑in shape)

```python
# executor_v1_hybrid.py
from __future__ import annotations
import anyio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar

T = TypeVar("T")

@dataclass
class WorkItem(Generic[T]):
    fn: Callable[[], Awaitable[T]]
    # optional metadata, ids, tracing fields...

@dataclass
class Completed(Generic[T]):
    value: T | None
    error: BaseException | None
    item: WorkItem[T]

class TypedPile(Generic[T]):
    """Thin facade: include -> spawn in shared nursery; next_completed() -> recv stream."""
    def __init__(self, tg: anyio.TaskGroup, max_inflight: int = 16) -> None:
        self._tg = tg
        self._send_q, self._recv_q = anyio.create_memory_object_stream   # work
        self._send_done, self._recv_done = anyio.create_memory_object_stream  # completions
        self._limiter = anyio.CapacityLimiter(max_inflight)

    async def start(self) -> None:
        # single processor loop; fan-out work into the same nursery
        self._tg.start_soon(self._processor)

    async def close(self) -> None:
        await self._send_q.aclose()   # ends processor via async-for
        await self._send_done.aclose()

    async def include(self, item: WorkItem[T]) -> None:
        await self._send_q.send(item)

    async def next_completed(self) -> Completed[T]:
        return await self._recv_done.receive()

    async def _processor(self) -> None:
        async for item in self._recv_q:   # consume work
            self._tg.start_soon(self._run_one, item)

    async def _run_one(self, item: WorkItem[T]) -> None:
        try:
            async with self._limiter:
                v = await item.fn()
            await self._send_done.send(Completed(value=v, error=None, item=item))
        except BaseException as exc:
            await self._send_done.send(Completed(value=None, error=exc, item=item))
```

**Why this works**

* **Single nursery** owns all children; no hidden cancellation boundaries. ([AnyIO][1])
* **Streams, not ad‑hoc queues** → backpressure + graceful termination via `.aclose()` and `async for`. ([AnyIO][2])
* **Limiter** gives a clear, tunable concurrency cap. ([AnyIO][3])

*Caller sketch*:

```python
async with anyio.create_task_group() as tg:
    pile = TypedPile[str](tg, max_inflight=32)
    await pile.start()

    # Submit N jobs
    for req in requests:
        await pile.include(WorkItem(fn=lambda r=req: service.call(r)))

    # Pull completions as they arrive
    done = 0
    while done < len(requests):
        c = await pile.next_completed()
        done += 1
        if c.error:
            # handle / retry with a cancel scope timeout if needed
            ...
```

> If you prefer the caller to `await` a specific request, add a **per‑call oneshot** (a 1‑item memory stream) to `WorkItem` and send the result down that channel instead of a shared completion stream. It’s slightly heavier but explicit and keeps object identity trivial. ([AnyIO][2])

---

## “Formal on top” (lightweight and effective)

Model the executor as a tiny state machine in **TLA+**:

* **State variables:** `Queued`, `InFlight`, `Completed`, `Failed`, `Cancelled`, `Tokens` (limiter).
* **Safety properties:**

  * *NoLostCompletion:* every enqueued item eventually appears in `Completed ∪ Failed ∪ Cancelled`.
  * *BoundedInflight:* `|InFlight| ≤ max_inflight`.
  * *NoDoubleFinish:* items move to a terminal set exactly once.
* **Liveness (optional):** under fair scheduling, `Queued` items eventually leave `Queued` unless shutdown.

This sits *above* the code and doesn’t mandate internal structure; it just constrains allowed behaviors. (Teams report success with this layering for concurrent services.) ([Datadog][4])

If you want executable tests: use **pytest‑trio** with **Hypothesis** to property‑check small queues (random enqueue/cancel/timeout sequences) against those invariants. ([pytest-trio.readthedocs.io][9])

---

## Guardrails & testing

* **Time‑bounded cancel scopes** around external calls; surface exceptions to the nursery (don’t swallow). ([AnyIO][8])
* **Shutdown:** close the work send‑stream → processor loop exits → nursery drains children. ([AnyIO][2])
* **Backpressure:** start with rendezvous (size 0) streams; increase if you need buffering. ([AnyIO][2])
* **Concurrency:** prefer `CapacityLimiter` (clearer than semaphores for “one token per task”). ([AnyIO][3])

---

## Recommendation

Choose **(b) Hybrid**:

* *Keep*: a **single, long‑lived TaskGroup** (nursery) per executor; cancel scopes; memory streams; capacity limiter. ([AnyIO][1])
* *Re‑use*: your **Pile** API as a thin layer that: `include()`, `next_completed()`, `drain()`, `stats()`.
* *Specify*: correctness with a **small TLA+ model** and **property‑based tests**. ([Datadog][4], [pytest-trio.readthedocs.io][9])

You’ll retain v1’s core guarantees and “enterprise‑ready” shutdown/error behavior, while getting back v0’s velocity and debuggability.

[1]: https://anyio.readthedocs.io/en/stable/tasks.html?utm_source=chatgpt.com "Creating and managing tasks — AnyIO 4.10.0 documentation"
[2]: https://anyio.readthedocs.io/en/stable/streams.html?utm_source=chatgpt.com "Streams — AnyIO 4.10.0 documentation - Read the Docs"
[3]: https://anyio.readthedocs.io/en/stable/synchronization.html?utm_source=chatgpt.com "Using synchronization primitives — AnyIO 4.10.0 documentation"
[4]: https://www.datadoghq.com/blog/engineering/formal-modeling-and-simulation/?utm_source=chatgpt.com "How we use formal modeling, lightweight simulations, and ..."
[5]: https://vorpus.org/blog/notes-on-structured-concurrency-or-go-statement-considered-harmful/?utm_source=chatgpt.com "Notes on structured concurrency, or: Go statement ..."
[6]: https://trio.discourse.group/t/structured-concurrency-kickoff/55?utm_source=chatgpt.com "Structured Concurrency Kickoff - Trio forum - Discourse"
[7]: https://docs.python.org/3/library/asyncio-task.html?utm_source=chatgpt.com "Coroutines and Tasks"
[8]: https://anyio.readthedocs.io/en/stable/cancellation.html?utm_source=chatgpt.com "Cancellation and timeouts — AnyIO 4.10.0 documentation"
[9]: https://pytest-trio.readthedocs.io/?utm_source=chatgpt.com "pytest-trio: Pytest plugin for trio — pytest-trio 0.8.0 ..."
