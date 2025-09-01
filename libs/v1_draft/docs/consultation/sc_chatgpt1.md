Ocean — here’s the pragmatic line I’d draw.

---

## Executive take

* **Root cause you observed (premature cancellations with nested groups) is expected structured‑concurrency behavior:** in AnyIO/Trio, **any unhandled exception inside a TaskGroup cancels all siblings**, and each TaskGroup comes with its own cancel scope. Nesting multiplies the surfaces where a parent/peer failure cancels children. Flattening the groups or catching exceptions inside workers avoids “collateral” cancels. ([AnyIO][1])
* **Keep structured concurrency for *lifecycle*; use your Pile semantics for *work management*.** Implement Pile on top of **AnyIO memory object streams** to collect per‑task results and errors, while a **single long‑lived TaskGroup** owns all worker tasks. This gives you the v0 ergonomics with v1 invariants (no orphan tasks, bounded lifetimes, graceful shutdown). ([AnyIO][2])

**Recommendation:** **(b) Hybrid** — single TaskGroup for lifecycle + a typed Pile built on AnyIO streams for result handling; add explicit error policy inside each worker. This is the 80/20 that ships. (Details + code below.)

---

## 1) Hybrid architecture: v1 guarantees, v0 ergonomics

**Goal:** preserve v1’s *lifecycle* invariants (no leaked tasks; deterministic shutdown) while letting your team keep the v0‑style “submit → process → as‑completed” flow.

**How:**

1. **One long‑lived TaskGroup per executor** (created in `__aenter__`). All background loops and per‑call tasks are spawned here. No nested groups for the steady‑state dataplane. ([AnyIO][1])
2. **Per‑call result path via a one‑shot stream** (sender kept by the worker, receiver returned to the caller). Streams are the idiomatic AnyIO producer/consumer primitive; they support closing and async iteration. ([AnyIO][2])
3. **Catch exceptions inside each worker** and **send an error outcome** over the result stream instead of letting it bubble out and cancel siblings. (Trio/AnyIO cancel siblings on unhandled exceptions; servers are expected to catch errors in handlers.) ([Trio Documentation][3], [AnyIO][1])
4. **Optional:** use `TaskGroup.start()` only for services that must signal readiness; everything else uses `start_soon`. ([AnyIO][1])
5. **Use CancelScope only for cleanups**, with `shield=True` around resource finalizers (so a group cancel doesn’t interrupt cleanup). ([AnyIO][4])

This keeps the formal lifecycle guarantees of structured concurrency while restoring the simplicity that made v0 productive.

---

## 2) Where to spend your complexity budget

**Essential (production):**

* **Deterministic lifetimes and shutdown** (structured concurrency / single TaskGroup). ([AnyIO][1])
* **Backpressure + bounded resources** (bounded streams; optional `CapacityLimiter` for concurrency and threadpool use). ([AnyIO][2])
* **Clear error policy**: handle in worker, report via result channel; aggregate with `ExceptionGroup` only at the API boundary. (Python 3.11 makes multi‑error handling first‑class.) ([Real Python][5])
* **Observability**: count in‑flight tasks, queue depth, cancels, timeouts.

**Nice‑to‑have (defer until stable):**

* Multi‑layer capability enforcement, restarts/supervision trees (unless you already need them at scale).
* Full formal models; start with *properties* you can test.

**Why:** libraries using AnyIO in production (e.g., Prefect) stick to TaskGroups for lifecycle and simple patterns for work submission/collection. It’s boring and it works. ([Prefect][6])

---

## 3) The Pile pattern: “elevate, don’t replace”

There’s precedent for “as‑completed” collectors in Python (`asyncio.as_completed`) and Trio/AnyIO designs commonly rebuild it using channels. That’s effectively your **Pile**: a collector that understands work states and yields results as they finish. Implement it as a tiny adapter over AnyIO streams and a single TaskGroup. ([Python documentation][7], [AnyIO][2])

> If you later want stronger guarantees, add a thin *verification* layer (properties/monitors/tests) above the Pile rather than swapping the runtime abstraction.

---

## 4) Minimal structured concurrency subset (the “80/20”)

1. **One TaskGroup** for the executor lifespan; **no nested groups** in hot paths. ([AnyIO][1])
2. **Per‑call results via streams**; no cross‑task mutable state or Events. ([AnyIO][2])
3. **Catch exceptions inside each worker**; never let them escape the group unless you want a fail‑fast shutdown. ([Trio Documentation][3])
4. **Use `start()` only for readiness handshakes** (listeners, accept loops). ([AnyIO][1])
5. **Use `CancelScope(shield=True)` only around finalizers**. ([AnyIO][4])
6. **Bound everything**: queue sizes; optional `CapacityLimiter` for concurrency. ([AnyIO][8])

---

## 5) Option trade‑offs

| Option                                        | Pros                                                          | Cons                                                            | When to choose                                                             |
| --------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **a) Flatten + structured**                   | Minimal diff; keeps SC invariants; easy to reason             | You’ll still rebuild a Pile‑like result path anyway             | If you want smallest change and can add channels for results               |
| **b) **Hybrid** (recommended)**               | v0 ergonomics; v1 lifecycle; clear error policy; easy to test | One small adapter to maintain                                   | Default for “ship now, stay safe later”                                    |
| **c) Strategic retreat (v0 + msgspec types)** | Fastest to ship if team is blocked                            | Risk of leaked tasks; ad‑hoc shutdown; harder to evolve         | Use only if deadlines force it; plan a return to SC                        |
| **d) Supervision trees (TriOTP‑like)**        | Robust restarts; isolation                                    | More moving parts; not needed unless you supervise many daemons | For long‑running daemons with restart policy needs ([linkdd.github.io][9]) |

---

## 6) Reference sketch: **TypedPile** over AnyIO streams + single TaskGroup

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, AsyncIterator
import anyio
from anyio import create_memory_object_stream, create_task_group

T = TypeVar("T")
R = TypeVar("R")

@dataclass
class Outcome(Generic[R]):
    result: R | None = None
    error: BaseException | None = None

class TypedPile(Generic[R]):
    """As-completed collector built on AnyIO streams."""
    def __init__(self, expected: int):
        self._expected = expected
        self._res_send, self._res_recv = create_memory_object_stream[Outcome[R]](expected)

    async def include(self, tg: anyio.abc.TaskGroup, coro) -> None:
        tg.start_soon(self._run_one, coro)

    async def _run_one(self, coro) -> None:
        try:
            res = await coro
            await self._res_send.send(Outcome(result=res))
        except BaseException as e:
            await self._res_send.send(Outcome(error=e))
        finally:
            # close when last result sent
            self._expected -= 1
            if self._expected == 0:
                await self._res_send.aclose()

    async def iter_completed(self) -> AsyncIterator[Outcome[R]]:
        async with self._res_recv:
            async for item in self._res_recv:
                yield item
```

**Executor skeleton using the pile:**

```python
class RateLimitedExecutor:
    def __init__(self, max_queue: int = 1024, max_concurrency: int = 64):
        self._work_send, self._work_recv = create_memory_object_stream(max_queue)
        self._limiter = anyio.CapacityLimiter(max_concurrency)  # optional concurrency cap
        self._tg = None

    async def __aenter__(self):
        self._tg = create_task_group()
        await self._tg.__aenter__()
        self._tg.start_soon(self._processor_loop)
        return self

    async def __aexit__(self, et, ev, tb):
        await self._work_send.aclose()
        await self._tg.__aexit__(et, ev, tb)

    async def submit(self, service, request, context):
        # one-shot result stream per call
        res_send, res_recv = create_memory_object_stream
        await self._work_send.send((service, request, context, res_send))
        return res_recv  # caller awaits its own result

    async def _processor_loop(self):
        async with self._work_recv:
            async for service, request, context, res_send in self._work_recv:
                # spawn each call in the SAME task group; catch exceptions inside
                self._tg.start_soon(self._execute_call, service, request, context, res_send)

    async def _execute_call(self, service, request, context, res_send):
        try:
            async with anyio.Semaphore(1):  # placeholder for per-service caps if needed
                # Optionally acquire tokens from self._limiter
                await self._limiter.acquire()
                try:
                    result = await service.call(request, context)
                    await res_send.send(Outcome(result=result))
                finally:
                    self._limiter.release()
        except BaseException as e:
            await res_send.send(Outcome(error=e))
        finally:
            await res_send.aclose()
```

**Caller:**

```python
res_recv = await executor.submit(service, req, ctx)
async with res_recv:
    outcome = await res_recv.receive()
if outcome.error:
    raise outcome.error
return outcome.result
```

**Why this works**

* One TaskGroup → predictable cancellation and shutdown. ([AnyIO][1])
* Results as messages → no cross‑task Events; clean completion signaling. ([AnyIO][2])
* Errors are **contained** inside workers → no sibling cancels unless you choose to escalate. ([Trio Documentation][3])
* Optional **`CapacityLimiter`** caps concurrency without extra groups. ([AnyIO][8])

---

## 7) Guardrails that prevent the “nested‑group” pitfall

* **Flatten**: one executor‑scope TaskGroup; if you *must* nest (e.g., a short‑lived sub‑batch), *never* let exceptions escape the inner group. ([AnyIO][1])
* **Shield only for cleanup** (`CancelScope(shield=True)`) so cancellations don’t interrupt closing resources. ([AnyIO][4])
* **Use `start()` only when needed** (readiness handshake), otherwise `start_soon()`. ([AnyIO][1])
* **Define an error policy**: log + report over the result stream; aggregate at boundaries via `ExceptionGroup`/`except*` if needed. ([Real Python][5])

---

## 8) If you want *formal-ish* confidence without rewriting everything

* **Property‑based tests** for invariants like “no result lost/duplicated” or “shutdown drains safely”, using Hypothesis + `pytest‑trio` (time‑control clock, nursery fixtures). This is lightweight and effective. ([pytest-trio.readthedocs.io][10], [Trio Documentation][11])
* **Model the queue/progress protocol** in a small **TLA+** sketch (1–2 pages) if you want to smoke‑test interleavings for deadlocks/“stuck forever” states. Keep it at the protocol layer, not the implementation layer. ([lamport.azurewebsites.net][12])

---

## 9) Direct answers to your questions

1. **Hybrid?** Yes. Keep v1’s lifecycle via TaskGroup + cancel scopes; implement Pile on top of memory object streams for per‑request completion/results. Formal checks live *above* the code (tests / small specs), not inside every abstraction. ([AnyIO][2])

2. **Complexity budget:** Spend on lifecycle, backpressure, and error policy. Treat advanced invariants and capability layers as incremental. Real‑world users of AnyIO do exactly this to stay robust without fighting the model. ([Prefect][6])

3. **Pile precedent:** Yes — Python has `as_completed`, and Trio/AnyIO patterns re‑create it with channels; your Pile is a great fit as a thin, typed adapter. ([Python documentation][7], [AnyIO][2])

4. **Minimal SC subset:** Single TaskGroup; streams for results; catch errors in workers; `start()` only when needed; shielded cleanup; optionally `CapacityLimiter`. That’s \~20% of the surface area for \~80% of the value. ([AnyIO][1])

5. **Ship choice:** **(b) Hybrid**. If time‑pressed, (a) is acceptable; avoid (c) unless blocked, and plan to return to SC. Supervision trees (d) are overkill unless you need restarts. ([linkdd.github.io][9])

---

### Closing

You don’t need “pure” SC everywhere to be safe. Use SC to *bound lifetimes*, then let a tiny **TypedPile** give you the v0 developer experience — with per‑call channels and explicit error policy so cancellations never surprise you again.

[1]: https://anyio.readthedocs.io/en/stable/tasks.html "Creating and managing tasks — AnyIO 4.10.0 documentation"
[2]: https://anyio.readthedocs.io/en/stable/streams.html?utm_source=chatgpt.com "Streams — AnyIO 4.10.0 documentation - Read the Docs"
[3]: https://trio.readthedocs.io/en/stable/reference-io.html?utm_source=chatgpt.com "I/O in Trio — Trio 0.30.0 documentation"
[4]: https://anyio.readthedocs.io/en/stable/cancellation.html "Cancellation and timeouts — AnyIO 4.10.0 documentation"
[5]: https://realpython.com/python311-exception-groups/?utm_source=chatgpt.com "Python 3.11 Preview: Task and Exception Groups"
[6]: https://www.prefect.io/blog/oss-love-letters-how-anyio-powers-prefects-async-architecture?utm_source=chatgpt.com "How AnyIO powers Prefect's Async Architecture"
[7]: https://docs.python.org/3/library/asyncio-task.html?utm_source=chatgpt.com "Coroutines and Tasks"
[8]: https://anyio.readthedocs.io/en/stable/synchronization.html?utm_source=chatgpt.com "Using synchronization primitives — AnyIO 4.10.0 documentation"
[9]: https://linkdd.github.io/triotp/api/triotp.html?utm_source=chatgpt.com "triotp package — triotp 0.3.1 documentation"
[10]: https://pytest-trio.readthedocs.io/?utm_source=chatgpt.com "pytest-trio - Read the Docs"
[11]: https://trio.readthedocs.io/en/latest/awesome-trio-libraries.html?utm_source=chatgpt.com "Awesome Trio Libraries — Trio 0.30.0+dev documentation"
[12]: https://lamport.azurewebsites.net/pubs/lamport-spec-tla-plus.pdf?utm_source=chatgpt.com "Specifying Concurrent Systems with TLA+ - Leslie Lamport"
