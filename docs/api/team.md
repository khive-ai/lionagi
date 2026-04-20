# Team Messaging API

The `Session` exchange enables `Branch` instances to communicate asynchronously without
direct references. This is the programmatic equivalent of `li team`.

For CLI team commands, see [CLI reference: `li team`](../cli-reference.md#li-team).

## Exchange primitives

All methods are on `Session`:

| Method | Signature | What it does |
|--------|-----------|-------------|
| `send` | `(sender_id, recipient_id, content, channel=None)` | Queue a message |
| `receive` | `(owner_id, sender=None)` → `list[Message]` | Peek inbound (non-destructive) |
| `pop_message` | `(owner_id, sender_id)` → `Message \| None` | Consume oldest (FIFO) |
| `collect` | `async (owner_id)` → `int` | Route one entity's outbox |
| `sync` | `async ()` → `int` | Route all pending messages |
| `register_participant` | `(entity_id)` → `Flow` | Register non-branch entity |

## Typical pattern

```python
import asyncio
import lionagi as li

async def main():
    session = li.Session()
    b1 = session.new_branch(name="researcher")
    b2 = session.new_branch(name="writer")

    # b1 does work, sends result to b2
    result = await b1.communicate("What are the top 3 advances in vector databases?")
    session.send(b1.ln_id, b2.ln_id, result)

    # route messages
    await session.sync()

    # b2 reads and responds
    msgs = session.receive(b2.ln_id)
    report = await b2.communicate(
        f"Write a blog post based on: {msgs[0].content}"
    )
    print(report)

asyncio.run(main())
```

## Message routing

```python
# send to a specific branch
session.send(b1.ln_id, b2.ln_id, "task complete: see attached analysis")

# route one branch's outbox
n_routed = await session.collect(b1.ln_id)

# route everything
n_routed = await session.sync()

# peek without consuming (returns list[Message])
inbox = session.receive(b2.ln_id)
inbox_from_b1 = session.receive(b2.ln_id, sender=b1.ln_id)

# consume FIFO
msg = session.pop_message(b2.ln_id, sender=b1.ln_id)
if msg:
    print(msg.content)
```

## Broadcast pattern

```python
# send same message to multiple recipients
for branch in [reviewer, auditor, critic]:
    session.send(coordinator.ln_id, branch.ln_id, draft_text)

await session.sync()
```

## API vs CLI comparison

| Aspect | Python `Session.exchange` | `li team` |
|--------|--------------------------|-----------|
| Persistence | In-memory (session lifetime) | File-backed (`~/.lionagi/teams/`) |
| Cross-invocation | No | Yes |
| Create | `Session()` | `li team create --name myteam` |
| Send | `session.send(...)` | `li team send "..." -t TEAM_ID` |
| Read | `session.receive(...)` | `li team receive -t TEAM_ID --as NAME` |
| Pop | `session.pop_message(...)` | (consumed on receive) |

Use `li team` when you need messages to survive across CLI invocations or background runs.
Use `Session.exchange` when coordinating branches within a single Python process.

## Example: parallel workers + coordinator

```python
import asyncio
import lionagi as li

async def main():
    session = li.Session()
    coord = session.new_branch(name="coordinator")
    workers = [session.new_branch(name=f"worker-{i}") for i in range(3)]

    # coordinator broadcasts task
    task = "Analyze the impact of LLMs on software engineering"
    for w in workers:
        session.send(coord.ln_id, w.ln_id, task)
    await session.sync()

    # workers process in parallel
    async def process(worker):
        msgs = session.receive(worker.ln_id)
        result = await worker.communicate(msgs[0].content)
        session.send(worker.ln_id, coord.ln_id, result)

    await asyncio.gather(*[process(w) for w in workers])
    await session.sync()

    # coordinator synthesizes
    findings = session.receive(coord.ln_id)
    synthesis = await coord.communicate(
        "Synthesize these three analyses:\n"
        + "\n---\n".join(m.content for m in findings)
    )
    print(synthesis)

asyncio.run(main())
```

Next: [`iModel`](imodel.md) — configure providers and rate limits
