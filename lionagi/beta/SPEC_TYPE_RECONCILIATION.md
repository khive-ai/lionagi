# Spec: Type Reconciliation — beta/core/base → production protocols

## Problem

Beta maintains parallel implementations of Element, Pile, Progression, Event,
Processor, Flow, Node, and Graph alongside production protocols/. These
diverged across prototyping iterations (v1, krons, lionherd-core). Beta code
should use production types directly, with beta-only features added to
production as backward-compatible extensions.

## Root Blocker: element.py created_at

Production: `created_at: float` (Unix timestamp)
Beta: `created_at: datetime` (timezone-aware UTC)

**Decision**: Beta adapts to production's `float` convention.

Rationale: Production is the stable, released API. Changing it to `datetime`
breaks all serialized data and downstream consumers. The `float` representation
is more portable (JSON, DB, msgspec). Production already provides
`.created_datetime` property for when datetime is needed.

Migration: Find all beta code using `created_at` as datetime object (e.g.
`event.created_at.timestamp()`, `.isoformat()`, datetime arithmetic) and
replace with float-based equivalents or the `.created_datetime` property.

## Phase 1: Direct Replacements (low risk)

### 1a. progression.py → USE_PRODUCTION

Beta adds only `pop(index, default=Undefined)` sentinel behavior.

Change: Add `pop` with sentinel default to production `Progression`:
```python
def pop(self, index: int = -1, default: Any = _MISSING):
    try:
        return self.order.pop(index)  # deque.pop is O(1) for ends
    except IndexError:
        if default is _MISSING:
            raise
        return default
```

Then: Beta imports `from lionagi.protocols.generic.progression import Progression`.
Delete `beta/core/base/progression.py`.

### 1b. graph.py → EXTEND_PRODUCTION then USE

Beta improves: separate `_out_edges`/`_in_edges` sets (vs nested dict),
immutable edge copies in `replace_node()`.

Change:
1. Add `_adjacency_out: dict[UUID, set[UUID]]` and `_adjacency_in` to
   production Graph (private, maintained alongside existing structure)
2. Fix `replace_node()` to copy edges instead of mutating in place

Then: Beta imports from `lionagi.protocols.graph.graph`.
Delete `beta/core/base/graph.py`.

### 1c. flow.py → EXTEND_PRODUCTION then USE

Beta adds `@synchronized` decorators on mutation methods and `to_dict`
format propagation.

Change: Add `@synchronized` to `add_progression`, `remove_progression`,
`add_item`, `remove_item`, `clear` in production Flow.

Then: Beta imports from production.
Delete `beta/core/base/flow.py`.

## Phase 2: Message Unification (medium risk)

### 2a. Trivial message types

| Beta | Production | Action |
|------|-----------|--------|
| `Role` enum | `MessageRole` enum | Alias, retire beta |
| `System` | `SystemContent` | Add `datetime_factory`, retire beta |
| `Assistant` | `AssistantResponseContent` | Bridge already works, retire beta |
| `ActionRequest` | `ActionRequestContent` | Add `render_compact()`, retire beta |
| `ActionResponse` | `ActionResponseContent` | Add `error` field + `render_summary()`, retire |

### 2b. Instruction (highest effort)

Field rename: beta `primary` → production `instruction`.

Strategy:
1. Add `primary` as a deprecated alias for `instruction` in production
   `InstructionContent` (property that reads/writes `instruction`)
2. Add `structure_format`, `custom_renderer` fields to production
3. Move LNDL rendering logic from beta `instruction.py` to production
4. Beta code continues using `primary` via the alias during transition

### 2c. Promote utility files

Move to `protocols/messages/`:
- `common.py` → `rendering.py` (CustomRenderer, StructureFormat)
- `prepare_msg.py` → `prepare.py` (chat preparation pipeline)
- `_validators.py` → `validators.py` (URL security validation)

## Phase 3: Complex Type Extensions (higher risk)

### 3a. element.py — Beta adapts to production

Do NOT change production Element.

Beta migration checklist:
- `created_at.timestamp()` → use `created_at` directly (already float)
- `created_at.isoformat()` → use `datetime.fromtimestamp(created_at).isoformat()`
- `kron_class` → `lion_class` (or add `kron_class` as alias in production)
- `to_dict(mode, created_at_format, meta_key)` → extend production `to_dict`
  to accept `meta_key` kwarg (simple addition)
- `@implements(Observable, Serializable, ...)` → drop, protocols are
  structural in production

Then: Beta imports from `lionagi.protocols.generic.element`.
Delete `beta/core/base/element.py`.

### 3b. pile.py — KEEP_BOTH (long-term converge)

Too many behavioral differences to merge safely in one pass:
- Private vs public storage fields
- Strict add (raises) vs idempotent include
- Different set operator support

Strategy: Beta Pile stays for now. Long-term, decide on the canonical
behavior (strict vs idempotent) and converge. The Principal/Morphism/Runner
layer doesn't use Pile directly — it's the session/conversation layer that
depends on it.

### 3c. event.py → EXTEND_PRODUCTION then USE

Add to production Event:
1. `timeout: float | None` field with validation
2. `streaming: bool` field
3. `@final` on `invoke()` with `@async_synchronized`
4. `assert_completed()` with typed `ValidationError`
5. Keep production's `completion_event` (asyncio.Event)

The `Execution` dataclass difference (slots vs class) is internal — not
visible to callers. Production's implementation is fine.

### 3d. processor.py → EXTEND_PRODUCTION then USE

Add to production Processor:
1. Priority queue option (PriorityQueue alongside existing FIFO)
2. Input validation ranges
3. Permission denial tracking (3-strike abort)

Prerequisite: element.py `created_at` must be resolved first (priority
queue sorts by `created_at.timestamp()` which needs float-compatible code).

### 3e. node.py — KEEP_BOTH (fundamentally different purpose)

Beta Node: DB-persistence-first (NodeConfig, DDL generation, integrity
hashing, soft delete). Production Node: graph-traversal-first (content: Any,
embedding, pydapter adapters).

These serve different domains. Don't force-merge. Long-term, consider a
`PersistableNode` mixin that adds beta's features to production Node.

## Implementation Order

```
Phase 1a: progression    (1 method addition, delete beta file)
Phase 1b: graph          (adjacency + replace_node fix, delete beta file)
Phase 1c: flow           (add @synchronized, delete beta file)
Phase 2a: messages       (5 additive changes to production content types)
Phase 2b: instruction    (field alias + structure_format + LNDL)
Phase 2c: promote utils  (3 file moves)
Phase 3a: element        (beta migration to float created_at)
Phase 3c: event          (timeout + final + assert_completed)
Phase 3d: processor      (priority queue, requires 3a)
```

Phase 3b (pile) and 3e (node) stay as-is — converge later.

## Verification Gates

After each phase:
1. `uv run python -c "import importlib, pathlib; ..."` — 0 beta import failures
2. `uv run pytest tests/ --timeout=30 -k "not integration..."` — 0 new failures
3. `git diff --stat` — net negative LOC (deleting beta duplicates)

## Non-Goals

- Changing production Element.created_at from float to datetime
- Merging beta Pile into production Pile
- Merging beta Node into production Node
- Changing production Message envelope structure
