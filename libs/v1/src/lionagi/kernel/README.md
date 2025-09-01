# Kernel: The Agent Operating System

The kernel is the foundational layer - an OS for agents that provides identity, lifecycle, execution, and safety guarantees.

## Philosophy

The kernel embodies Ocean's insight: "lion system=observer, observation=AI model give output, and system validate"

The kernel IS the Lion system - the ultimate Observer that:
1. Observes AI outputs
2. Validates them against invariants
3. Checks capabilities and privileges
4. Performs authorized events

## Components

### foundation/
Core contracts and philosophical foundation.

**Key Concepts:**
- `Observable` - Protocol (not base class) that everything adapts to
- `Observer` - The Lion system itself
- `Invariant` - Truths that must hold for trust
- `Composable` - Safe composition with guaranteed invariants

**Design Decisions:**
- Protocols over inheritance (duck typing flexibility)
- Minimal Observable (just needs `id`)
- Immutable data structures
- Effects are explicit and traceable

### execution/
How work gets done - the runtime engine.

**Key Concepts:**
- `Runner` - Single-threaded async executor (no nested TaskGroups!)
- `Branch` - Isolated execution context with capabilities
- `Session` - Stateful conversation management
- `Flow` - Composable execution patterns

**Design Decisions:**
- Single TaskGroup pattern (learned from v1 hanging issues)
- Pure async/await, no threads
- Capabilities checked at execution boundaries
- Structured concurrency without nesting complexity

### safety/
Trust and security mechanisms.

**Key Concepts:**
- `IPU` - Invariant Protection Unit (the trust mechanism)
- `Capability` - What a branch can do
- `Policy` - Rules for capability grants
- `Audit` - Complete observation history

**IPU Design:**
```python
# IPU validates composition maintains trust
ipu.validate(a)  # ✓ Trust a
ipu.validate(b)  # ✓ Trust b  
ipu.validate(a.compose(b))  # ✓ Trust composition
```

**Design Decisions:**
- Trust once verified (caching)
- Capability-based, not role-based
- Fail closed (deny by default)
- Complete audit trail

### interface/
Clean boundaries between kernel and other layers.

**Key Concepts:**
- `Adapter` - Convert between representations
- `Protocol` - Communication contracts
- `Bridge` - Connect different subsystems

**Design Decisions:**
- No direct dependencies on domain/wire layers
- Everything goes through adapters
- Protocols define contracts, not implementations
- Dependency injection for flexibility

## The Observation Chain

```
1. AI produces output
2. Kernel wraps as Observable
3. IPU validates against invariants
4. Capabilities computed from validation
5. Effects determined from capabilities
6. Events triggered based on effects
7. System state updated
8. Audit trail recorded
```

## Why Kernel?

Like an OS kernel, it provides:
- **Process isolation** - Branches can't interfere
- **Resource management** - Capabilities control access
- **Security** - IPU ensures trust
- **Scheduling** - Runner manages execution
- **System calls** - Interface layer provides services

## Design Principles

1. **The kernel IS the observer** - Not passive infrastructure
2. **Trust through validation** - IPU approves everything
3. **Capabilities over permissions** - What you can do, not who you are
4. **Immutable state transitions** - Never mutate, always evolve
5. **Complete observability** - Every action is observable

## Relationship to V0

V0 had these concepts implicitly:
- Branch existed but without formal capabilities
- Validation happened but without IPU formalization
- Observable existed but as base class not protocol

V1 elevates these to first-class architectural elements with mathematical guarantees.

## Future Considerations

- Distributed kernel (multi-node)
- Capability delegation chains
- Temporal capabilities (expire after time)
- Hierarchical IPU (domain-specific validators)
- Quantum states (superposition of capabilities)