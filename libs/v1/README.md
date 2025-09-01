# LION V1: Elevated Architecture

> "composability is the sh*t of our framework. the ability to compose complex systems from small parts, through a mechanism we can TRUST" - Ocean

## Philosophy

V1 is not a rewrite - it's an elevation. After 20-30 organic iterations of v0, we discovered profound truths about agent systems:

1. **Everything is Observable** - Not just data, but the act of observation itself has effects
2. **Trust Through Invariants** - Composition only works when you can trust the parts
3. **LION is THE Observer** - The system observes AI outputs and decides what happens
4. **ABCs Change Everything** - They're not interfaces, they're compositional contracts

## The Core Chain

```
AI Output → LION Observes → Validates → Checks Capabilities → Performs Events
    ↑                                                              ↓
    └──────────────────── System learns/adapts ───────────────────┘
```

## Architecture Layers

### Kernel Layer (`kernel/`)
The agent OS - provides identity, lifecycle, execution, and safety.

- **foundation/** - Observable protocol, ABCs, core contracts
- **execution/** - Runner, Branch, TaskGroup (single, not nested)
- **safety/** - IPU (Invariant Protection Unit), capabilities
- **interface/** - Decoupling layer for clean boundaries

### Domain Layer (`domain/`)
Battle-tested v0 patterns elevated with formal invariants.

- **generic/** - Pile, Element, Event, Progression (v0 gold)
- **models/** - FieldModel, OperableModel with compositional validation
- **patterns/** - ReAct, Instruct, other proven patterns

### Wire Layer (`wire/`)
Pure speed for network operations.

- **dto/** - msgspec structs for wire protocol
- **transport/** - Pure HTTP, no SDK dependencies
- **dialects/** - OpenAI, Anthropic, etc. compatibility

### Services Layer (`services/`)
High-level APIs that compose lower layers.

- **orchestration/** - Multi-agent coordination
- **persistence/** - Durable state management
- **monitoring/** - Observability and metrics

## Key Innovations

### IPU: Invariant Protection Unit
The trust mechanism that ensures composition maintains system invariants. When IPU validates something, the system can trust it completely.

### Observable as Protocol
Not a base class but a protocol - everything adapts to it, not inherits from it. Minimal atom: just needs an `id`.

### Hybrid Data Model
- **Domain Objects**: Pydantic for rich validation and developer experience
- **Wire Protocol**: msgspec for blazing fast serialization
- **Seamless Conversion**: Adapters handle the transformation

### Composability First
Every component is immutable and composable. Transformations return new instances, maintaining trust through invariant preservation.

## Design Principles

1. **Immutable Composition** - Never mutate, always transform
2. **Trust by Default** - If IPU approves, system trusts
3. **Effects are Explicit** - Observation causes predictable effects
4. **Performance Matters** - msgspec for hot paths, Pydantic where it counts
5. **V0 Wisdom Preserved** - 20-30 iterations encoded deep truths

## Migration from V0

V1 preserves v0's battle-tested patterns:
- Pile remains first-class (Observable collection)
- Element, Event, Progression restored
- Pure HTTP patterns maintained
- OperableModel's dynamic fields
- ReAct's intermediate response options

What's elevated:
- Formal invariant system
- Mathematical trust guarantees
- Structured concurrency (single TaskGroup)
- Capability-based security
- Wire protocol optimization

## The Path Forward

```python
# V0 Wisdom
observable = Element()  # Simple, works

# V1 Elevation  
observable = Element()  # Simple, works, AND mathematically verified
ipu.validate(observable)  # Trust established
```

The goal: Enterprise-ready, powerful, rigorous - suitable for serious adoption while maintaining v0's elegant simplicity.