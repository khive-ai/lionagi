# Domain: Battle-Tested Patterns Elevated

The domain layer contains v0's proven patterns elevated with formal invariants and compositional trust.

## Philosophy

Ocean: "my efforts in v0 are actually pretty decent"

V0's organic evolution over 20-30 rewrites discovered fundamental patterns that work. V1 doesn't replace these - it elevates them with mathematical rigor while preserving their elegant simplicity.

## Core Generic Components

### Pile
Observable collection with identity - not just a container.

**V0 Wisdom:**
- Pile has its own UUID and lifecycle
- Can observe the pile itself, not just contents
- Changes to pile trigger observations

**V1 Elevation:**
```python
class Pile(Observable, Collective[T]):
    """Observable collection - best of both worlds"""
    # Pile itself is observable (has id, created_at)
    # Contains observables
    # Composition maintains invariants
```

**Design Decisions:**
- Pile is first-class Observable entity
- Thread-safe by default (immutable updates)
- Bulk operations preserve invariants
- Lazy evaluation where possible

### Element
The fundamental building block - an observable atom.

**V0 Wisdom:**
- Everything is an Element
- Elements have identity and metadata
- Elements can be composed

**V1 Elevation:**
- Elements carry invariants
- Composition validated by IPU
- Metadata is typed and validated

### Event
Things that happen in the system.

**V0 Wisdom:**
- Events are immutable records
- Events have causality chains
- Events trigger effects

**V1 Elevation:**
- Events as Observable with guaranteed delivery
- Causal ordering preserved
- Effects are capability-gated

### Progression
Ordered sequence of states.

**V0 Wisdom:**
- Natural way to model workflows
- State transitions are explicit
- Progress is observable

**V1 Elevation:**
- State machine with invariant checking
- Transitions are atomic
- Rollback capability built-in

## Model Components

### FieldModel
The genesis of the invariant concept.

**The Evolution:**
```
Pydantic Field → FieldModel → dataclass(frozen=True) → Invariant concept
```

**Key Innovation:**
- Fields carry their own validation
- Composition through immutable transformation
- `as_nullable()`, `with_validator()` return NEW instances
- Trust built into the field itself

**Design Pattern:**
```python
field = FieldModel(str)
nullable = field.as_nullable()  # New instance!
validated = nullable.with_validator(lambda x: len(x) > 0)  # New!
# Each transformation preserves trust
```

### OperableModel  
Dynamic field management with validation.

**V0 Wisdom:**
- Add/remove fields at runtime
- Fields carry metadata
- Validation happens automatically

**V1 Elevation:**
- Field changes go through IPU
- Capability-gated field access
- Audit trail of field evolution

### ModelParams
Dynamic model generation.

**Key Innovation from ReAct:**
```python
# Create new Pydantic model at runtime
params = ModelParams(name="Dynamic", field_models=[...])
model_class = params.create_new_model()
```

This enables intermediate_response_options - runtime model composition!

## Pattern Components

### ReAct
Reasoning and acting pattern with intermediate outputs.

**The Genius of intermediate_response_options:**
- Dynamic model creation at runtime
- Validators attached to fields
- Type safety without rigid structure
- Composable response formats

### Instruct
Structured instruction handling.

**V0 Wisdom:**
- Instructions have context
- Instructions can be refined
- Instructions produce observables

**V1 Elevation:**
- Instructions validated before execution
- Capability requirements computed
- Effects predetermined by IPU

## Design Principles

1. **Preserve V0 Wisdom** - Don't fix what isn't broken
2. **Elevate with Invariants** - Add mathematical guarantees
3. **Composability First** - Everything composes safely
4. **Runtime Flexibility** - Dynamic but validated
5. **Trust Through Validation** - IPU approves all compositions

## Relationship to Kernel

Domain objects:
- Implement Observable protocol
- Validated by kernel's IPU
- Use kernel's execution context
- Respect capability boundaries

## Why These Patterns Matter

After 20-30 iterations, Ocean discovered these aren't arbitrary patterns - they're fundamental to how agent systems work:

- **Pile**: Agents need observable collections
- **Element**: Everything needs identity
- **Event**: Systems are event-driven
- **Progression**: Workflows need state machines
- **FieldModel**: Validation must be composable
- **OperableModel**: Runtime flexibility is essential

## Future Patterns

- **Conversation**: Structured dialogue management
- **Tool**: Capability-aware function calls
- **Memory**: Temporal state with decay
- **Objective**: Goal-directed behavior
- **Constraint**: Behavioral boundaries