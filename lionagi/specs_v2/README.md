# LSpec v2: Minimal Field Specification System

**Status**: Alpha - For review

## Overview

LSpec v2 is a lightweight field specification system designed as a thin Python interface layer that routes to pluggable validation backends.

### Design Principles

1. **Python = Distribution Layer** - Simple DSL + routing, not validation
2. **Backends = Validation Layer** - Complexity lives in backends (Pydantic, Rust, Cloud)
3. **Composition = Core Value** - Reusable field specs that compose into models

## Architecture

```
┌─────────────────────────────────────────────┐
│  Python (LionAGI)                           │
│  ├─ FieldSpec: Thin field specification    │  ← Simple, ergonomic
│  └─ BackendRegistry: Route to backends     │  ← Plugin system
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Backend Tier (Pluggable)                   │
│                                             │
│  Free:     Pydantic (basic validation)      │  ← No guarantees
│  Local:    Rust (formal verification)       │  ← Paid, optional
│  Cloud:    Enterprise (full guarantees)     │  ← SaaS, $$$
└─────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage (Free Tier)

```python
from lionagi.specs_v2 import FieldSpec, BackendRegistry, PydanticBackend

# Register free Pydantic backend
BackendRegistry.register("pydantic", PydanticBackend())
BackendRegistry.set_default("pydantic")

# Define reusable field specs
AGE_SPEC = FieldSpec(int, {"min": 0, "max": 120})
EMAIL_SPEC = FieldSpec(str, {"pattern": r"^[\w\.-]+@[\w\.-]+$"})

# Validate with default backend
age = BackendRegistry.validate(AGE_SPEC, 25)
email = BackendRegistry.validate(EMAIL_SPEC, "user@example.com")
```

### Composition (Key Feature)

```python
# Reusable field specs
BASE_INSTRUCT = FieldSpec(str, {"description": "Task to perform"})
NULLABLE_INSTRUCT = BASE_INSTRUCT.as_nullable()
LIST_INSTRUCT = BASE_INSTRUCT.as_listable()

# Chain transformations
OPTIONAL_LIST = FieldSpec(str, {}).as_listable().as_nullable()
```

### Paid Tier (Rust Formal Verification)

```python
# Requires: pip install lionagi[rust]
from lionagi.specs_v2 import RustBackend

try:
    BackendRegistry.register("rust", RustBackend())

    # Use formal verification locally
    age = BackendRegistry.validate(AGE_SPEC, 25, backend="rust")
    # Rust provides mathematical guarantee this passes constraints
except ImportError:
    print("Rust backend not installed")
```

### Enterprise Tier (Cloud)

```python
from lionagi.specs_v2 import CloudBackend
import os

# Configure cloud backend
cloud = CloudBackend(
    api_key=os.getenv("LIONAGI_API_KEY"),
    endpoint="https://api.lionagi.ai"
)
BackendRegistry.register("cloud", cloud)

# Use enterprise validation
age = BackendRegistry.validate(AGE_SPEC, 25, backend="cloud")
# Cloud provides audit trail, compliance cert, insurance-backed guarantee
```

## API Reference

### FieldSpec

Lightweight field specification data structure.

```python
@dataclass(frozen=True)
class FieldSpec:
    type: type
    constraints: dict[str, Any]

    def with_constraint(self, key: str, value: Any) -> FieldSpec
    def as_nullable(self) -> FieldSpec
    def as_listable(self) -> FieldSpec
    def to_dict(self) -> dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict) -> FieldSpec
```

**Example**:
```python
# Create spec
age = FieldSpec(int, {"min": 0, "max": 120})

# Add constraints
age = age.with_constraint("description", "User age")

# Transform
optional_age = age.as_nullable()

# Serialize
spec_dict = age.to_dict()
```

### BackendRegistry

Global registry for validation backends.

```python
class BackendRegistry:
    @classmethod
    def register(cls, name: str, backend: Backend)

    @classmethod
    def get(cls, name: str | None = None) -> Backend

    @classmethod
    def set_default(cls, name: str)

    @classmethod
    def list_backends(cls) -> list[str]

    @classmethod
    def validate(cls, spec: FieldSpec, value: Any, backend: str | None = None)

    @classmethod
    def create_field(cls, spec: FieldSpec, backend: str | None = None)
```

**Example**:
```python
# Register backends
BackendRegistry.register("pydantic", PydanticBackend())
BackendRegistry.register("rust", RustBackend())

# Set default
BackendRegistry.set_default("pydantic")

# List available
print(BackendRegistry.list_backends())  # ['pydantic', 'rust']

# Validate with specific backend
value = BackendRegistry.validate(spec, 42, backend="rust")
```

### Backend Protocol

Interface all backends must implement.

```python
class Backend(Protocol):
    def validate(self, spec: FieldSpec, value: Any) -> Any
    def create_field(self, spec: FieldSpec) -> Any
```

## Common Constraints

### Numeric Constraints
- `min`: Minimum value (inclusive)
- `max`: Maximum value (inclusive)

### String Constraints
- `pattern`: Regex pattern
- `min_length`: Minimum string length
- `max_length`: Maximum string length

### General Constraints
- `nullable`: Allow None value
- `description`: Field description
- `default`: Default value

## Integration with Existing LionAGI

```python
from lionagi.models import OperableModel
from lionagi.specs_v2 import FieldSpec

# Define field specs
instruct_spec = FieldSpec(str, {"description": "Task instruction"})
context_spec = FieldSpec(dict, {})

# Use with OperableModel
model = OperableModel()
model.add_field("instruction", field_spec=instruct_spec)
model.add_field("context", field_spec=context_spec)

RequestModel = model.new_model(name="Request")
```

## Differences from LSpec v1

| Feature | v1 (Current) | v2 (This) |
|---------|-------------|-----------|
| LOC | 800-1600 | ~400 |
| Constraints | frozenset of tuples | dict |
| Category theory | Explicit (Morphism, etc.) | Hidden/removed |
| Capability system | Included | Removed (future) |
| Backend system | Partial | Full plugin architecture |
| Serialization | Not built-in | Built-in to_dict/from_dict |
| Multi-backend | Planned | Implemented |

## Roadmap

### v2.0-alpha (Current)
- ✅ Core FieldSpec
- ✅ BackendRegistry
- ✅ PydanticBackend (functional)
- ✅ RustBackend (stub)
- ✅ CloudBackend (stub)

### v2.0-beta (Week 1)
- [ ] Integration tests with existing LionAGI
- [ ] Performance benchmarks
- [ ] Documentation improvements
- [ ] Migration guide from v1

### v2.1 (Month 1)
- [ ] RustBackend implementation (PyO3)
- [ ] Formal verification integration
- [ ] Kani property tests

### v2.2 (Month 2-3)
- [ ] CloudBackend implementation
- [ ] Enterprise features
- [ ] Audit logging
- [ ] Compliance certifications

## Design Decisions

### Why dict instead of frozenset for constraints?

**v1**: `frozenset[tuple[str, Any]]`
**v2**: `dict[str, Any]`

**Reasons**:
1. **No duplicate keys** - Dict naturally prevents duplicate constraint keys (fixes ChatGPT report issue)
2. **JSON-friendly** - Easier serialization for backend communication
3. **O(1) lookup** - Better performance for constraint access
4. **Simpler** - More Pythonic and easier to understand

### Why remove category theory?

Category theory (Morphism, Functor, etc.) adds significant complexity without clear benefit for the Python distribution layer. The formal properties belong in Rust/Cloud backends where they can be properly verified.

**Kept**: Compositional operations (as_nullable, as_listable, with_constraint)
**Removed**: Explicit category theory classes and terminology

### Why no capability system yet?

Capability-based security is planned for future but not needed for initial multi-backend system. It will be added when multi-trust domain scenarios are validated.

## Contributing

This is alpha software for internal review. Feedback welcome on:

1. API ergonomics
2. Backend abstraction design
3. Integration with existing LionAGI code
4. Performance characteristics
5. Missing features

## License

Apache 2.0 (same as LionAGI)
