# Services: High-Level APIs for Humans

The services layer provides elegant, high-level APIs that compose the lower layers into useful patterns.

## Philosophy

Services are what developers actually use. They hide complexity while preserving power.

Ocean: "composability is the sh*t of our framework"

Services compose kernel, domain, and wire into patterns developers love.

## Core Services

### Session Service
Manages conversations with full observation chain.

```python
async with Session() as session:
    # Automatically observes, validates, and tracks
    response = await session.chat("Hello")
    # IPU validates, capabilities checked, effects applied
```

### Branch Service  
Isolated execution contexts with capability management.

```python
branch = Branch(capabilities={"read", "write"})
async with branch:
    # All operations capability-checked
    result = await branch.operate(instruction)
```

### Orchestration Service
Multi-agent coordination with formal guarantees.

```python
orchestrator = Orchestrator()
# Parallel execution with deadlock freedom (Theorem 3.2)
results = await orchestrator.map_reduce(
    agents=[researcher, analyst, critic],
    task="Analyze codebase"
)
```

### Tool Service
Function calling with capability awareness.

```python
@tool(capabilities={"filesystem:read"})
async def read_file(path: str) -> str:
    # Automatically capability-checked by IPU
    return Path(path).read_text()
```

### Memory Service
Temporal state with decay and importance.

```python
memory = Memory()
await memory.store("key_insight", value, importance=0.9)
# Automatically decays based on access patterns
```

## Pluggable Backend Architecture

### How It Works

```python
# Default: Pure Python
from lionagi.services import Session

# Enterprise: With Rust backend
from lionagi_enterprise import RustBackend
from lionagi.services import Session

Session.set_backend(RustBackend())  # Drop-in replacement
```

### What Changes with Rust Backend

**Python (Open Core):**
- Async/await execution
- Python-based validation
- Standard performance

**Rust (Enterprise):**
- Proven deadlock freedom
- Formal capability verification  
- 10-100x performance
- Mathematical security guarantees

### The Abstraction Boundary

Services define the interface, backends provide implementation:

```python
class SessionBackend(Protocol):
    """What every backend must provide"""
    
    async def observe(self, observable: Observable) -> Observation:
        """Rust: Formally verified IPU"""
        
    async def validate(self, value: Any, invariants: list[Invariant]) -> bool:
        """Rust: Parallel SIMD validation"""
        
    async def execute(self, branch: Branch, instruction: Instruct) -> Response:
        """Rust: Actor model with deadlock freedom"""
```

## Integration Patterns

### With NVIDIA NIM

```python
# Leverages your team connections
from lionagi.services import Session
from lionagi.wire.dialects import NVIDIADialect

session = Session(dialect=NVIDIADialect())
# Native NIM support, ready for GPU acceleration
```

### With Enterprise Workflows

```python
from lionagi.services import Workflow

# Three-valued policy logic (Chapter 4)
workflow = Workflow(
    policy=Policy(
        allow=["read", "analyze"],
        deny=["write", "delete"],
        unknown="ask"  # Three-valued logic
    )
)
```

## Why Services Matter

1. **Developer Experience**: Clean APIs that feel natural
2. **Power Preserved**: Full access to formal guarantees
3. **Backend Agnostic**: Same code works with Python or Rust
4. **Enterprise Ready**: Capability-aware, policy-driven, auditable

## The Commercial Opportunity

For NVIDIA or enterprise customers:

**Level 1: Open Core**
- All services available
- Python backend
- Community support

**Level 2: Enterprise** 
- Rust backend plugin
- Formal verification reports
- SLA guarantees

**Level 3: Custom Integration**
- Direct NIM integration
- Custom capability policies
- On-premise deployment

## Design Principles

1. **Services compose, not inherit** - Use protocols not base classes
2. **Explicit over implicit** - Capabilities always visible
3. **Graceful degradation** - Works without Rust, better with it
4. **Observable by default** - Everything goes through IPU

## Future Services

- **Training Service**: Fine-tuning with formal convergence guarantees
- **Deployment Service**: Model serving with capability isolation  
- **Analytics Service**: Observability with causal tracing
- **Compliance Service**: Regulatory audit trails

The services layer is where the magic happens - where formal verification meets developer ergonomics.