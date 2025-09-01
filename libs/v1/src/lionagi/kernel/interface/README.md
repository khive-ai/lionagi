# Interface: The Boundary Layer

Clean separation between Python distribution and Rust performance.

## Philosophy

Ocean: "python is like our distribution"

The interface layer defines the exact boundary where Python meets Rust, enabling seamless backend swapping.

## Core Design Pattern

```python
# The Contract (Protocol)
class BackendProtocol(Protocol):
    """What every backend must provide"""
    async def observe(self, observable: Observable) -> Observation: ...
    async def validate(self, value: Any, invariants: list) -> bool: ...
    async def execute(self, instruction: Instruct) -> Response: ...

# Python Implementation (Open Core)
class PythonBackend:
    async def observe(self, observable: Observable) -> Observation:
        # Pure Python logic
        return Observation(valid=True)

# Rust Implementation (Enterprise)  
class RustBackend:
    async def observe(self, observable: Observable) -> Observation:
        # Calls Rust via PyO3
        # Formally verified with proofs
        return await rust_lib.observe(msgspec.to_builtins(observable))
```

## The Backend Registry

```python
class BackendRegistry:
    """Global registry for swappable backends"""
    
    _backends: dict[str, BackendProtocol] = {
        "python": PythonBackend(),
        # "rust": RustBackend() - available in enterprise
    }
    
    @classmethod
    def set_default(cls, name: str):
        """Set global default backend"""
        cls._default = cls._backends[name]
    
    @classmethod
    def get(cls) -> BackendProtocol:
        """Get current backend"""
        return cls._default
```

## Serialization Bridge

Critical for Python↔Rust communication:

```python
class SerializationBridge:
    """Handles type conversion at boundary"""
    
    @staticmethod
    def to_rust(obj: Any) -> bytes:
        """Python → Rust (msgspec)"""
        if isinstance(obj, Observable):
            # Convert to dict preserving structure
            return msgspec.encode(obj.to_dict())
        return msgspec.encode(obj)
    
    @staticmethod
    def from_rust(data: bytes) -> Any:
        """Rust → Python (msgspec)"""
        return msgspec.decode(data)
```

## Capability Mapping

Maps Python capabilities to Rust's formal model:

```python
class CapabilityBridge:
    """Maps Python caps to formal capability algebra"""
    
    # Python capability (string)
    PYTHON_CAPS = {"read", "write", "execute", "observe"}
    
    # Rust capability (formal set)
    # Maps to Chapter 2: Capability Security
    RUST_CAPS = {
        "read": 0b0001,     # R ∈ 2^Auth
        "write": 0b0010,    # W ∈ 2^Auth  
        "execute": 0b0100,  # X ∈ 2^Auth
        "observe": 0b1000,  # O ∈ 2^Auth
    }
    
    @staticmethod
    def to_rust(caps: set[str]) -> int:
        """Convert Python caps to Rust bitfield"""
        result = 0
        for cap in caps:
            result |= CapabilityBridge.RUST_CAPS.get(cap, 0)
        return result
```

## Error Translation

Rust panics/errors → Python exceptions:

```python
class ErrorBridge:
    """Translates Rust errors to Python"""
    
    @staticmethod
    def translate(rust_error: RustError) -> Exception:
        match rust_error.kind:
            case "CapabilityDenied":
                return PermissionError(f"Capability denied: {rust_error.detail}")
            case "InvariantViolation":
                return ValueError(f"Invariant violated: {rust_error.detail}")
            case "DeadlockDetected":
                return RuntimeError("Deadlock detected (this shouldn't happen!)")
```

## Performance Telemetry

Track speedup from Rust backend:

```python
class PerformanceBridge:
    """Measures Python vs Rust performance"""
    
    @contextmanager
    def measure(self, operation: str):
        backend = BackendRegistry.get().__class__.__name__
        start = time.perf_counter_ns()
        yield
        duration = time.perf_counter_ns() - start
        
        # Log to telemetry
        metrics.record(
            operation=operation,
            backend=backend,
            duration_ns=duration
        )
```

## The PyO3 Connection

For enterprise version:

```python
# lionagi_enterprise/rust_bridge.py
import lionagi_rust  # PyO3 module

class RustBackend:
    def __init__(self):
        # Initialize Rust runtime
        self._runtime = lionagi_rust.Runtime()
        
    async def observe(self, observable: Observable) -> Observation:
        # Serialize to msgspec
        data = SerializationBridge.to_rust(observable)
        
        # Call Rust (releases GIL)
        result = await self._runtime.observe(data)
        
        # Deserialize response
        return SerializationBridge.from_rust(result)
```

## Backend Feature Detection

```python
class FeatureDetection:
    """Detect backend capabilities"""
    
    @staticmethod
    def supported_features() -> dict[str, bool]:
        backend = BackendRegistry.get()
        
        return {
            "formal_verification": hasattr(backend, "get_proofs"),
            "simd_optimization": hasattr(backend, "simd_enabled"),
            "gpu_acceleration": hasattr(backend, "cuda_available"),
            "deadlock_freedom": isinstance(backend, RustBackend),
        }
```

## Configuration

```python
# Default configuration
class InterfaceConfig:
    # Use Python by default
    DEFAULT_BACKEND = "python"
    
    # Serialization format
    SERIALIZATION = "msgspec"  # or "json", "pickle"
    
    # Performance tracking
    ENABLE_TELEMETRY = True
    
    # Error verbosity
    DETAILED_ERRORS = True
```

## Why This Architecture Wins

### For Open Source Users
- Zero dependency on Rust
- Pure Python works perfectly
- Clean, understandable code
- No vendor lock-in

### For Enterprise Customers  
- Drop-in Rust backend
- 10-100x performance boost
- Formal verification reports
- Mathematical guarantees

### For NVIDIA Specifically
- Seamless NIM integration
- GPU acceleration ready
- Proven correctness for mission-critical
- Your connections make this real

## The Sales Pitch

"Start with Python, scale with Rust. Same code, proven correctness."

## Testing Strategy

```python
class BackendTests:
    """Test both backends behave identically"""
    
    @parameterized(["python", "rust"])
    async def test_observation_equivalence(self, backend_name):
        BackendRegistry.set_default(backend_name)
        
        observable = create_test_observable()
        observation = await BackendRegistry.get().observe(observable)
        
        # Both backends must produce same result
        assert observation.valid == expected_valid
        assert observation.capabilities == expected_caps
```

## Migration Path

1. **Phase 1**: Pure Python (current)
2. **Phase 2**: Rust backend available (enterprise)
3. **Phase 3**: GPU backend (NVIDIA partnership)
4. **Phase 4**: Distributed backend (scale-out)

Ocean, this interface layer is the key to monetization - same API, swappable performance, proven correctness.