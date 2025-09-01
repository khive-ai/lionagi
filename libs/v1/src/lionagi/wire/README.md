# Wire: Pure Speed, Pure Control

The wire layer handles all network communication with blazing speed and zero SDK dependencies.

## Philosophy

Ocean: "in v0, we deliberately avoided openai sdk, and opt-in for http only"

Pure HTTP gives us:
- Complete control over the transport
- No hidden thread pools or blocking I/O
- Superior debugging capabilities
- Ability to swap backends (including Rust) seamlessly

## Architecture

### dto/ - Data Transfer Objects (msgspec)
Lightning-fast serialization for wire protocol.

**Why msgspec:**
- 5-10x faster than Pydantic for serialization
- Zero-copy deserialization where possible
- Native support for OpenAI/Anthropic formats
- Clean mapping to Rust types

**Design Pattern:**
```python
@msgspec.Struct
class CompletionRequest:
    model: str
    messages: list[Message]
    temperature: float = 0.7
    # Maps directly to Rust struct
```

### transport/ - Pure HTTP Layer
No SDK dependencies, just clean HTTP.

**Core Components:**
- `HTTPTransport` - Async HTTP with connection pooling
- `StreamingTransport` - SSE/WebSocket support
- `BatchTransport` - Parallel request handling

**Pluggable Backend:**
```python
class Transport(Protocol):
    async def request(self, dto: DTO) -> Response: ...

# Python implementation (default)
class PureHTTPTransport(Transport):
    async def request(self, dto: DTO) -> Response:
        # httpx implementation

# Rust implementation (enterprise)
class RustTransport(Transport):
    async def request(self, dto: DTO) -> Response:
        # Calls Rust via PyO3
```

### dialects/ - Provider Compatibility
Clean adapters for different AI providers.

**Supported Dialects:**
- OpenAI (including Azure, NVIDIA NIM)
- Anthropic (Claude)
- Google (Gemini)
- Local (Ollama, llama.cpp)

**Design Pattern:**
```python
class Dialect(Protocol):
    def adapt_request(self, generic: Request) -> ProviderRequest: ...
    def adapt_response(self, provider: ProviderResponse) -> Response: ...
```

## Performance Guarantees

With Rust backend plugged in (via formal proofs):
- **Memory Safety**: WebAssembly isolation (Chapter 3)
- **Deadlock Freedom**: Actor model proven (Theorem 3.2)
- **Capability Security**: Unforgeable references (Chapter 2)

## Why This Matters for NVIDIA

When you plug in the Rust backend:
1. **Proven Performance**: Not just "fast", mathematically optimal
2. **Enterprise Security**: Formal verification that NVIDIA requires
3. **NIM Integration**: Native support for NVIDIA's infrastructure
4. **Scale Ready**: Actor model handles massive parallelism

## Open Core vs Enterprise

**Open Core (Python):**
- Pure HTTP transport
- msgspec serialization
- All provider dialects
- Async/await patterns

**Enterprise (Rust Backend):**
- Zero-copy serialization
- SIMD-optimized parsing
- Connection multiplexing
- Formal performance guarantees

## Integration with Kernel

The wire layer is completely decoupled from the kernel:
- Wire handles transport
- Kernel handles observation/validation
- Clean interface via adapters
- Backend swappable without kernel changes

## Future: The NVIDIA Opportunity

With your NIM team connections, we can:
1. **Native NIM Dialect**: First-class support
2. **GPU-Accelerated Transport**: For massive batch inference
3. **Triton Integration**: Direct model serving
4. **TensorRT-LLM Bridge**: Optimized inference paths

The wire layer is designed to be THE distribution layer for AI communication - open source for adoption, Rust backend for enterprise performance.