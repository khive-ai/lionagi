# LionAGI V1 Services Performance Test Suite

This directory contains comprehensive P2 performance benchmarks and validation tests for the lionagi v1 services architecture. These tests validate the performance characteristics that matter for production Agent Kernel usage.

## Overview

The performance test suite validates the key architectural decisions of lionagi v1:

1. **msgspec over Pydantic** for serialization performance
2. **Structured concurrency** with AnyIO for reliable execution
3. **Efficient queuing** using memory object streams
4. **Streaming performance** with no buffering regression
5. **Circuit breaker and hooks** with minimal latency impact

## Test Categories

### 1. Msgspec Performance (`test_performance_msgspec.py`)

Validates measurable gains from v1 architectural decision to use msgspec over Pydantic.

**Key Tests:**
- Core structure serialization (CallContext, ServiceCall, ExecutorConfig)
- Transport JSON parsing (4KB, 64KB, 1MB response payloads)
- Batch serialization performance
- Memory usage efficiency
- Validation performance comparison

**Success Criteria:**
- msgspec ≥2x faster than Pydantic for serialization
- ≥20% memory improvement over Pydantic
- Consistent performance across payload sizes

### 2. Executor Performance (`test_performance_executor.py`)

Tests executor throughput, queue efficiency, and memory usage under realistic load.

**Key Tests:**
- High throughput fast calls (≥50 calls/sec target)
- Mixed load performance (fast + slow calls)
- Queue latency measurement (no polling delays)
- Memory usage under sustained load
- Overhead measurement vs direct service calls
- Rate limiting accuracy under concurrent load

**Success Criteria:**
- Efficient queuing with <10ms submission latency
- Memory usage bounded under sustained load
- Rate limiting accurate within ±20%
- Minimal overhead (<5x direct call time)

### 3. Streaming Performance (`test_performance_streaming.py`)

Validates streaming performance characteristics for real-time applications.

**Key Tests:**
- Time-to-first-byte (TTFB) measurement
- Sustained throughput (10k small chunks)
- Memory consumption (no buffering regression)
- Concurrent streams performance
- Large payload streaming (1MB+)

**Success Criteria:**
- TTFB <50ms under normal load
- ≥1000 chunks/sec sustained throughput
- Memory usage not scaling with stream size
- No degradation with circuit breaker active

### 4. Integration Performance (`test_performance_integration.py`)

End-to-end pipeline performance with all middleware enabled.

**Key Tests:**
- Full pipeline latency measurement
- Concurrent request handling scalability
- Mixed workload handling (calls + streams)
- Burst traffic patterns
- Resource utilization monitoring

**Success Criteria:**
- Pipeline latency <300ms including middleware
- Linear scalability up to concurrency limits
- Bounded memory usage under load
- Graceful handling of traffic bursts

## Running the Tests

### Prerequisites

Install required dependencies:
```bash
uv add --dev pytest pytest-anyio pytest-benchmark psutil msgspec pydantic
```

### Run All Performance Tests

```bash
# Run all performance tests
pytest tests/services/test_performance_*.py -v

# Run with benchmark output
pytest tests/services/test_performance_*.py --benchmark-only
```

### Run Specific Test Categories

```bash
# msgspec benchmarks only
pytest tests/services/test_performance_msgspec.py -v

# Executor performance only  
pytest tests/services/test_performance_executor.py -v

# Streaming performance only
pytest tests/services/test_performance_streaming.py -v  

# Integration performance only
pytest tests/services/test_performance_integration.py -v
```

### Advanced Options

```bash
# Run with detailed benchmark statistics
pytest tests/services/ --benchmark-only --benchmark-verbose

# Skip slow tests, run core benchmarks only
pytest tests/services/ -m "not slow" --benchmark-only

# Save benchmark results for comparison
pytest tests/services/ --benchmark-only --benchmark-save=baseline

# Compare with previous baseline
pytest tests/services/ --benchmark-only --benchmark-compare=baseline
```

## Performance Thresholds

The tests include specific performance thresholds defined in `conftest.py`:

```python
PERFORMANCE_THRESHOLDS = {
    "msgspec_vs_pydantic_min_improvement": 2.0,  # 2x faster minimum
    "executor_min_throughput_fast_calls": 50,    # 50 calls/sec minimum
    "streaming_min_chunks_per_second": 1000,     # 1000 chunks/sec
    "streaming_max_ttfb_ms": 50,                 # 50ms TTFB maximum
    "pipeline_max_latency_ms": 300,              # 300ms pipeline max
    "concurrent_max_degradation": 0.3,           # 30% max degradation
}
```

## Key Validations

### Architecture Decision Validation

1. **Structured Concurrency**: Tests validate that executor uses TaskGroup for lifecycle management with no leaked tasks or orphaned operations.

2. **Efficient Queuing**: Validates that memory object streams eliminate polling delays and provide immediate queue feedback.

3. **msgspec Performance**: Demonstrates measurable serialization performance gains over Pydantic baseline.

4. **Streaming Efficiency**: Confirms streaming doesn't buffer entire responses and maintains consistent memory usage.

### Production Readiness Indicators

- **Throughput**: Can handle realistic concurrent loads (20+ calls/sec)
- **Latency**: TTFB and pipeline latencies suitable for interactive applications
- **Memory**: Bounded memory usage that doesn't scale with request volume
- **Reliability**: Graceful handling of failures and resource cleanup

### Test Quality Standards

Following Ocean's requirement for **meaningful tests that validate actual behavior**:

- Tests use realistic data sizes and request patterns
- Performance assertions include statistical validation
- Memory usage measured with actual tooling (tracemalloc, psutil)
- Concurrent execution tested with actual task groups
- Error scenarios and resource cleanup validated

## Interpreting Results

### Benchmark Output

pytest-benchmark provides detailed timing statistics:

```
Name (time in ms)                     Min      Max     Mean   StdDev
test_msgspec_serialization           0.12     0.18     0.14    0.02  
test_pydantic_serialization          0.28     0.35     0.31    0.03
```

### Key Metrics to Monitor

1. **Serialization Performance**: msgspec should consistently outperform Pydantic by 2x+
2. **Executor Throughput**: Should scale linearly with concurrency up to limits
3. **Memory Efficiency**: Peak memory usage should be bounded regardless of load
4. **TTFB Consistency**: Streaming TTFB should remain stable under concurrent load

## Troubleshooting

### Common Issues

1. **High Variance in Results**: 
   - Disable garbage collection during benchmarks
   - Run with sufficient warmup iterations
   - Close other applications to reduce system noise

2. **Memory Tests Failing**:
   - Ensure adequate system memory (8GB+ recommended)
   - Run garbage collection between test phases
   - Check for memory leaks in test setup

3. **Concurrency Tests Unstable**:
   - Verify system can handle test concurrency levels
   - Check for resource limits (file descriptors, etc.)
   - Reduce concurrency if system constrained

### Performance Regression Detection

If tests consistently fail performance thresholds:

1. **Profile Recent Changes**: Use profiling tools to identify bottlenecks
2. **Compare Baseline**: Run benchmark comparison with known good baseline
3. **Check Resource Usage**: Monitor CPU, memory, and I/O during tests
4. **Validate Test Environment**: Ensure consistent test execution environment

## Continuous Integration

For CI/CD integration:

```bash
# Fast performance smoke tests
pytest tests/services/ -m "performance and not slow" --benchmark-skip

# Full performance validation (longer running)
pytest tests/services/ --benchmark-only --benchmark-min-rounds=5
```

## Contributing

When adding new performance tests:

1. **Follow Naming Convention**: `test_performance_[category]_[specific_test].py`
2. **Include Performance Assertions**: Use specific thresholds, not just "faster"  
3. **Test Realistic Scenarios**: Use actual data sizes and patterns
4. **Document Expected Performance**: Include success criteria in docstrings
5. **Validate Resource Cleanup**: Ensure tests don't leak memory or tasks

## References

- [TDD Specification](../../docs/design/services_tdd.md) - Comprehensive test plan
- [V1 Architecture](../../docs/design/initial_design.md) - Core design decisions
- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)
- [AnyIO Documentation](https://anyio.readthedocs.io/) - Structured concurrency
