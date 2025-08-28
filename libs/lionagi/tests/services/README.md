# Lionagi V1 Services Test Suite

Comprehensive P0 tests for the RateLimitedExecutor focusing on critical flaws and real behavior validation as specified in the TDD documentation.

## Overview

This test suite implements the V1_Executor test specifications with emphasis on catching critical flaws that Ocean identified, particularly the **deadline-unaware waiting flaw** in the `_wait_for_capacity` method.

## Critical Tests

### 1. ExecutorQueueWaitDeadline (CRITICAL FLAW VALIDATION)
**File**: `test_executor_reliability.py`  
**Function**: `test_executor_queue_wait_deadline_critical_flaw()`

**What it validates**: The critical flaw where the executor waits for rate limit capacity but doesn't respect the call deadline. When rate limits force a wait time longer than the call deadline, the call should fail promptly with TimeoutError rather than waiting the full rate limit time.

**Why it's critical**: This flaw violates deadline guarantees and can cause calls to hang much longer than intended, breaking timeout contracts.

### 2. Structured Concurrency Tests
**File**: `test_executor_lifecycle.py`

- **StructuredStartupAndShutdown**: Validates TaskGroup usage and clean shutdown
- **StructuredShutdownUnderLoad**: Tests shutdown with active/queued calls
- **CancellationPropagation**: Ensures cancellation propagates through entire stack

### 3. Rate Limiting Accuracy Tests  
**File**: `test_executor_rate_limiting.py`

- **RateLimitAccuracyAndSafety**: Stress test for thread-safe counters under concurrent load
- **Request/Token Limit Enforcement**: Validates proper rate limiting behavior
- **Combined Rate Limits**: Tests interaction of request and token limits

## Test Structure

```
tests/services/
├── __init__.py              # Package exports and documentation
├── conftest.py              # Shared fixtures, mock services, utilities
├── pytest.ini              # Test configuration and markers
├── test_executor_lifecycle.py    # P0 structured concurrency tests
├── test_executor_reliability.py  # P0 critical flaw validation  
├── test_executor_rate_limiting.py # P0 rate limiting accuracy
└── README.md               # This file
```

## Running Tests

### Basic Test Execution
```bash
# Run all services tests
pytest libs/lionagi/tests/services/

# Run with both asyncio and trio backends (recommended)
pytest libs/lionagi/tests/services/ --anyio-backends=asyncio,trio

# Run only P0 (critical) tests
pytest libs/lionagi/tests/services/ -m p0

# Run only the critical deadline flaw test
pytest libs/lionagi/tests/services/test_executor_reliability.py::test_executor_queue_wait_deadline_critical_flaw
```

### Test Categories

```bash
# Run by priority level
pytest -m p0                    # P0: Must pass to ship
pytest -m p1                    # P1: Integration tests
pytest -m p2                    # P2: Performance tests

# Run by test type  
pytest -m reliability           # Critical reliability flaws
pytest -m lifecycle             # Structured concurrency
pytest -m rate_limiting         # Rate limiting accuracy
pytest -m deadline              # Deadline awareness
pytest -m concurrency           # Thread safety
pytest -m cancellation          # Cancellation propagation

# Run by speed
pytest -m fast                  # Quick tests (< 1s)
pytest -m "not slow"            # Skip slow tests
```

### Backend Testing
```bash
# Test on specific backend
pytest --anyio-backends=asyncio
pytest --anyio-backends=trio

# Backend compatibility tests only
pytest -m backend_agnostic
```

## Test Fixtures and Utilities

### Mock Services
- **MockService**: Highly configurable service with delays, failures, hanging
- **EchoService**: Simple echo for basic functionality tests
- **ProgressiveFailureService**: Increasing failure probability over time

### Test Utilities
- **TestExecutor**: Enhanced executor with proper token estimation
- **create_test_context()**: Helper for creating CallContext
- **StatsCollector**: For analyzing executor statistics over time
- **TimingContext**: For measuring execution timing
- **expect_timing()**: Async context manager for timing assertions

### Assertion Helpers
- **assert_call_completed_successfully()**: Validates successful completion
- **assert_stats_consistency()**: Validates internal stat consistency  
- **assert_rate_limiting_effective()**: Validates rate limiting worked

## Key Test Validations

### 1. Deadline Awareness
- Calls fail promptly when deadlines expire during rate limit waiting
- No hanging on expired deadlines
- Proper TimeoutError propagation

### 2. Structured Concurrency
- TaskGroup properly manages all spawned tasks
- Clean shutdown with no orphaned tasks
- Proper cancellation propagation through call stack

### 3. Rate Limiting Accuracy
- Request/token counters accurate under concurrent load
- No race conditions in rate limiting logic
- Proper batching across refresh periods

### 4. Concurrency Safety
- Thread-safe access to shared state
- No data races in statistics updates
- Proper locking around critical sections

### 5. Memory Management
- Memory object streams properly closed
- No resource leaks during shutdown
- Proper cleanup of completed calls

## Expected Behavior

### Rate Limiting
- Requests should be batched according to limits
- Token usage should be accurately tracked
- Combined limits should work correctly

### Lifecycle Management  
- Executor starts and stops cleanly
- All calls reach terminal states
- No active calls remain after shutdown

### Error Handling
- Service failures are properly caught and reported
- Cancellations propagate correctly
- Stats reflect all outcomes accurately

## Performance Expectations

### Timing Constraints
- Queue pickup: < 50ms (validates no polling)
- Rate limit accuracy: ±300ms tolerance
- Shutdown: < 5s under normal load

### Throughput
- Should handle 100+ concurrent requests
- Memory usage should remain bounded
- No significant memory leaks over time

## Debugging Failed Tests

### Common Issues
1. **Timing failures**: Check system load, may need to adjust tolerances
2. **Backend failures**: Some tests may be backend-specific
3. **Resource cleanup**: Ensure proper test isolation

### Diagnostic Information
- All tests include detailed timing information
- Stats snapshots for analyzing executor state
- Call history tracking in mock services
- Structured logging for debugging

## Integration with CI/CD

### Required Test Passes
- All P0 tests must pass for deployment
- Both asyncio and trio backends must pass
- Coverage must be ≥85%

### Test Markers for CI
```bash
# Critical path - must always pass
pytest -m "p0 and not slow"

# Full validation - nightly builds  
pytest -m "p0 or p1" --anyio-backends=asyncio,trio

# Performance validation - weekly
pytest -m p2
```

This test suite ensures that the lionagi v1 services layer meets the rigorous standards required for production deployment while catching the critical flaws that could cause system instability.