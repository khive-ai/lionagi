# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Lionagi V1 Services Test Suite

Comprehensive P0 tests for the lionagi services layer focusing on critical flaws
and real behavior validation as specified in the TDD documentation.

Test Structure:
- test_executor_reliability.py: CRITICAL ExecutorQueueWaitDeadline flaw validation
- test_executor_lifecycle.py: Structured concurrency and TaskGroup management
- test_executor_rate_limiting.py: Rate limiting accuracy and enforcement
- conftest.py: Shared fixtures, mock services, and test utilities

Key Features:
- pytest-anyio compatibility (asyncio and trio backends)
- Comprehensive mock services for various testing scenarios
- Timing constraints and rate limiting validation
- Structured concurrency leak detection
- Statistical accuracy validation under concurrent load

Critical Tests:
- ExecutorQueueWaitDeadline: Validates deadline-aware waiting in _wait_for_capacity
- StructuredShutdownUnderLoad: Validates proper TaskGroup cleanup
- RateLimitAccuracyAndSafety: Validates thread-safe rate limiting counters
- CancellationPropagation: Validates cancellation through entire stack
"""

__version__ = "1.0.0"
__author__ = "HaiyangLi (Ocean)"

# Re-export commonly used test utilities
# Temporarily commented out for middleware test updates
# from .conftest import (  # Request models; Mock services; Test executor; Utilities; Assertions
#     EchoService,
#     HeavyRequest,
#     LightRequest,
#     MockService,
#     ProgressiveFailureService,
#     StatsCollector,
#     TestExecutor,
#     TestRequest,
#     TimingContext,
#     assert_call_completed_successfully,
#     assert_rate_limiting_effective,
#     assert_stats_consistency,
#     create_test_context,
#     create_test_request,
#     expect_timing,
#     submit_and_wait,
#     submit_multiple_and_wait,
# )

__all__ = [
    # Request models
    "TestRequest",
    "HeavyRequest",
    "LightRequest",
    # Mock services
    "MockService",
    "EchoService",
    "ProgressiveFailureService",
    # Test executor
    "TestExecutor",
    # Utilities
    "create_test_context",
    "create_test_request",
    "submit_and_wait",
    "submit_multiple_and_wait",
    "StatsCollector",
    "TimingContext",
    "expect_timing",
    # Assertions
    "assert_call_completed_successfully",
    "assert_stats_consistency",
    "assert_rate_limiting_effective",
]
