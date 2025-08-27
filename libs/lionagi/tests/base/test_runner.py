"""Test suite for runner.py (The Execution Kernel) - TDD Specification Implementation.

Focus: The core integration tests. Validates structured concurrency, security enforcement, and IPU integration during graph execution.
"""

import time
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import anyio
import pytest

from lionagi.base.graph import OpGraph, OpNode
from lionagi.base.ipu import LenientIPU, StrictIPU, default_invariants
from lionagi.base.morphism import Morphism
from lionagi.base.runner import Runner
from lionagi.base.types import Branch, create_branch
from lionagi.ln.concurrency import create_task_group, fail_after, get_cancelled_exc_class


# Helper functions
def create_test_runner():
    """Create a Runner with default LenientIPU for testing."""
    return Runner(LenientIPU(default_invariants()))


# Mock morphism implementations for testing
class FastMorphism(Morphism):
    """Morphism that completes quickly."""

    def __init__(self, name: str, duration: float = 0.01):
        self.name = name
        self.duration = duration
        self.requires = set()

    async def pre(self, branch, **kwargs) -> bool:
        return True

    async def apply(self, branch, **kwargs) -> dict:
        await anyio.sleep(self.duration)
        return {"result": f"completed_{self.name}"}

    async def post(self, branch, result, **kwargs) -> bool:
        return True


class SlowMorphism(Morphism):
    """Morphism that takes a long time to complete."""

    def __init__(self, name: str, duration: float = 2.0):
        self.name = name
        self.duration = duration
        self.requires = set()

    async def pre(self, branch, **kwargs) -> bool:
        return True

    async def apply(self, branch, **kwargs) -> dict:
        await anyio.sleep(self.duration)
        return {"result": f"slow_completed_{self.name}"}

    async def post(self, branch, result, **kwargs) -> bool:
        return True


class FailingMorphism(Morphism):
    """Morphism that fails during execution."""

    def __init__(self, name: str):
        self.name = name
        self.requires = set()

    async def pre(self, branch, **kwargs) -> bool:
        return True

    async def apply(self, branch, **kwargs) -> dict:
        raise RuntimeError(f"Morphism {self.name} failed intentionally")

    async def post(self, branch, result, **kwargs) -> bool:
        return True


class PreConditionFailMorphism(Morphism):
    """Morphism that fails pre-condition check."""

    def __init__(self, name: str):
        self.name = name
        self.requires = set()

    async def pre(self, branch, **kwargs) -> bool:
        return False  # Pre-condition fails

    async def apply(self, branch, **kwargs) -> dict:
        return {"result": "should_not_reach_here"}

    async def post(self, branch, result, **kwargs) -> bool:
        return True


class PostConditionFailMorphism(Morphism):
    """Morphism that fails post-condition check."""

    def __init__(self, name: str):
        self.name = name
        self.requires = set()

    async def pre(self, branch, **kwargs) -> bool:
        return True

    async def apply(self, branch, **kwargs) -> dict:
        return {"result": f"completed_{self.name}"}

    async def post(self, branch, result, **kwargs) -> bool:
        return False  # Post-condition fails


class TestRunnerExecutionFlow:
    """TestSuite: RunnerExecutionFlow - Async tests for execution flow and concurrency."""

    @pytest.mark.anyio
    async def test_parallel_execution_efficiency(self):
        """Test: ParallelExecutionEfficiency

        GIVEN a graph (A -> B, A -> C) where B and C are independent and sleep(100ms)
        WHEN Runner.run() is called
        THEN total execution time should be approx 100ms (not 200ms), confirming concurrency.
        """
        # Create nodes - A is fast, B and C are parallel slow operations
        node_a = OpNode(id=uuid4(), m=FastMorphism("A", 0.01))
        node_b = OpNode(id=uuid4(), m=SlowMorphism("B", 0.1), deps={node_a.id})  # 100ms
        node_c = OpNode(id=uuid4(), m=SlowMorphism("C", 0.1), deps={node_a.id})  # 100ms

        # Create graph with parallel structure: A -> (B, C)
        graph = OpGraph(
            nodes={node_a.id: node_a, node_b.id: node_b, node_c.id: node_c}, roots={node_a.id}
        )

        # Create branch and runner
        branch = Branch(id=uuid4())
        runner = create_test_runner()

        # Measure execution time
        start_time = time.perf_counter()
        await runner.run(branch, graph)
        end_time = time.perf_counter()

        execution_time = end_time - start_time

        # Should take ~100ms (parallel) not ~200ms (sequential)
        assert (
            execution_time < 0.15
        ), f"Parallel execution should take ~100ms but took {execution_time:.3f}s"
        assert (
            execution_time > 0.09
        ), f"Execution should take at least 100ms but took {execution_time:.3f}s"

    @pytest.mark.anyio
    async def test_dependency_resolution(self):
        """Test: DependencyResolution

        GIVEN a diamond graph (A -> B, A -> C, B/C -> D). B is slow, C is fast.
        WHEN Runner.run() is called
        THEN D must not start until both B and C have completed.
        """
        execution_order = []

        class TrackingMorphism(Morphism):
            def __init__(self, name: str, duration: float = 0.01):
                self.name = name
                self.duration = duration
                self.requires = set()

            async def pre(self, branch, **kwargs) -> bool:
                return True

            async def apply(self, branch, **kwargs) -> dict:
                execution_order.append(f"{self.name}_start")
                await anyio.sleep(self.duration)
                execution_order.append(f"{self.name}_end")
                return {"result": f"completed_{self.name}"}

            async def post(self, branch, result, **kwargs) -> bool:
                return True

        # Create diamond dependency graph
        node_a = OpNode(id=uuid4(), m=TrackingMorphism("A", 0.01))
        node_b = OpNode(id=uuid4(), m=TrackingMorphism("B", 0.1), deps={node_a.id})  # Slow
        node_c = OpNode(id=uuid4(), m=TrackingMorphism("C", 0.02), deps={node_a.id})  # Fast
        node_d = OpNode(id=uuid4(), m=TrackingMorphism("D", 0.01), deps={node_b.id, node_c.id})

        graph = OpGraph(
            nodes={node_a.id: node_a, node_b.id: node_b, node_c.id: node_c, node_d.id: node_d},
            roots={node_a.id},
        )

        branch = Branch(id=uuid4())
        runner = create_test_runner()

        await runner.run(branch, graph)

        # Verify dependency constraints
        assert "A_start" in execution_order, "A should start"
        assert "A_end" in execution_order, "A should complete"

        # B and C should both complete before D starts
        b_end_idx = execution_order.index("B_end")
        c_end_idx = execution_order.index("C_end")
        d_start_idx = execution_order.index("D_start")

        assert b_end_idx < d_start_idx, "B must complete before D starts"
        assert c_end_idx < d_start_idx, "C must complete before D starts"


class TestRunnerStructuredConcurrency:
    """TestSuite: RunnerStructuredConcurrency (CRITICAL) - Fail-fast behavior and cleanup."""

    @pytest.mark.anyio
    async def test_fail_fast_behavior(self):
        """Test: FailFastBehavior

        GIVEN a parallel graph (A, B, C)
        AND B is designed to fail quickly
        AND C is designed to run slowly
        WHEN Runner.run() is called
        THEN B fails.
        AND C must be cancelled immediately (verify C did not complete its slow run).
        AND Runner.run() raises the exception from B promptly.
        """
        completion_status = []

        class TrackingSlowMorphism(Morphism):
            def __init__(self, name: str):
                self.name = name
                self.requires = set()

            async def pre(self, branch, **kwargs) -> bool:
                return True

            async def apply(self, branch, **kwargs) -> dict:
                try:
                    await anyio.sleep(2.0)  # Long operation
                    completion_status.append(f"{self.name}_completed")
                    return {"result": f"completed_{self.name}"}
                except get_cancelled_exc_class():
                    completion_status.append(f"{self.name}_cancelled")
                    raise

            async def post(self, branch, result, **kwargs) -> bool:
                return True

        # Create parallel nodes: A (fast success), B (fast fail), C (slow)
        node_a = OpNode(id=uuid4(), m=FastMorphism("A"))
        node_b = OpNode(id=uuid4(), m=FailingMorphism("B"))
        node_c = OpNode(id=uuid4(), m=TrackingSlowMorphism("C"))

        graph = OpGraph(
            nodes={node_a.id: node_a, node_b.id: node_b, node_c.id: node_c},
            roots={node_a.id, node_b.id, node_c.id},  # All are roots (parallel)
        )

        branch = Branch(id=uuid4())
        runner = create_test_runner()

        # Runner should fail fast when B fails (ExceptionGroup from structured concurrency)
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await runner.run(branch, graph)

        # Check that the ExceptionGroup contains the expected RuntimeError
        exceptions = list(exc_info.value.exceptions)
        assert len(exceptions) == 1
        assert isinstance(exceptions[0], RuntimeError)
        assert "Morphism B failed intentionally" in str(exceptions[0])

        # Verify C was cancelled, not completed
        assert "C_cancelled" in completion_status, "Slow operation C should be cancelled"
        assert "C_completed" not in completion_status, "Slow operation C should NOT complete"

    @pytest.mark.anyio
    async def test_guaranteed_cleanup(self):
        """Test: GuaranteedCleanup

        GIVEN a graph with nodes that use try/finally for resource cleanup
        WHEN a failure occurs (as in FailFastBehavior)
        THEN verify that the finally blocks of all started nodes (even cancelled ones) executed.
        AND Runner.run() waits for this cleanup before exiting.
        """
        cleanup_status = []

        class CleanupMorphism(Morphism):
            def __init__(self, name: str, should_fail: bool = False, duration: float = 1.0):
                self.name = name
                self.should_fail = should_fail
                self.duration = duration
                self.requires = set()

            async def pre(self, branch, **kwargs) -> bool:
                return True

            async def apply(self, branch, **kwargs) -> dict:
                try:
                    cleanup_status.append(f"{self.name}_started")

                    if self.should_fail:
                        await anyio.sleep(0.01)  # Short delay before failing
                        raise RuntimeError(f"{self.name} failed")

                    await anyio.sleep(self.duration)
                    cleanup_status.append(f"{self.name}_completed")
                    return {"result": f"completed_{self.name}"}

                except get_cancelled_exc_class():
                    cleanup_status.append(f"{self.name}_cancelled")
                    raise

                finally:
                    # Simulate cleanup operations
                    cleanup_status.append(f"{self.name}_cleanup")

            async def post(self, branch, result, **kwargs) -> bool:
                return True

        # Create nodes with cleanup requirements
        node_a = OpNode(id=uuid4(), m=CleanupMorphism("A", should_fail=False, duration=0.5))
        node_b = OpNode(id=uuid4(), m=CleanupMorphism("B", should_fail=True))  # Fails quickly
        node_c = OpNode(id=uuid4(), m=CleanupMorphism("C", should_fail=False, duration=2.0))  # Slow

        graph = OpGraph(
            nodes={node_a.id: node_a, node_b.id: node_b, node_c.id: node_c},
            roots={node_a.id, node_b.id, node_c.id},
        )

        branch = Branch(id=uuid4())
        runner = create_test_runner()

        # Should fail due to B, but cleanup should be guaranteed (ExceptionGroup from structured concurrency)
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await runner.run(branch, graph)

        # Check that the ExceptionGroup contains the expected RuntimeError
        exceptions = list(exc_info.value.exceptions)
        assert len(exceptions) == 1
        assert isinstance(exceptions[0], RuntimeError)
        assert "B failed" in str(exceptions[0])

        # Verify all started nodes performed cleanup
        assert "A_cleanup" in cleanup_status, "Node A should perform cleanup"
        assert "B_cleanup" in cleanup_status, "Failing node B should perform cleanup"
        assert "C_cleanup" in cleanup_status, "Cancelled node C should perform cleanup"

        # Verify cancellation occurred for long-running task
        assert "C_cancelled" in cleanup_status, "Long-running node C should be cancelled"
        assert "C_completed" not in cleanup_status, "Cancelled node C should not complete"

    @pytest.mark.anyio
    async def test_external_cancellation(self):
        """Test: ExternalCancellation

        GIVEN a long-running graph
        WHEN the Runner.run() awaitable is externally cancelled (e.g., via a parent TaskGroup cancellation)
        THEN the entire graph must be torn down cleanly and promptly.
        """
        cancellation_status = []

        class CancellableMorphism(Morphism):
            def __init__(self, name: str):
                self.name = name
                self.requires = set()

            async def pre(self, branch, **kwargs) -> bool:
                return True

            async def apply(self, branch, **kwargs) -> dict:
                try:
                    cancellation_status.append(f"{self.name}_started")
                    await anyio.sleep(10.0)  # Very long operation
                    cancellation_status.append(f"{self.name}_completed")
                    return {"result": f"completed_{self.name}"}
                except get_cancelled_exc_class():
                    cancellation_status.append(f"{self.name}_cancelled")
                    raise
                finally:
                    cancellation_status.append(f"{self.name}_cleanup")

            async def post(self, branch, result, **kwargs) -> bool:
                return True

        # Create long-running graph
        node_a = OpNode(id=uuid4(), m=CancellableMorphism("A"))
        node_b = OpNode(id=uuid4(), m=CancellableMorphism("B"))

        graph = OpGraph(nodes={node_a.id: node_a, node_b.id: node_b}, roots={node_a.id, node_b.id})

        branch = Branch(id=uuid4())
        runner = create_test_runner()

        # Test cancellation by using timeout that expires quickly
        # External cancellation can manifest as either TimeoutError or CancelledException
        with pytest.raises((get_cancelled_exc_class(), TimeoutError)):
            with fail_after(0.05):  # Short timeout to force cancellation
                await runner.run(branch, graph)

        # Verify nodes were cancelled and cleaned up
        assert "A_started" in cancellation_status, "Node A should start"
        assert "A_cancelled" in cancellation_status, "Node A should be cancelled"
        assert "A_cleanup" in cancellation_status, "Node A should perform cleanup"

        assert "B_started" in cancellation_status, "Node B should start"
        assert "B_cancelled" in cancellation_status, "Node B should be cancelled"
        assert "B_cleanup" in cancellation_status, "Node B should perform cleanup"

        # Should not complete normally
        assert "A_completed" not in cancellation_status, "Node A should not complete"
        assert "B_completed" not in cancellation_status, "Node B should not complete"


class TestRunnerSecurityEnforcement:
    """TestSuite: RunnerSecurityEnforcement (CRITICAL) - Dynamic rights enforcement and fail-closed behavior."""

    @pytest.mark.anyio
    async def test_dynamic_rights_enforcement(self):
        """Test: DynamicRightsEnforcement

        GIVEN a Branch with capabilities={"net.out:safe.com"}
        AND an HTTPGet node configured to calculate dynamic rights based on the URL
        WHEN the node executes with URL="api.safe.com"
        THEN the Runner must call required_rights, get {"net.out:api.safe.com"}, and validate it successfully.
        WHEN the node executes with URL="api.unsafe.com"
        THEN validation must fail and raise PermissionError.
        """

        class DynamicRightsMorphism(Morphism):
            def __init__(self, url: str):
                self.name = f"DynamicRightsMorphism_{url}"
                self.url = url
                self.requires = set()

            def required_rights(self, **kwargs):
                # Dynamic rights calculation based on URL
                return {f"net.out:{self.url}"}

            async def pre(self, branch, **kwargs) -> bool:
                return True

            async def apply(self, branch, **kwargs) -> dict:
                return {"result": f"fetched_from_{self.url}"}

            async def post(self, branch, result, **kwargs) -> bool:
                return True

        # Test case 1: Valid URL within capabilities
        branch_valid = create_branch(
            id=uuid4(), capabilities={"net.out:safe.com", "net.out:api.safe.com"}
        )

        node_valid = OpNode(id=uuid4(), m=DynamicRightsMorphism("api.safe.com"))
        graph_valid = OpGraph(nodes={node_valid.id: node_valid}, roots={node_valid.id})

        runner_valid = create_test_runner()

        # Should execute successfully
        try:
            await runner_valid.run(branch_valid, graph_valid)
        except PermissionError:
            pytest.fail("Valid URL should pass security validation")

        # Test case 2: Invalid URL outside capabilities
        branch_invalid = create_branch(
            id=uuid4(),
            capabilities={"net.out:safe.com"},  # Does not include unsafe.com
        )

        node_invalid = OpNode(id=uuid4(), m=DynamicRightsMorphism("api.unsafe.com"))
        graph_invalid = OpGraph(nodes={node_invalid.id: node_invalid}, roots={node_invalid.id})

        runner_invalid = create_test_runner()

        # Should fail with PermissionError (may be wrapped in ExceptionGroup)
        with pytest.raises((PermissionError, BaseExceptionGroup)) as exc_info:
            await runner_invalid.run(branch_invalid, graph_invalid)

        # If it's an ExceptionGroup, check the contained exception
        if isinstance(exc_info.value, BaseExceptionGroup):
            exceptions = list(exc_info.value.exceptions)
            assert len(exceptions) == 1
            assert isinstance(exceptions[0], PermissionError)
        else:
            assert isinstance(exc_info.value, PermissionError)

    @pytest.mark.anyio
    async def test_security_fail_closed(self):
        """Test: SecurityFailClosed (CRITICAL)

        GIVEN a node where the required_rights() function itself raises an exception
        WHEN Runner.run() attempts to execute the node
        THEN the Runner must NOT fall back to static rights.
        AND it must halt execution immediately and raise PermissionError (Failing Closed).
        """

        class FailingRightsMorphism(Morphism):
            def __init__(self):
                self.name = "FailingRightsMorphism"
                self.requires = set()

            def required_rights(self, **kwargs):
                # This function fails during execution
                raise RuntimeError("Rights calculation failed")

            async def pre(self, branch, **kwargs) -> bool:
                return True

            async def apply(self, branch, **kwargs) -> dict:
                return {"result": "should_not_reach_here"}

            async def post(self, branch, result, **kwargs) -> bool:
                return True

        branch = create_branch(id=uuid4(), capabilities={"some.capability"})
        node = OpNode(id=uuid4(), m=FailingRightsMorphism())
        graph = OpGraph(nodes={node.id: node}, roots={node.id})

        runner = create_test_runner()

        # Should fail closed with PermissionError, not proceed with static rights - may be wrapped in ExceptionGroup
        with pytest.raises((PermissionError, BaseExceptionGroup)) as exc_info:
            await runner.run(branch, graph)

        # If it's an ExceptionGroup, check the contained exception
        if isinstance(exc_info.value, BaseExceptionGroup):
            exceptions = list(exc_info.value.exceptions)
            assert len(exceptions) == 1
            assert isinstance(exceptions[0], PermissionError)
        else:
            assert isinstance(exc_info.value, PermissionError)


class TestRunnerValidationHardening:
    """TestSuite: RunnerValidationHardening - Pre/post condition validation."""

    @pytest.mark.anyio
    async def test_pre_condition_failure_halts_node(self):
        """Test: PreConditionFailureHaltsNode

        GIVEN a node where Morphism.pre() returns False
        WHEN Runner.run() attempts to execute it
        THEN it must raise RuntimeError (verify it is NOT an AssertionError, which might be compiled out).
        AND Morphism.apply() must not be called.
        """
        apply_called = []

        class TestPreFailMorphism(PreConditionFailMorphism):
            async def apply(self, branch, **kwargs) -> dict:
                apply_called.append("apply_called")
                return await super().apply(branch, **kwargs)

        branch = Branch(id=uuid4())
        node = OpNode(id=uuid4(), m=TestPreFailMorphism("test"))
        graph = OpGraph(nodes={node.id: node}, roots={node.id})

        runner = create_test_runner()

        # Should raise RuntimeError (not AssertionError) - may be wrapped in ExceptionGroup
        with pytest.raises((RuntimeError, BaseExceptionGroup)) as exc_info:
            await runner.run(branch, graph)

        # If it's an ExceptionGroup, check the contained exception
        if isinstance(exc_info.value, BaseExceptionGroup):
            exceptions = list(exc_info.value.exceptions)
            assert len(exceptions) == 1
            assert isinstance(exceptions[0], RuntimeError)
            assert "pre" in str(exceptions[0]) and "condition" in str(exceptions[0])
        else:
            assert "pre" in str(exc_info.value) and "condition" in str(exc_info.value)

        # Verify apply() was not called
        assert (
            len(apply_called) == 0
        ), "Morphism.apply() must not be called when pre-condition fails"

    @pytest.mark.anyio
    async def test_post_condition_failure_raises_error(self):
        """Test: PostConditionFailureRaisesError

        GIVEN a node where Morphism.post() returns False
        WHEN Runner.run() executes it
        THEN Morphism.apply() is called.
        AND it must raise RuntimeError after execution.
        """
        apply_called = []

        class TestPostFailMorphism(PostConditionFailMorphism):
            async def apply(self, branch, **kwargs) -> dict:
                apply_called.append("apply_called")
                return await super().apply(branch, **kwargs)

        branch = Branch(id=uuid4())
        node = OpNode(id=uuid4(), m=TestPostFailMorphism("test"))
        graph = OpGraph(nodes={node.id: node}, roots={node.id})

        runner = create_test_runner()

        # Should raise RuntimeError after apply() completes - may be wrapped in ExceptionGroup
        with pytest.raises((RuntimeError, BaseExceptionGroup)) as exc_info:
            await runner.run(branch, graph)

        # If it's an ExceptionGroup, check the contained exception
        if isinstance(exc_info.value, BaseExceptionGroup):
            exceptions = list(exc_info.value.exceptions)
            assert len(exceptions) == 1
            assert isinstance(exceptions[0], RuntimeError)
            assert "post" in str(exceptions[0]) and "condition" in str(exceptions[0])
        else:
            assert "post" in str(exc_info.value) and "condition" in str(exc_info.value)

        # Verify apply() was called (unlike pre-condition failure)
        assert len(apply_called) == 1, "Morphism.apply() must be called before post-condition check"
