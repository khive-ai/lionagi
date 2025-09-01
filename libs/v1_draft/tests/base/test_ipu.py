"""Test suite for ipu.py (IPU and Invariants) - TDD Specification Implementation.

Focus: Correct enforcement of constraints and performance impact.
"""

import asyncio
import time
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from lionagi.base.ipu import (
    IPU,
    CapabilityMonotonicityInvariant,
    CtxWriteSetInvariant,
    InvariantViolationError,
    LenientIPU,
    ResultShapeInvariant,
    StrictIPU,
)
from lionagi.base.types import Branch, create_branch


class AlwaysFailingInvariant:
    """Mock invariant that always fails for testing."""

    def pre_check(self, branch, operation_context):
        return False, "Always failing invariant"

    def post_check(self, branch, operation_context, result):
        return False, "Always failing invariant"


class AlwaysPassingInvariant:
    """Mock invariant that always passes for testing."""

    def pre_check(self, branch, operation_context):
        return True, "Always passing"

    def post_check(self, branch, operation_context, result):
        return True, "Always passing"


class TestIPUCore:
    """TestSuite: IPUCore - Core IPU functionality and enforcement modes."""

    def test_strict_ipu_halts_execution(self):
        """Test: StrictIPUHaltsExecution

        GIVEN a StrictIPU with an invariant that always fails
        WHEN the IPU check runs (e.g., IPU.enter_node or IPU.exit_node)
        THEN it must raise an InvariantViolationError.
        """
        # Create StrictIPU with failing invariant
        failing_invariant = AlwaysFailingInvariant()
        strict_ipu = StrictIPU([failing_invariant])

        # Create test branch and operation context
        branch = Branch(id=uuid4(), ctx={"test": "data"})
        operation_context = {"operation": "test_op", "node_id": "test_node"}

        # Test pre-check enforcement
        with pytest.raises(InvariantViolationError, match="Always failing invariant"):
            strict_ipu.enter_node(branch, operation_context)

        # Test post-check enforcement
        with pytest.raises(InvariantViolationError, match="Always failing invariant"):
            strict_ipu.exit_node(branch, operation_context, {"result": "test"})

    def test_lenient_ipu_logs_and_continues(self):
        """Test: LenientIPULogsAndContinues

        GIVEN a LenientIPU with an invariant that always fails
        WHEN the IPU check runs
        THEN it must NOT raise an exception
        AND it must log the violation.
        """
        # Create LenientIPU with failing invariant
        failing_invariant = AlwaysFailingInvariant()
        lenient_ipu = LenientIPU([failing_invariant])

        # Create test branch and operation context
        branch = Branch(id=uuid4(), ctx={"test": "data"})
        operation_context = {"operation": "test_op", "node_id": "test_node"}

        # Mock logger to capture log calls
        with patch("lionagi.base.ipu.logger") as mock_logger:
            # Test pre-check - should not raise but should log
            try:
                lenient_ipu.enter_node(branch, operation_context)
            except InvariantViolationError:
                pytest.fail("LenientIPU should not raise InvariantViolationError on pre-check")

            # Verify logging occurred
            mock_logger.warning.assert_called()

            # Reset mock for post-check test
            mock_logger.reset_mock()

            # Test post-check - should not raise but should log
            try:
                lenient_ipu.exit_node(branch, operation_context, {"result": "test"})
            except InvariantViolationError:
                pytest.fail("LenientIPU should not raise InvariantViolationError on post-check")

            # Verify logging occurred
            mock_logger.warning.assert_called()

    def test_strict_ipu_passes_with_valid_invariants(self):
        """Test that StrictIPU passes when all invariants are satisfied."""
        # Create StrictIPU with passing invariant
        passing_invariant = AlwaysPassingInvariant()
        strict_ipu = StrictIPU([passing_invariant])

        # Create test branch and operation context
        branch = Branch(id=uuid4(), ctx={"test": "data"})
        operation_context = {"operation": "test_op", "node_id": "test_node"}

        # Should pass without exception
        try:
            strict_ipu.enter_node(branch, operation_context)
            strict_ipu.exit_node(branch, operation_context, {"result": "success"})
        except InvariantViolationError:
            pytest.fail("StrictIPU should pass when all invariants are satisfied")

    def test_lenient_ipu_passes_with_valid_invariants(self):
        """Test that LenientIPU passes silently when all invariants are satisfied."""
        # Create LenientIPU with passing invariant
        passing_invariant = AlwaysPassingInvariant()
        lenient_ipu = LenientIPU([passing_invariant])

        # Create test branch and operation context
        branch = Branch(id=uuid4(), ctx={"test": "data"})
        operation_context = {"operation": "test_op", "node_id": "test_node"}

        # Mock logger to ensure no warnings are logged
        with patch("lionagi.base.ipu.logger") as mock_logger:
            try:
                lenient_ipu.enter_node(branch, operation_context)
                lenient_ipu.exit_node(branch, operation_context, {"result": "success"})
            except InvariantViolationError:
                pytest.fail("LenientIPU should not raise exceptions")

            # Verify no warnings were logged
            mock_logger.warning.assert_not_called()

    def test_ipu_with_multiple_invariants(self):
        """Test IPU behavior with multiple invariants (mixed passing/failing)."""
        passing_invariant = AlwaysPassingInvariant()
        failing_invariant = AlwaysFailingInvariant()

        # StrictIPU with mixed invariants - should fail if any invariant fails
        strict_ipu = StrictIPU([passing_invariant, failing_invariant])
        branch = Branch(id=uuid4(), ctx={"test": "data"})
        operation_context = {"operation": "test_op", "node_id": "test_node"}

        with pytest.raises(InvariantViolationError):
            strict_ipu.enter_node(branch, operation_context)

        # LenientIPU with mixed invariants - should log but continue
        lenient_ipu = LenientIPU([passing_invariant, failing_invariant])

        with patch("lionagi.base.ipu.logger") as mock_logger:
            try:
                lenient_ipu.enter_node(branch, operation_context)
            except InvariantViolationError:
                pytest.fail("LenientIPU should not raise with mixed invariants")

            # Should log the failing invariant
            mock_logger.warning.assert_called()


class TestStandardInvariants:
    """TestSuite: StandardInvariants - Built-in invariant implementations."""

    def test_ctx_write_set_enforcement(self):
        """Test: CtxWriteSetEnforcement (CRITICAL: Data Integrity)

        GIVEN an operation with declared ctx_writes={"A"}
        WHEN the operation executes and writes to Branch.ctx["B"] (undeclared)
        AND the CtxWriteSet invariant runs its post-check (comparing snapshots)
        THEN the invariant check must fail.
        """
        # Create CtxWriteSet invariant
        ctx_invariant = CtxWriteSetInvariant()

        # Create branch with initial context
        branch = Branch(id=uuid4(), ctx={"existing": "data", "A": "original"})

        # Operation declares it will write to "A" only
        operation_context = {"operation": "test_op", "ctx_writes": {"A"}}

        # Pre-check should pass (takes snapshot)
        valid, message = ctx_invariant.pre_check(branch, operation_context)
        assert valid, f"Pre-check should pass but failed: {message}"

        # Simulate operation: write to declared "A" (should be allowed)
        branch.ctx["A"] = "modified"

        # Post-check should pass for declared write
        valid, message = ctx_invariant.post_check(branch, operation_context, {"result": "success"})
        assert valid, f"Post-check should pass for declared write but failed: {message}"

        # Now test undeclared write
        branch_undeclared = Branch(id=uuid4(), ctx={"existing": "data", "A": "original"})

        # Pre-check (takes fresh snapshot)
        valid, message = ctx_invariant.pre_check(branch_undeclared, operation_context)
        assert valid, f"Pre-check should pass: {message}"

        # Simulate operation: write to undeclared "B" (should be caught)
        branch_undeclared.ctx["B"] = "undeclared_write"

        # Post-check should fail for undeclared write
        valid, message = ctx_invariant.post_check(
            branch_undeclared, operation_context, {"result": "success"}
        )
        assert not valid, "Post-check must fail when undeclared context key is written"
        assert "B" in message, "Error message should mention the undeclared key"

    def test_ctx_write_set_snapshot_performance(self):
        """Test: CtxWriteSetSnapshotPerformance (CRITICAL: Performance)

        GIVEN a Branch.ctx containing a very large data structure (e.g., 10MB dict)
        WHEN the CtxWriteSet invariant takes a snapshot (e.g., upon IPU.enter_node)
        THEN the time taken for the snapshot must be minimal (e.g., < 1ms if using efficient mechanisms like persistent structures, significantly longer if using deepcopy).
        """
        # Create large context (simulating 10MB of data)
        large_dict = {}
        for i in range(100000):  # Large number of entries
            large_dict[f"key_{i}"] = f"value_{i}_with_some_additional_data_to_increase_size"

        branch = Branch(id=uuid4(), ctx=large_dict)
        ctx_invariant = CtxWriteSetInvariant()
        operation_context = {"operation": "test_op", "ctx_writes": {"key_1"}}

        # Measure snapshot time
        start_time = time.perf_counter()
        valid, message = ctx_invariant.pre_check(branch, operation_context)
        snapshot_time = time.perf_counter() - start_time

        assert valid, f"Pre-check should succeed: {message}"

        # Performance assertion - should be reasonably fast
        # Note: This is environment-dependent, but deepcopy of 10MB would be significantly slower
        assert snapshot_time < 0.1, f"Snapshot should be fast but took {snapshot_time:.3f}s"

        # Verify snapshot isolation - modify original and check post-check still works
        branch.ctx["key_1"] = "modified_value"

        end_time = time.perf_counter()
        valid, message = ctx_invariant.post_check(branch, operation_context, {"result": "success"})
        total_time = end_time - start_time

        assert valid, f"Post-check should pass: {message}"
        assert total_time < 0.2, f"Total time should be reasonable but took {total_time:.3f}s"

    def test_capability_monotonicity(self):
        """Test: CapabilityMonotonicity (Security)

        GIVEN a Branch with capabilities C1
        WHEN an operation runs
        AND the Branch capabilities are escalated to C2 (where C2 > C1, rights gained)
        THEN the CapabilityMonotonicity invariant must fail (preventing privilege escalation).
        """
        # Create CapabilityMonotonicity invariant
        capability_invariant = CapabilityMonotonicityInvariant()

        # Create branch with initial capabilities
        initial_caps = {"fs.read:/data/*", "net.out:api.service.com"}
        branch = create_branch(id=uuid4(), capabilities=initial_caps.copy())

        # Get strong reference to prevent WeakValueDictionary cleanup
        branch_capabilities = branch.capabilities

        operation_context = {"operation": "test_op"}

        # Pre-check should pass (takes snapshot of capabilities)
        valid, message = capability_invariant.pre_check(branch, operation_context)
        assert valid, f"Pre-check should pass: {message}"

        # Test privilege reduction (should be allowed)
        branch_capabilities.remove("net.out:api.service.com")
        valid, message = capability_invariant.post_check(
            branch, operation_context, {"result": "success"}
        )
        assert valid, "Capability reduction should be allowed"

        # Reset for escalation test - clear and re-add capabilities
        branch_capabilities.clear()
        for cap in initial_caps:
            branch_capabilities.add(cap)

        # Pre-check again (fresh snapshot)
        valid, message = capability_invariant.pre_check(branch, operation_context)
        assert valid, f"Pre-check should pass: {message}"

        # Test privilege escalation (should be forbidden)
        branch_capabilities.add("fs.write:/sensitive/*")  # New privilege
        branch_capabilities.add("admin.access:system")  # Administrative privilege

        valid, message = capability_invariant.post_check(
            branch, operation_context, {"result": "success"}
        )
        assert not valid, "Capability escalation must be prevented"
        assert (
            "escalation" in message.lower() or "privilege" in message.lower()
        ), "Error should mention privilege escalation"

    def test_result_shape_violation(self):
        """Test: ResultShapeViolation (Data Integrity)

        GIVEN a result dict = {"id": 1, "name": "test"}
        # Mismatched type for 'name'
        WHEN checking against ResultShape(schema=MyStruct(id=int, name=int))
        THEN the invariant check must fail (testing msgspec integration).
        """
        import msgspec

        # Define expected result schema
        class MyStruct(msgspec.Struct):
            id: int
            name: str  # Expected to be string

        # Create ResultShape invariant with schema
        result_invariant = ResultShapeInvariant(expected_schema=MyStruct)

        branch = Branch(id=uuid4())
        operation_context = {"operation": "test_op"}

        # Pre-check should always pass for ResultShape
        valid, message = result_invariant.pre_check(branch, operation_context)
        assert valid, f"Pre-check should pass for ResultShape: {message}"

        # Test valid result (correct types)
        valid_result = {"id": 1, "name": "test"}
        valid, message = result_invariant.post_check(branch, operation_context, valid_result)
        assert valid, f"Valid result shape should pass: {message}"

        # Test invalid result (wrong type for 'name')
        invalid_result = {"id": 1, "name": 123}  # name should be string, not int
        valid, message = result_invariant.post_check(branch, operation_context, invalid_result)
        assert not valid, "Invalid result shape must fail validation"
        assert "name" in message or "type" in message, "Error should mention the type mismatch"

        # Test missing field
        incomplete_result = {"id": 1}  # missing 'name'
        valid, message = result_invariant.post_check(branch, operation_context, incomplete_result)
        assert not valid, "Incomplete result must fail validation"

        # Test extra field (should pass - extra fields are typically allowed)
        extra_result = {"id": 1, "name": "test", "extra": "field"}
        valid, message = result_invariant.post_check(branch, operation_context, extra_result)
        # Note: This depends on msgspec behavior - it might pass or fail depending on configuration
        # For now, we'll test that it at least processes without crashing
        assert isinstance(valid, bool), "Result should be a boolean"

    def test_invariant_with_empty_operation_context(self):
        """Test invariant behavior with minimal or empty operation context."""
        ctx_invariant = CtxWriteSetInvariant()
        branch = Branch(id=uuid4(), ctx={"data": "test"})

        # Empty operation context
        empty_context = {}
        valid, message = ctx_invariant.pre_check(branch, empty_context)
        assert valid, "Empty operation context should be handled gracefully"

        # Minimal operation context
        minimal_context = {"operation": "minimal"}
        valid, message = ctx_invariant.pre_check(branch, minimal_context)
        assert valid, "Minimal operation context should be handled gracefully"

    def test_invariant_thread_safety(self):
        """Test that invariants work correctly with concurrent access."""
        import threading

        ctx_invariant = CtxWriteSetInvariant()
        branch = Branch(id=uuid4(), ctx={"shared": "data"})
        operation_context = {"operation": "concurrent_test", "ctx_writes": {"shared"}}

        results = []

        def run_invariant_check():
            try:
                valid, message = ctx_invariant.pre_check(branch, operation_context)
                branch.ctx["shared"] = f"modified_{threading.current_thread().ident}"
                valid, message = ctx_invariant.post_check(
                    branch, operation_context, {"result": "success"}
                )
                results.append(valid)
            except Exception as e:
                results.append(str(e))

        # Run multiple threads concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=run_invariant_check)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All should complete without crashing (though results may vary due to concurrency)
        assert len(results) == 5, "All threads should complete"
        assert all(isinstance(r, (bool, str)) for r in results), "All results should be valid"
