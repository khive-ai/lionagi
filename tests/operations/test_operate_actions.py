"""
Comprehensive tests for operate() action functionality.

These tests ensure that action fields are properly injected and populated
when using operate() with actions=True or with action_param.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from lionagi.operations.act.act import _get_default_call_params
from lionagi.operations.fields import Instruct
from lionagi.operations.operate.operate import operate, prepare_operate_kw
from lionagi.operations.operate.step import Step
from lionagi.operations.types import ActionParam, ChatParam, ParseParam
from lionagi.session.branch import Branch


# Mock tools for testing
def mock_add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def mock_multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def mock_greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


class SimpleRequest(BaseModel):
    """Simple request model for testing."""

    query: str
    count: int = 1


class TestOperateActionFields:
    """Test that operate() properly adds action fields when needed."""

    @pytest.mark.asyncio
    async def test_operate_with_actions_true_adds_fields(self):
        """Test that actions=True adds action fields to response."""
        branch = Branch(tools=[mock_add, mock_multiply])

        # Mock the communicate to return a response with action fields
        test_response = MagicMock()
        test_response.action_required = True
        test_response.action_requests = [
            {"function": "mock_add", "arguments": {"a": 1, "b": 2}}
        ]
        test_response.action_responses = None

        # No need to patch communicate for this test - just testing prepare_operate_kw
        # Prepare kwargs using prepare_operate_kw
        kwargs = prepare_operate_kw(
            branch,
            instruction="Test instruction",
            actions=True,  # This should trigger action field injection
        )

        # Verify operative was created with action fields
        assert (
            kwargs.get("operative") is not None
        ), "Operative should be created when actions=True"
        operative = kwargs["operative"]

        # Check that operative has action fields in its operable
        assert (
            operative.operable.get("action_required") is not None
        ), "action_required field missing"
        assert (
            operative.operable.get("action_requests") is not None
        ), "action_requests field missing"
        assert (
            operative.operable.get("action_responses") is not None
        ), "action_responses field missing"

        # Check response type includes all action fields
        assert (
            operative.response_type is not None
        ), "Response type should be created"
        response_fields = operative.response_type.model_fields
        assert (
            "action_required" in response_fields
        ), "Response should have action_required"
        assert (
            "action_requests" in response_fields
        ), "Response should have action_requests"
        assert (
            "action_responses" in response_fields
        ), "Response should have action_responses"

    @pytest.mark.asyncio
    async def test_operate_with_action_param_adds_fields(self):
        """Test that providing action_param adds action fields."""
        branch = Branch(tools=[mock_greet])

        # Create action_param explicitly
        action_param = ActionParam(
            action_call_params=_get_default_call_params(),
            tools=True,
            strategy="concurrent",
        )

        # Mock communicate using patch
        test_response = MagicMock()
        test_response.action_required = True
        test_response.action_requests = [
            {"function": "mock_greet", "arguments": {"name": "Alice"}}
        ]
        test_response.action_responses = []

        with patch(
            "lionagi.operations.communicate.communicate.communicate",
            new_callable=AsyncMock,
        ) as mock_comm:
            mock_comm.return_value = test_response

            # Call operate directly with action_param
            result = await operate(
                branch,
                instruction="Greet Alice",
                chat_param=ChatParam(),
                action_param=action_param,
                invoke_actions=False,  # Don't actually invoke to simplify test
            )

            # Verify result has action fields
            assert hasattr(
                result, "action_required"
            ), "Result should have action_required"
            assert hasattr(
                result, "action_requests"
            ), "Result should have action_requests"

    @pytest.mark.asyncio
    async def test_operate_without_actions_no_fields(self):
        """Test that without actions=True, no action fields are added."""
        branch = Branch()  # No tools

        # Prepare kwargs without actions
        kwargs = prepare_operate_kw(
            branch,
            instruction="Test instruction",
            actions=False,  # No actions
            response_format=SimpleRequest,
        )

        # Operative should still be created if response_format provided
        operative = kwargs.get("operative")
        if operative:
            # But it should NOT have action fields
            assert (
                operative.operable.get("action_required") is None
            ), "Should not have action_required"
            assert (
                operative.operable.get("action_requests") is None
            ), "Should not have action_requests"
            assert (
                operative.operable.get("action_responses") is None
            ), "Should not have action_responses"

    @pytest.mark.asyncio
    async def test_operate_with_base_model_and_actions(self):
        """Test that actions work with a base model."""
        branch = Branch(tools=[mock_add])

        # Prepare with both response_format and actions
        kwargs = prepare_operate_kw(
            branch,
            instruction="Add 5 and 3",
            response_format=SimpleRequest,
            actions=True,
        )

        operative = kwargs.get("operative")
        assert operative is not None, "Operative should be created"

        # Check base model fields are preserved
        response_type = operative.response_type
        assert response_type is not None
        response_fields = response_type.model_fields

        # Should have both base model fields and action fields
        assert (
            "query" in response_fields
        ), "Base model field 'query' should be present"
        assert (
            "count" in response_fields
        ), "Base model field 'count' should be present"
        assert (
            "action_required" in response_fields
        ), "Action field should be added"
        assert (
            "action_requests" in response_fields
        ), "Action field should be added"
        assert (
            "action_responses" in response_fields
        ), "Action field should be added"

    def test_operate_action_responses_field_exists(self):
        """Test that action_responses field exists in response model."""
        branch = Branch(tools=[mock_multiply])

        # Prepare kwargs with actions
        kwargs = prepare_operate_kw(
            branch,
            instruction="Multiply 3 by 5",
            actions=True,
            invoke_actions=True,
        )

        # The operative should have response_type with action_responses
        operative = kwargs["operative"]
        assert operative.response_type is not None

        # Verify the response would have action_responses field
        response_fields = operative.response_type.model_fields
        assert "action_responses" in response_fields


class TestOperateRegressions:
    """Regression tests for specific issues that were fixed."""

    def test_regression_operate_returns_correct_type(self):
        """
        Regression test for issue where operate() returned OperativeRequest
        instead of OperativeResponse when actions=True.
        """
        branch = Branch(tools=[mock_add])

        # This was the failing case - operate with actions=True
        kwargs = prepare_operate_kw(branch, instruction="Test", actions=True)

        # Verify operative was created and passed
        assert "operative" in kwargs, "Operative should be in kwargs"
        assert kwargs["operative"] is not None, "Operative should not be None"

        # Verify response type name (should be Response, not Request)
        operative = kwargs["operative"]
        if operative.response_type:
            type_name = operative.response_type.__name__
            assert (
                "Response" in type_name
            ), f"Type should be Response-like, got {type_name}"
            assert (
                "Request" not in type_name or "Response" in type_name
            ), f"Should be Response type, not Request type, got {type_name}"

    @pytest.mark.asyncio
    async def test_regression_react_action_fields(self):
        """
        Regression test to ensure ReAct creates operatives with action fields.
        """
        from lionagi.operations.ReAct.utils import ReActAnalysis

        # ReAct always uses ReActAnalysis as base
        operative = Step.request_operative(
            base_type=ReActAnalysis,
            reason=False,  # ReActAnalysis already has reason
            actions=True,  # ReAct needs action fields
        )

        # Verify action fields are added
        assert operative.operable.get("action_required") is not None
        assert operative.operable.get("action_requests") is not None

        # Create response model
        operative = Step.respond_operative(operative)

        # Verify response has all fields including action_responses
        response_type = operative.response_type
        assert response_type is not None
        fields = response_type.model_fields

        # Should have ReActAnalysis fields plus action fields
        assert "analysis" in fields, "Should have ReActAnalysis field"
        assert "planned_actions" in fields, "Should have ReActAnalysis field"
        assert "action_required" in fields, "Should have action field"
        assert "action_requests" in fields, "Should have action field"
        assert (
            "action_responses" in fields
        ), "Should have action response field"

    @pytest.mark.asyncio
    async def test_regression_instruct_with_actions(self):
        """Test that Instruct properly propagates actions flag."""
        instruct = Instruct(instruction="Test instruction", actions=True)

        assert (
            instruct.actions is True
        ), "Instruct should preserve actions flag"

        # Test with prepare_operate_kw
        branch = Branch(tools=[mock_add])
        kwargs = prepare_operate_kw(
            branch, instruct=instruct  # Pass Instruct object
        )

        # Should create operative with action fields
        operative = kwargs.get("operative")
        assert (
            operative is not None
        ), "Operative should be created from Instruct.actions"
        assert operative.operable.get("action_requests") is not None


class TestStepOperative:
    """Test Step.request_operative and Step.respond_operative."""

    def test_request_operative_with_actions(self):
        """Test that request_operative properly adds action fields."""
        operative = Step.request_operative(name="TestOp", actions=True)

        # Check operable has action fields
        assert operative.operable.get("action_required") is not None
        assert operative.operable.get("action_requests") is not None
        assert operative.operable.get("action_responses") is not None

        # Check request excludes action_responses
        assert "action_responses" in operative.request_exclude

    def test_respond_operative_inherits_fields(self):
        """Test that respond_operative creates proper inheritance."""
        # Create request operative with actions
        req_op = Step.request_operative(name="TestOp", actions=True)

        # Create response operative
        resp_op = Step.respond_operative(req_op)

        # Response type should inherit from request type
        assert resp_op.response_type is not None
        assert resp_op.request_type is not None

        # Response should have all fields including action_responses
        response_fields = resp_op.response_type.model_fields
        request_fields = resp_op.request_type.model_fields

        # Request should NOT have action_responses
        assert "action_responses" not in request_fields

        # Response SHOULD have action_responses
        assert "action_responses" in response_fields

    def test_request_operative_with_base_type(self):
        """Test that base_type is properly inherited."""

        class MyBase(BaseModel):
            base_field: str = "default"
            base_number: int = 42

        operative = Step.request_operative(base_type=MyBase, actions=True)

        # Create models
        operative.create_request_model()
        request_type = operative.request_type

        # Should have both base fields and action fields
        fields = request_type.model_fields
        assert "base_field" in fields
        assert "base_number" in fields
        assert "action_required" in fields
        assert "action_requests" in fields
        # But NOT action_responses in request
        assert "action_responses" not in fields


class TestPrepareOperateKw:
    """Test the prepare_operate_kw function."""

    def test_prepare_with_actions_creates_operative(self):
        """Test that actions=True triggers operative creation."""
        branch = Branch(tools=[mock_add])

        kwargs = prepare_operate_kw(branch, instruction="Test", actions=True)

        assert "operative" in kwargs
        assert kwargs["operative"] is not None

        # Check action_param is created
        assert kwargs["action_param"] is not None
        assert kwargs["invoke_actions"] is True

    def test_prepare_with_reason_creates_operative(self):
        """Test that reason=True triggers operative creation."""
        branch = Branch()

        kwargs = prepare_operate_kw(branch, instruction="Test", reason=True)

        assert "operative" in kwargs
        operative = kwargs["operative"]
        if operative:
            assert operative.operable.get("reason") is not None

    def test_prepare_without_special_params_no_operative(self):
        """Test that without special params, operative might not be created."""
        branch = Branch()

        kwargs = prepare_operate_kw(
            branch, instruction="Test", actions=False, reason=False
        )

        # Operative should be None or minimal
        operative = kwargs.get("operative")
        if operative:
            # Should not have action or reason fields
            assert operative.operable.get("action_required") is None
            assert operative.operable.get("reason") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
