"""
Simplified tests for ReAct action functionality.

Focus on testing the critical regression: that action fields are properly
added when using ReAct.
"""

import pytest

from lionagi.operations.operate.operate import prepare_operate_kw
from lionagi.operations.operate.step import Step
from lionagi.operations.ReAct.utils import PlannedAction, ReActAnalysis
from lionagi.session.branch import Branch


def mock_tool(x: int) -> int:
    """Simple mock tool."""
    return x * 2


class TestReActActionFieldGeneration:
    """Test that ReAct properly generates action fields."""

    def test_react_analysis_with_action_fields(self):
        """Test that ReActAnalysis can be extended with action fields."""
        # Create operative with ReActAnalysis as base and action fields
        operative = Step.request_operative(
            base_type=ReActAnalysis, actions=True
        )

        # Verify action fields were added to operable
        assert operative.operable.get("action_required") is not None
        assert operative.operable.get("action_requests") is not None
        assert operative.operable.get("action_responses") is not None

        # Create request model
        request_model = operative.create_request_model()
        fields = request_model.model_fields

        # Should have ReActAnalysis fields
        assert "analysis" in fields
        assert "planned_actions" in fields
        assert "extension_needed" in fields

        # Should have action fields (except action_responses in request)
        assert "action_required" in fields
        assert "action_requests" in fields

        # Create response model
        operative = Step.respond_operative(operative)
        response_model = operative.create_response_model()
        response_fields = response_model.model_fields

        # Response should have all fields including action_responses
        assert "action_responses" in response_fields

    def test_react_prepare_kwargs_with_tools(self):
        """Test that prepare_operate_kw creates proper operative for ReAct-like usage."""
        branch = Branch(tools=[mock_tool])

        # Simulate ReAct-like kwargs preparation
        kwargs = prepare_operate_kw(
            branch,
            instruction="Test with tool",
            response_format=ReActAnalysis,
            actions=True,
        )

        # Should create operative
        assert kwargs["operative"] is not None
        operative = kwargs["operative"]

        # Should have both ReActAnalysis base and action fields
        assert operative.operable.get("action_required") is not None
        assert operative.operable.get("action_requests") is not None

    def test_planned_action_structure(self):
        """Test that PlannedAction has correct structure."""
        # Create a valid PlannedAction
        action = PlannedAction(
            action_type="search",
            description="Search for information about lions",
        )

        assert action.action_type == "search"
        assert action.description == "Search for information about lions"

    def test_react_analysis_instance_creation(self):
        """Test creating ReActAnalysis instance with action fields."""
        # Create operative with action fields
        operative = Step.request_operative(
            base_type=ReActAnalysis, actions=True
        )
        operative = Step.respond_operative(operative)

        # Create instance
        response_type = operative.response_type
        instance = response_type(
            analysis="Test analysis",
            planned_actions=[
                PlannedAction(
                    action_type="calculate", description="Calculate 2+2"
                )
            ],
            extension_needed=False,
            action_required=True,
            action_requests=[{"function": "mock_tool", "arguments": {"x": 2}}],
            action_responses=[],  # Empty list is valid
        )

        # Verify structure
        assert instance.analysis == "Test analysis"
        assert len(instance.planned_actions) == 1
        assert instance.action_required is True
        assert len(instance.action_requests) == 1
        assert instance.action_responses == []


class TestReActOperativeInheritance:
    """Test that ReAct operatives properly inherit."""

    def test_response_inherits_from_request(self):
        """Test that response model inherits from request model."""
        # Create request operative
        req_op = Step.request_operative(base_type=ReActAnalysis, actions=True)

        # Create response operative
        resp_op = Step.respond_operative(req_op)

        # Both should exist
        assert resp_op.request_type is not None
        assert resp_op.response_type is not None

        # Response should inherit from request
        assert issubclass(resp_op.response_type, resp_op.request_type)

    def test_action_responses_only_in_response(self):
        """Test that action_responses is only in response model."""
        operative = Step.request_operative(
            base_type=ReActAnalysis, actions=True
        )

        # Request model should NOT have action_responses
        request_model = operative.create_request_model()
        assert "action_responses" not in request_model.model_fields

        # Response model SHOULD have action_responses
        operative = Step.respond_operative(operative)
        response_model = operative.create_response_model()
        assert "action_responses" in response_model.model_fields


class TestReActRegression:
    """Regression tests for the specific ReAct issue."""

    def test_regression_react_with_actions_creates_proper_fields(self):
        """
        Regression test for the critical issue where ReAct with actions
        didn't create proper action fields.
        """
        branch = Branch(tools=[mock_tool])

        # This is what ReAct does internally
        kwargs = prepare_operate_kw(
            branch,
            instruct={"instruction": "Calculate something", "context": {}},
            response_format=ReActAnalysis,
            actions=True,
            reason=False,  # ReActAnalysis already has analysis field
        )

        # Must create operative with action fields
        assert kwargs["operative"] is not None
        operative = kwargs["operative"]

        # Must have response type
        assert operative.response_type is not None

        # Response type must have action fields
        response_fields = operative.response_type.model_fields
        assert "action_required" in response_fields
        assert "action_requests" in response_fields
        assert "action_responses" in response_fields

        # Must also have ReActAnalysis fields
        assert "analysis" in response_fields
        assert "planned_actions" in response_fields
        assert "extension_needed" in response_fields

    def test_regression_notebook_example(self):
        """
        Test that simulates the notebook example that was failing.
        operate() with actions=True should return object with action_responses.
        """
        branch = Branch(tools=[mock_tool])

        kwargs = prepare_operate_kw(
            branch, instruction="Use the tool to double 5", actions=True
        )

        operative = kwargs["operative"]
        assert operative is not None

        # The response type should be called ...Response, not ...Request
        response_type_name = operative.response_type.__name__
        assert "Response" in response_type_name

        # Should have action_responses field
        assert "action_responses" in operative.response_type.model_fields


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
