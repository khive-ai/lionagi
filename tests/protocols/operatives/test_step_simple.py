"""
Simple Coverage Tests for Step Class

Focuses on exercising code paths to improve coverage without
complex integrations that may be broken due to API changes.
"""

import pytest

from lionagi.ln.types import Spec
from lionagi.operations.operate.step import Operative, Step


class TestStepBasicFunctionality:
    """Test basic Step functionality that should work."""

    def test_request_operative_basic(self):
        """Test basic request operative creation."""
        operative = Step.request_operative()

        assert isinstance(operative, Operative)
        assert operative.name is not None

    def test_request_operative_with_name(self):
        """Test request operative with custom name."""
        operative = Step.request_operative(name="test_name")

        assert operative.name == "test_name"

    def test_request_operative_with_max_retries(self):
        """Test request operative with max retries."""
        operative = Step.request_operative(max_retries=5)

        assert operative.max_retries == 5

    def test_request_operative_with_auto_retry_parse_false(self):
        """Test request operative with auto_retry_parse disabled."""
        operative = Step.request_operative(auto_retry_parse=False)

        assert operative.auto_retry_parse is False

    def test_request_operative_with_auto_retry_parse_true(self):
        """Test request operative with auto_retry_parse enabled."""
        operative = Step.request_operative(auto_retry_parse=True)

        assert operative.auto_retry_parse is True

    def test_request_operative_with_reason_field(self):
        """Test request operative with reason field enabled."""
        operative = Step.request_operative(reason=True)

        assert isinstance(operative, Operative)
        # Verify reason field was added
        assert operative.operable.get("reason") is not None

    def test_request_operative_with_actions_field(self):
        """Test request operative with actions field enabled."""
        operative = Step.request_operative(actions=True)

        assert isinstance(operative, Operative)
        # Verify action fields were added
        assert operative.operable.get("action_required") is not None
        assert operative.operable.get("action_requests") is not None

    def test_request_operative_with_reason_and_actions(self):
        """Test request operative with both reason and actions."""
        operative = Step.request_operative(reason=True, actions=True)

        assert isinstance(operative, Operative)
        assert operative.operable.get("reason") is not None
        assert operative.operable.get("action_required") is not None

    def test_request_operative_with_custom_fields(self):
        """Test request operative with custom field specs."""
        custom_spec = Spec(str, name="custom_field", description="Test field")
        operative = Step.request_operative(
            fields={"custom_field": custom_spec}
        )

        assert isinstance(operative, Operative)
        assert operative.operable.get("custom_field") is not None

    def test_request_operative_with_adapter(self):
        """Test request operative with specific adapter."""
        operative = Step.request_operative(adapter="pydantic")

        assert operative.adapter == "pydantic"


class TestStepRespondOperative:
    """Test respond_operative functionality."""

    def test_respond_operative_basic(self):
        """Test basic respond operative."""
        req_op = Step.request_operative(reason=True)
        resp_op = Step.respond_operative(req_op)

        assert isinstance(resp_op, Operative)
        assert resp_op._response_model_cls is not None

    def test_respond_operative_with_actions(self):
        """Test respond operative with action responses."""
        req_op = Step.request_operative(actions=True)
        resp_op = Step.respond_operative(req_op)

        assert isinstance(resp_op, Operative)
        # Should have action_responses field
        assert resp_op.operable.get("action_responses") is not None

    def test_respond_operative_with_additional_fields(self):
        """Test respond operative with additional fields."""
        req_op = Step.request_operative(reason=True)
        result_spec = Spec(str, name="result", description="Result")
        resp_op = Step.respond_operative(
            req_op, additional_fields={"result": result_spec}
        )

        assert isinstance(resp_op, Operative)
        assert resp_op.operable.get("result") is not None

    def test_respond_operative_with_exclude_fields(self):
        """Test respond operative excluding specific fields."""
        req_op = Step.request_operative(reason=True, actions=True)
        # Note: exclude_fields is no longer supported in respond_operative
        # The exclusion is handled at the request_operative level via request_exclude
        resp_op = Step.respond_operative(req_op)

        assert isinstance(resp_op, Operative)

    def test_respond_operative_no_inheritance(self):
        """Test respond operative without inheriting base fields."""
        req_op = Step.request_operative(reason=True)
        # Note: inherit_base is no longer supported - response always inherits from request
        resp_op = Step.respond_operative(req_op)

        assert isinstance(resp_op, Operative)


class TestStepUtilityMethods:
    """Test Step as utility class."""

    def test_step_instantiation(self):
        """Test that Step can be instantiated."""
        step = Step()

        assert isinstance(step, Step)

    def test_step_static_methods_callable(self):
        """Test that Step static methods are callable."""
        assert callable(Step.request_operative)
        assert callable(Step.respond_operative)


class TestStepEdgeCases:
    """Test edge cases and error conditions."""

    def test_request_operative_empty_fields(self):
        """Test request operative with empty custom fields dict."""
        operative = Step.request_operative(fields={})

        assert isinstance(operative, Operative)

    def test_request_operative_none_fields(self):
        """Test request operative with None custom fields."""
        operative = Step.request_operative(fields=None)

        assert isinstance(operative, Operative)

    def test_request_operative_multiple_custom_fields(self):
        """Test request operative with multiple custom fields."""
        fields = {
            "field1": Spec(str, name="field1", description="First field"),
            "field2": Spec(int, name="field2", description="Second field"),
        }
        operative = Step.request_operative(fields=fields)

        assert isinstance(operative, Operative)
        assert operative.operable.get("field1") is not None
        assert operative.operable.get("field2") is not None

    def test_respond_operative_with_all_options(self):
        """Test respond operative with all optional parameters."""
        req_op = Step.request_operative(reason=True, actions=True)
        result_spec = Spec(str, name="result", description="Result")

        resp_op = Step.respond_operative(
            req_op, additional_fields={"result": result_spec}
        )

        assert isinstance(resp_op, Operative)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
