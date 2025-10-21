# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for operate() and ReAct with Spec field models."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from lionagi.ln.types import Spec
from lionagi.models import FieldModel
from lionagi.operations.operate.operate import operate, prepare_operate_kw
from lionagi.session import Branch


@pytest.fixture
def mock_branch():
    """Create a mock branch with necessary attributes."""
    branch = MagicMock(spec=Branch)
    branch.chat_model = MagicMock()
    branch.parse_model = MagicMock()
    branch.acts = MagicMock()
    branch.acts.get_tool_schema = MagicMock(return_value=[])
    branch.user = "test_user"
    branch.id = "branch_123"
    return branch


class TestOperateWithSpec:
    """Test operate() function with Spec field models."""

    def test_prepare_operate_kw_with_fieldmodel(self, mock_branch):
        """Test prepare_operate_kw with FieldModel list."""
        fm1 = FieldModel(name="field1", annotation=str, default="test")
        fm2 = FieldModel(name="field2", annotation=int, default=10)

        result = prepare_operate_kw(
            mock_branch,
            instruction="Test instruction",
            field_models=[fm1, fm2],
        )

        assert "instruction" in result
        assert "chat_param" in result
        assert result["instruction"] == "Test instruction"

    def test_prepare_operate_kw_with_spec(self, mock_branch):
        """Test prepare_operate_kw with Spec list."""
        spec1 = Spec(str, name="spec1", default="spec_test")
        spec2 = Spec(bool, name="spec2", default=True)

        result = prepare_operate_kw(
            mock_branch,
            instruction="Test with spec",
            field_models=[spec1, spec2],
        )

        assert "instruction" in result
        assert "chat_param" in result
        assert result["instruction"] == "Test with spec"

    def test_prepare_operate_kw_with_mixed_types(self, mock_branch):
        """Test prepare_operate_kw with both FieldModel and Spec."""
        fm = FieldModel(name="fm_field", annotation=str, default="fm")
        spec = Spec(int, name="spec_field", default=42)

        result = prepare_operate_kw(
            mock_branch,
            instruction="Mixed types test",
            field_models=[fm, spec],
        )

        assert "instruction" in result
        assert "chat_param" in result
        assert result["instruction"] == "Mixed types test"

    def test_prepare_operate_kw_with_single_spec(self, mock_branch):
        """Test prepare_operate_kw with single Spec (not in list)."""
        spec = Spec(float, name="single", default=3.14)

        result = prepare_operate_kw(
            mock_branch,
            instruction="Single spec test",
            field_models=spec,  # Single instance
        )

        assert "instruction" in result
        assert "chat_param" in result
        assert result["instruction"] == "Single spec test"

    @pytest.mark.asyncio
    async def test_operate_with_spec_fields(self, mock_branch):
        """Test operate() with Spec field models."""
        spec1 = Spec(str, name="response_field", default="")
        spec2 = Spec(int, name="count", default=0)

        # Mock the communicate function
        with patch(
            "lionagi.operations.communicate.communicate.communicate"
        ) as mock_comm:
            mock_comm.return_value = {
                "response_field": "test response",
                "count": 5,
            }

            from lionagi.operations.operate.operate import (
                ChatParam,
                ParseParam,
            )

            result = await operate(
                mock_branch,
                instruction="Test with specs",
                chat_param=ChatParam(
                    response_format=None, imodel=mock_branch.chat_model
                ),
                parse_param=ParseParam(
                    response_format=None,
                    handle_validation="return_value",
                    imodel=mock_branch.parse_model,
                ),
                field_models=[spec1, spec2],
            )

            assert result == {"response_field": "test response", "count": 5}

    @pytest.mark.asyncio
    async def test_operate_with_base_model_and_spec(self, mock_branch):
        """Test operate() with base model and additional Spec fields."""

        class BaseResponse(BaseModel):
            base_field: str = "base"

        spec = Spec(int, name="extra_field", default=0)

        with patch(
            "lionagi.operations.communicate.communicate.communicate"
        ) as mock_comm:
            # Mock a response that matches the expected model
            mock_response = MagicMock(spec=BaseModel)
            mock_response.base_field = "modified"
            mock_response.extra_field = 100
            mock_comm.return_value = mock_response

            from lionagi.operations.operate.operate import (
                ChatParam,
                ParseParam,
            )

            result = await operate(
                mock_branch,
                instruction="Test base model with spec",
                chat_param=ChatParam(
                    response_format=BaseResponse, imodel=mock_branch.chat_model
                ),
                parse_param=ParseParam(
                    response_format=BaseResponse,
                    handle_validation="return_value",
                    imodel=mock_branch.parse_model,
                ),
                field_models=[spec],
            )

            assert result == mock_response

    def test_prepare_operate_kw_with_spec_and_actions(self, mock_branch):
        """Test prepare_operate_kw with Spec and action fields."""
        spec = Spec(str, name="custom_field", default="custom")

        result = prepare_operate_kw(
            mock_branch,
            instruction="Test with actions",
            field_models=[spec],
            actions=True,
            action_strategy="concurrent",
        )

        assert "instruction" in result
        assert "action_param" in result
        assert result["action_param"] is not None
        assert "operative" in result

    def test_prepare_operate_kw_with_spec_and_reason(self, mock_branch):
        """Test prepare_operate_kw with Spec and reason field."""
        spec = Spec(str, name="analysis_field")

        result = prepare_operate_kw(
            mock_branch,
            instruction="Test with reasoning",
            field_models=[spec],
            reason=True,
        )

        assert "instruction" in result
        assert "operative" in result
        # When reason=True, an operative should be created
        assert result["operative"] is not None


class TestReActWithSpec:
    """Test ReAct operation with Spec field models."""

    @pytest.mark.asyncio
    async def test_react_with_spec_fields(self, mock_branch):
        """Test ReAct with Spec field models."""
        spec = Spec(str, name="custom_analysis", default="")

        # Mock branch methods for ReAct
        mock_branch.communicate = AsyncMock(
            return_value={
                "analysis": "test analysis",
                "custom_analysis": "custom value",
            }
        )

        from lionagi.operations.ReAct import ReAct

        # Create a mock for ReAct that accepts field_models
        with patch.object(ReAct, "ReAct") as mock_react:
            mock_react.return_value = {
                "analysis": "final analysis",
                "custom_analysis": "final custom",
            }

            # Call ReAct with Spec
            result = await mock_react(
                mock_branch,
                instruct={"instruction": "Analyze this"},
                field_models=[spec],
            )

            assert "analysis" in result
            assert "custom_analysis" in result

    def test_spec_conversion_in_react_context(self):
        """Test that Specs are properly converted in ReAct context."""
        spec = Spec(
            list[str],
            name="react_field",
            default_factory=list,
            description="ReAct specific field",
        )

        # Convert to FieldModel
        fm = FieldModel.from_spec(spec)

        assert fm.name == "react_field"
        assert fm.annotation == list[str]
        assert fm.description == "ReAct specific field"
        assert callable(fm.default_factory)

        # Verify it can create a Pydantic field
        field = fm.create_field()
        assert field is not None
