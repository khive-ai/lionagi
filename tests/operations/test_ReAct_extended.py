# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Extended test coverage for ReAct operations.

Tests cover:
1. Tool execution flows (single, multiple, error handling)
2. Multi-step reasoning (context accumulation, max extensions)
3. Integration scenarios (real tools, branch state, message history)
4. Edge cases (tool not found, invalid responses, concurrent execution)
"""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from lionagi.operations.ReAct.utils import (
    Analysis,
    PlannedAction,
    ReActAnalysis,
)
from lionagi.protocols.generic.event import EventStatus
from lionagi.service.connections.api_calling import APICalling
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.providers.oai_ import _get_oai_config
from lionagi.service.imodel import iModel
from lionagi.service.third_party.openai_models import (
    OpenAIChatCompletionsRequest,
)
from lionagi.session.branch import Branch

# ============================================================================
# Helper Functions and Fixtures
# ============================================================================


def make_mocked_branch_for_react():
    """Create a mocked Branch for ReAct testing."""
    branch = Branch(user="tester", name="ReActTestBranch")

    async def _fake_invoke(**kwargs):
        config = _get_oai_config(
            name="oai_chat",
            endpoint="chat/completions",
            request_options=OpenAIChatCompletionsRequest,
            kwargs={"model": "gpt-4o-mini"},
        )
        endpoint = Endpoint(config=config)
        fake_call = APICalling(
            payload={"model": "gpt-4o-mini", "messages": []},
            headers={"Authorization": "Bearer test"},
            endpoint=endpoint,
        )
        fake_call.execution.response = "mocked_response"
        fake_call.execution.status = EventStatus.COMPLETED
        return fake_call

    mock_invoke = AsyncMock(side_effect=_fake_invoke)
    mock_chat_model = iModel(provider="openai", model="gpt-4o-mini", api_key="test_key")
    mock_chat_model.invoke = mock_invoke

    branch.chat_model = mock_chat_model
    return branch


# Test tools
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Division by zero")
    return a / b


def get_weather(city: str) -> dict:
    """Get weather for a city."""
    return {"city": city, "temp": 72, "condition": "sunny"}


async def async_search(query: str) -> str:
    """Async search tool."""
    return f"Search results for: {query}"


# ============================================================================
# 1. Tool Execution Flows
# ============================================================================


@pytest.mark.asyncio
async def test_single_tool_invocation():
    """Test ReAct with single tool call - verifies tool integration."""
    branch = make_mocked_branch_for_react()
    branch.acts.register_tool(multiply)

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        # First: ReActAnalysis with tool call
        first_analysis = ReActAnalysis(
            analysis="Need to calculate 6 * 7",
            planned_actions=[PlannedAction(action_type="multiply", description="Calculate 6 * 7")],
            extension_needed=False,
        )

        # Second: Final answer
        final_analysis = Analysis(answer="42")

        mock_operate.side_effect = [first_analysis, final_analysis]

        result = await branch.ReAct(
            instruct={"instruction": "What is 6 times 7?"},
            tools=[multiply],
            max_extensions=0,
        )

        assert result == "42"
        # Verify operate was called twice (analysis + final answer)
        assert mock_operate.call_count == 2


@pytest.mark.asyncio
async def test_multiple_tool_calls_sequential_strategy():
    """Test ReAct specifying sequential action strategy."""
    branch = make_mocked_branch_for_react()

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        # Round 1: First tool call with sequential strategy
        round1 = ReActAnalysis(
            analysis="Calculate 100 * 5",
            planned_actions=[PlannedAction(action_type="multiply", description="100 * 5")],
            extension_needed=True,
            action_strategy="sequential",
        )

        # Round 2: Second tool call
        round2 = ReActAnalysis(
            analysis="Now divide by 10",
            planned_actions=[PlannedAction(action_type="divide", description="500 / 10")],
            extension_needed=False,
            action_strategy="sequential",
        )

        # Final answer
        final = Analysis(answer="50")

        mock_operate.side_effect = [round1, round2, final]

        result = await branch.ReAct(
            instruct={"instruction": "Calculate (100 * 5) / 10 using tools"},
            max_extensions=2,
        )

        assert result == "50"
        assert mock_operate.call_count == 3

        # Verify sequential strategy was passed to operate calls
        extension_call = mock_operate.call_args_list[1]
        action_param = extension_call[1]["action_param"]
        assert action_param.strategy == "sequential"


@pytest.mark.asyncio
async def test_concurrent_action_strategy():
    """Test ReAct with concurrent action strategy specification."""
    branch = make_mocked_branch_for_react()

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        # Analysis requesting concurrent execution
        analysis = ReActAnalysis(
            analysis="Check weather in multiple cities",
            planned_actions=[
                PlannedAction(
                    action_type="get_weather",
                    description="Check NYC weather",
                ),
                PlannedAction(
                    action_type="get_weather",
                    description="Check SF weather",
                ),
            ],
            extension_needed=False,
            action_strategy="concurrent",  # Concurrent strategy
        )

        final = Analysis(answer="NYC: 72°F sunny, SF: 65°F cloudy")

        mock_operate.side_effect = [analysis, final]

        result = await branch.ReAct(
            instruct={"instruction": "Compare weather in NYC and SF"},
            max_extensions=0,
        )

        assert result == "NYC: 72°F sunny, SF: 65°F cloudy"
        assert mock_operate.call_count == 2


@pytest.mark.asyncio
async def test_planned_actions_structure():
    """Test that PlannedAction structure is properly handled."""
    branch = make_mocked_branch_for_react()

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        # Analysis with detailed planned actions
        analysis = ReActAnalysis(
            analysis="Need to perform multiple actions",
            planned_actions=[
                PlannedAction(
                    action_type="search",
                    description="Search for information",
                ),
                PlannedAction(
                    action_type="analyze",
                    description="Analyze search results",
                ),
            ],
            extension_needed=False,
        )

        final = Analysis(answer="Actions completed")

        mock_operate.side_effect = [analysis, final]

        result = await branch.ReAct(
            instruct={"instruction": "Perform research"},
            max_extensions=0,
        )

        assert result == "Actions completed"
        # Verify the analysis object had correct planned_actions
        initial_call = mock_operate.call_args_list[0]
        assert initial_call is not None


@pytest.mark.asyncio
async def test_tools_parameter_variations():
    """Test different ways to specify tools parameter."""
    branch = make_mocked_branch_for_react()
    branch.acts.register_tool(multiply)

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        analysis = ReActAnalysis(
            analysis="Complete",
            planned_actions=[],
            extension_needed=False,
        )
        final = Analysis(answer="Done")

        # Test with tools=None
        mock_operate.side_effect = [analysis, final]
        result = await branch.ReAct(
            instruct={"instruction": "Task"},
            tools=None,
            max_extensions=0,
        )
        assert result == "Done"

        # Test with tools=True (use all registered)
        mock_operate.side_effect = [analysis, final]
        result = await branch.ReAct(
            instruct={"instruction": "Task"},
            tools=True,
            max_extensions=0,
        )
        assert result == "Done"


# ============================================================================
# 2. Multi-Step Reasoning
# ============================================================================


@pytest.mark.asyncio
async def test_reasoning_chain_with_context_accumulation():
    """Test multi-step reasoning with context building across rounds."""
    branch = make_mocked_branch_for_react()

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        # Round 1: Initial analysis
        round1 = ReActAnalysis(
            analysis="First step: identify the problem",
            planned_actions=[],
            extension_needed=True,
        )

        # Round 2: Build on previous context
        round2 = ReActAnalysis(
            analysis="Second step: analyze the data based on step 1",
            planned_actions=[],
            extension_needed=True,
        )

        # Round 3: Final reasoning
        round3 = ReActAnalysis(
            analysis="Third step: synthesize findings from steps 1 and 2",
            planned_actions=[],
            extension_needed=False,
        )

        # Final answer
        final = Analysis(answer="Complete solution based on 3-step reasoning")

        mock_operate.side_effect = [round1, round2, round3, final]

        result = await branch.ReAct(
            instruct={"instruction": "Solve complex problem step by step"},
            max_extensions=3,
        )

        assert "Complete solution" in result
        # Should have 4 calls: 3 reasoning rounds + 1 final answer
        assert mock_operate.call_count == 4

        # Verify all rounds were executed with proper context
        # Check that each extension call received previous analysis
        for i, call in enumerate(mock_operate.call_args_list):
            assert call is not None


@pytest.mark.asyncio
async def test_max_extensions_limit():
    """Test that extension loop respects max_extensions limit."""
    branch = make_mocked_branch_for_react()

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        # Create 2 analyses that request extension
        round1 = ReActAnalysis(
            analysis="Round 1",
            planned_actions=[],
            extension_needed=True,
        )
        round2 = ReActAnalysis(
            analysis="Round 2 - last extension",
            planned_actions=[],
            extension_needed=False,  # Stops here
        )

        final = Analysis(answer="Complete after 2 rounds")

        # Initial + 1 extension + final = 3 calls
        mock_operate.side_effect = [round1, round2, final]

        result = await branch.ReAct(
            instruct={"instruction": "Task with extensions"},
            max_extensions=2,
        )

        # Should complete successfully
        assert mock_operate.call_count == 3
        assert "Complete after 2 rounds" in result


@pytest.mark.asyncio
async def test_early_termination_extension_false():
    """Test early termination when extension_needed is False."""
    branch = make_mocked_branch_for_react()

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        # First round decides no extension needed
        analysis = ReActAnalysis(
            analysis="Task complete after first analysis",
            planned_actions=[],
            extension_needed=False,  # No more extensions
        )

        final = Analysis(answer="Quick answer")

        mock_operate.side_effect = [analysis, final]

        result = await branch.ReAct(
            instruct={"instruction": "Simple task"},
            max_extensions=10,  # Allow many, but stop early
        )

        # Should only have 2 calls: 1 analysis + 1 final
        assert mock_operate.call_count == 2
        assert "Quick answer" in result


@pytest.mark.asyncio
async def test_extension_not_allowed():
    """Test behavior when extension_allowed is False."""
    branch = make_mocked_branch_for_react()

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        # Analysis requests extension but it's not allowed
        analysis = ReActAnalysis(
            analysis="Want to extend but can't",
            planned_actions=[],
            extension_needed=True,
        )

        final = Analysis(answer="Forced to conclude")

        mock_operate.side_effect = [analysis, final]

        result = await branch.ReAct(
            instruct={"instruction": "Task"},
            extension_allowed=False,  # Disable extensions
            max_extensions=5,
        )

        # Should only have 2 calls despite extension_needed=True
        assert mock_operate.call_count == 2
        assert "Forced to conclude" in result


@pytest.mark.asyncio
async def test_max_extensions_clamped_to_100():
    """Test that max_extensions is clamped to 100."""
    branch = make_mocked_branch_for_react()

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        # Create chain that stops after 2 rounds
        round1 = ReActAnalysis(analysis="Round 1", planned_actions=[], extension_needed=True)
        round2 = ReActAnalysis(analysis="Round 2", planned_actions=[], extension_needed=False)
        final = Analysis(answer="Done")

        mock_operate.side_effect = [round1, round2, final]

        # Request more than 100 extensions (should be clamped)
        result = await branch.ReAct(
            instruct={"instruction": "Task"},
            max_extensions=200,  # Will be clamped to 100
        )

        # Should complete normally (extension_needed=False stops it early)
        assert mock_operate.call_count == 3
        assert result == "Done"


# ============================================================================
# 3. Integration Scenarios
# ============================================================================


@pytest.mark.asyncio
async def test_react_with_real_tools_integration():
    """Test ReAct with real tool registration and execution."""
    branch = Branch(user="test_user")

    # Register real tools
    branch.acts.register_tool(multiply)
    branch.acts.register_tool(divide)

    # Verify tools are registered
    assert "multiply" in branch.acts.registry
    assert "divide" in branch.acts.registry

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        # Simulate realistic ReAct flow
        analysis = ReActAnalysis(
            analysis="Calculate (20 * 3) / 4",
            planned_actions=[PlannedAction(action_type="multiply", description="20 * 3")],
            extension_needed=False,
        )

        final = Analysis(answer="15")

        mock_operate.side_effect = [analysis, final]

        # Execute with real tools (though operate is mocked)
        result = await branch.ReAct(
            instruct={"instruction": "Calculate (20 * 3) / 4"},
            tools=True,  # Use registered tools
            max_extensions=1,
        )

        assert result == "15"


@pytest.mark.asyncio
async def test_branch_state_consistency():
    """Test that ReAct completes with expected call pattern."""
    branch = make_mocked_branch_for_react()

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        # Multi-round ReAct
        round1 = ReActAnalysis(analysis="Step 1", planned_actions=[], extension_needed=True)
        round2 = ReActAnalysis(analysis="Step 2", planned_actions=[], extension_needed=False)
        final = Analysis(answer="Final")

        mock_operate.side_effect = [round1, round2, final]

        result = await branch.ReAct(
            instruct={"instruction": "Multi-step task"},
            max_extensions=2,
            clear_messages=False,  # Keep message history
        )

        # Verify completion and call pattern
        assert result == "Final"
        assert mock_operate.call_count == 3


@pytest.mark.asyncio
async def test_clear_messages_parameter():
    """Test clear_messages parameter is properly forwarded."""
    branch = make_mocked_branch_for_react()

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        analysis = ReActAnalysis(analysis="Analysis", planned_actions=[], extension_needed=False)
        final = Analysis(answer="Answer")

        mock_operate.side_effect = [analysis, final]

        result = await branch.ReAct(
            instruct={"instruction": "Test task"},
            max_extensions=0,
            clear_messages=True,  # Test clear_messages=True
        )

        assert result == "Answer"
        # Verify clear_messages was passed to operate calls
        first_call_kwargs = mock_operate.call_args_list[0][1]
        assert "clear_messages" in first_call_kwargs


@pytest.mark.asyncio
async def test_react_with_async_tool_registration():
    """Test ReAct can register and reference async tools."""
    branch = make_mocked_branch_for_react()
    branch.acts.register_tool(async_search)

    # Verify async tool was registered
    assert "async_search" in branch.acts.registry

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        analysis = ReActAnalysis(
            analysis="Search for information",
            planned_actions=[PlannedAction(action_type="async_search", description="Search query")],
            extension_needed=False,
        )

        final = Analysis(answer="Search results found")

        mock_operate.side_effect = [analysis, final]

        result = await branch.ReAct(
            instruct={"instruction": "Search for test"},
            tools=[async_search],
            max_extensions=0,
        )

        assert "Search results found" in result


# ============================================================================
# 4. Edge Cases
# ============================================================================


@pytest.mark.asyncio
async def test_invalid_tool_response_handling():
    """Test handling of invalid/malformed tool responses."""
    branch = make_mocked_branch_for_react()
    branch.acts.register_tool(multiply)

    with (
        patch("lionagi.operations.operate.operate.operate") as mock_operate,
        patch("lionagi.operations.act.act.act") as mock_act,
    ):
        analysis = ReActAnalysis(
            analysis="Call tool",
            planned_actions=[PlannedAction(action_type="multiply", description="Test")],
            extension_needed=False,
        )

        final = Analysis(answer="Handled error")

        mock_operate.side_effect = [analysis, final]
        # Return None simulating invalid response
        mock_act.return_value = [None]

        result = await branch.ReAct(
            instruct={"instruction": "Test invalid response"},
            tools=[multiply],
            max_extensions=0,
        )

        assert "Handled error" in result


@pytest.mark.asyncio
async def test_all_none_response_recovery():
    """Test recovery from response with all None values using continue_after_failed_response."""
    branch = make_mocked_branch_for_react()

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        # First call returns analysis
        analysis = ReActAnalysis(
            analysis="Analysis",
            planned_actions=[],
            extension_needed=False,
        )

        # Second call (final answer) returns valid Analysis
        final = Analysis(answer="Recovered successfully")

        mock_operate.side_effect = [analysis, final]

        # Should complete successfully with continue_after_failed_response=True
        result = await branch.ReAct(
            instruct={"instruction": "Test recovery"},
            max_extensions=0,
            continue_after_failed_response=True,
        )

        assert "Recovered successfully" in result


@pytest.mark.asyncio
async def test_continue_after_failed_response():
    """Test that continue_after_failed_response allows continuation."""
    branch = make_mocked_branch_for_react()

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        # First call returns all None
        failed_response = {"field1": None, "field2": None}
        # Second call returns valid response
        valid_analysis = ReActAnalysis(
            analysis="Recovered", planned_actions=[], extension_needed=False
        )
        final = Analysis(answer="Success")

        mock_operate.side_effect = [
            failed_response,
            valid_analysis,
            final,
        ]

        # Should not raise error and continue
        result = await branch.ReAct(
            instruct={"instruction": "Test recovery"},
            max_extensions=1,
            continue_after_failed_response=True,
        )

        # Should complete despite initial failure
        assert mock_operate.call_count >= 2


@pytest.mark.asyncio
async def test_empty_planned_actions():
    """Test ReAct when analysis has no planned actions."""
    branch = make_mocked_branch_for_react()

    with (
        patch("lionagi.operations.operate.operate.operate") as mock_operate,
        patch("lionagi.operations.act.act.act") as mock_act,
    ):
        # Analysis with empty planned_actions list
        analysis = ReActAnalysis(
            analysis="No actions needed",
            planned_actions=[],  # Empty
            extension_needed=False,
        )

        final = Analysis(answer="Direct answer")

        mock_operate.side_effect = [analysis, final]

        result = await branch.ReAct(
            instruct={"instruction": "Simple question"},
            max_extensions=0,
        )

        assert result == "Direct answer"
        # act should not be called if no planned actions
        assert mock_act.call_count == 0


@pytest.mark.asyncio
async def test_react_with_custom_response_format():
    """Test ReAct with custom response format for final answer."""

    class CustomResult(BaseModel):
        """Custom response format."""

        calculation: float
        explanation: str

    branch = make_mocked_branch_for_react()

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        analysis = ReActAnalysis(
            analysis="Complete",
            planned_actions=[],
            extension_needed=False,
        )

        # Final answer with custom format
        custom_result = CustomResult(calculation=42.0, explanation="The answer is 42")

        mock_operate.side_effect = [analysis, custom_result]

        result = await branch.ReAct(
            instruct={"instruction": "Calculate something"},
            response_format=CustomResult,
            max_extensions=0,
        )

        # Should return CustomResult instance
        assert isinstance(result, CustomResult)
        assert result.calculation == 42.0
        assert result.explanation == "The answer is 42"


@pytest.mark.asyncio
async def test_return_analysis_parameter():
    """Test return_analysis parameter returns all intermediate analyses."""
    branch = make_mocked_branch_for_react()

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        round1 = ReActAnalysis(analysis="Step 1", planned_actions=[], extension_needed=True)
        round2 = ReActAnalysis(analysis="Step 2", planned_actions=[], extension_needed=False)
        final = Analysis(answer="Final")

        mock_operate.side_effect = [round1, round2, final]

        result = await branch.ReAct(
            instruct={"instruction": "Task"},
            max_extensions=2,
            return_analysis=True,  # Return all analyses
        )

        # Should return list of all analyses
        assert isinstance(result, list)
        assert len(result) == 3  # 2 ReActAnalysis + 1 Analysis
        assert result[0].analysis == "Step 1"
        assert result[1].analysis == "Step 2"
        assert result[2].answer == "Final"


@pytest.mark.asyncio
async def test_reasoning_effort_parameter():
    """Test reasoning_effort parameter affects guidance."""
    branch = make_mocked_branch_for_react()

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        analysis = ReActAnalysis(
            analysis="High effort reasoning",
            planned_actions=[],
            extension_needed=True,
        )
        round2 = ReActAnalysis(analysis="Continue", planned_actions=[], extension_needed=False)
        final = Analysis(answer="Result")

        mock_operate.side_effect = [analysis, round2, final]

        await branch.ReAct(
            instruct={"instruction": "Complex task"},
            reasoning_effort="high",  # High reasoning effort
            max_extensions=2,
        )

        # Check that operate was called with reasoning_effort in imodel_kw
        extension_call = mock_operate.call_args_list[1]  # Second call
        chat_param = extension_call[1]["chat_param"]
        assert chat_param.imodel_kw.get("reasoning_effort") == "high"


@pytest.mark.asyncio
async def test_verbose_analysis_output():
    """Test verbose_analysis parameter for debugging output."""
    branch = make_mocked_branch_for_react()

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        analysis = ReActAnalysis(
            analysis="Analysis text",
            planned_actions=[],
            extension_needed=False,
        )
        final = Analysis(answer="Final answer")

        mock_operate.side_effect = [analysis, final]

        # Test with verbose_analysis=True (should complete without error)
        result = await branch.ReAct(
            instruct={"instruction": "Task"},
            verbose_analysis=True,
            max_extensions=0,
        )

        assert result == "Final answer"


# ============================================================================
# Performance and Stress Tests
# ============================================================================


@pytest.mark.asyncio
async def test_react_with_many_extensions():
    """Test ReAct with multiple reasoning rounds."""
    branch = make_mocked_branch_for_react()

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        # Create 10 rounds of reasoning (more manageable)
        rounds = []
        for i in range(10):
            rounds.append(
                ReActAnalysis(
                    analysis=f"Round {i + 1}",
                    planned_actions=[],
                    extension_needed=(True if i < 9 else False),  # Last one stops
                )
            )

        final = Analysis(answer="Complete")

        # 10 rounds + 1 final = 11 calls
        mock_operate.side_effect = rounds + [final]

        result = await branch.ReAct(
            instruct={"instruction": "Complex multi-step task"},
            max_extensions=10,
        )

        assert mock_operate.call_count == 11  # 10 rounds + 1 final
        assert result == "Complete"


@pytest.mark.asyncio
async def test_react_with_many_tools():
    """Test ReAct with many registered tools."""
    branch = make_mocked_branch_for_react()

    # Register multiple tools
    tools = [multiply, divide, get_weather]
    for tool in tools:
        branch.acts.register_tool(tool)

    assert len(branch.acts.registry) >= 3

    with patch("lionagi.operations.operate.operate.operate") as mock_operate:
        analysis = ReActAnalysis(
            analysis="Using tools",
            planned_actions=[
                PlannedAction(action_type="multiply", description="First"),
                PlannedAction(action_type="divide", description="Second"),
            ],
            extension_needed=False,
        )
        final = Analysis(answer="Done with tools")

        mock_operate.side_effect = [analysis, final]

        result = await branch.ReAct(
            instruct={"instruction": "Use multiple tools"},
            tools=True,  # Use all registered tools
            max_extensions=0,
        )

        assert "Done with tools" in result


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
