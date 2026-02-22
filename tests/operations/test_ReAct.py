# tests/operations/test_ReAct.py
from unittest.mock import AsyncMock, patch

import pytest

# We'll import or define the ReActAnalysis class to create a real instance:
from lionagi.operations.ReAct.utils import ReActAnalysis
from lionagi.protocols.generic.event import EventStatus
from lionagi.service.connections.api_calling import APICalling
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.providers.oai_ import _get_oai_config
from lionagi.service.imodel import iModel
from lionagi.service.third_party.openai_models import (
    OpenAIChatCompletionsRequest,
)
from lionagi.session.branch import Branch


def make_mocked_branch_for_react():
    branch = Branch(user="tester_fixture", name="BranchForTests_ReAct")

    async def _fake_invoke(**kwargs):
        config = _get_oai_config(
            name="oai_chat",
            endpoint="chat/completions",
            request_options=OpenAIChatCompletionsRequest,
            kwargs={"model": "gpt-4.1-mini"},
        )
        endpoint = Endpoint(config=config)
        fake_call = APICalling(
            payload={"model": "gpt-4.1-mini", "messages": []},
            headers={"Authorization": "Bearer test"},
            endpoint=endpoint,
        )
        fake_call.execution.response = "mocked_response_string"
        fake_call.execution.status = EventStatus.COMPLETED
        return fake_call

    mock_invoke = AsyncMock(side_effect=_fake_invoke)
    mock_chat_model = iModel(provider="openai", model="gpt-4.1-mini", api_key="test_key")
    mock_chat_model.invoke = mock_invoke

    branch.chat_model = mock_chat_model
    return branch


@pytest.mark.asyncio
async def test_react_basic_flow():
    """
    ReAct(...) => calls branch.operate for analysis, then for final answer.
    We'll patch branch.operate to yield a real ReActAnalysis -> Analysis object.
    """
    from lionagi.operations.ReAct.utils import Analysis

    branch = make_mocked_branch_for_react()

    # 1) Create a mock ReActAnalysis object with extension_needed=False so we skip expansions:
    class FakeAnalysis(ReActAnalysis):
        extension_needed: bool = False

    # 2) Create call counter to return different values for different calls
    call_count = 0

    async def mock_operate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call - return ReActAnalysis
            return FakeAnalysis(
                analysis="intermediate_reasoning",
                extension_needed=False,
                planned_actions=[],
            )
        else:
            # Second call - return final Analysis
            return Analysis(answer="final_answer_mock")

    # 3) Patch operate
    with patch(
        "lionagi.operations.operate.operate.operate",
        new=AsyncMock(side_effect=mock_operate),
    ):
        res = await branch.ReAct(
            instruct={"instruction": "Solve a puzzle with ReAct strategy"},
            interpret=False,
            extension_allowed=False,
        )

    # 4) Confirm we got the final answer as a string
    assert res == "final_answer_mock"
