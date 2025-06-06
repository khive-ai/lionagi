# tests/operations/test_ReAct.py
from unittest.mock import AsyncMock, patch

import pytest

# We'll import or define the ReActAnalysis class to create a real instance:
from lionagi.operations.ReAct.utils import ReActAnalysis
from lionagi.protocols.generic.event import EventStatus
from lionagi.service.connections.api_calling import APICalling
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.providers.oai_ import (
    OPENAI_CHAT_ENDPOINT_CONFIG,
)
from lionagi.service.imodel import iModel
from lionagi.session.branch import Branch


def make_mocked_branch_for_react():
    branch = Branch(user="tester_fixture", name="BranchForTests_ReAct")

    async def _fake_invoke(**kwargs):
        endpoint = Endpoint(config=OPENAI_CHAT_ENDPOINT_CONFIG)
        fake_call = APICalling(
            payload={"model": "gpt-4o-mini", "messages": []},
            headers={"Authorization": "Bearer test"},
            endpoint=endpoint,
        )
        fake_call.execution.response = "mocked_response_string"
        fake_call.execution.status = EventStatus.COMPLETED
        return fake_call

    mock_invoke = AsyncMock(side_effect=_fake_invoke)
    mock_chat_model = iModel(
        provider="openai", model="gpt-4o-mini", api_key="test_key"
    )
    mock_chat_model.invoke = mock_invoke

    branch.chat_model = mock_chat_model
    return branch


@pytest.mark.asyncio
async def test_react_basic_flow():
    """
    ReAct(...) => calls operate for analysis, then calls branch.communicate for final answer.
    We'll patch them at the class level to yield a real ReActAnalysis -> "final_answer_mock".
    """
    branch = make_mocked_branch_for_react()

    # 1) Create a mock ReActAnalysis object with extension_needed=False so we skip expansions:
    class FakeAnalysis(ReActAnalysis):
        extension_needed: bool = False
        # optionally override other fields if needed

    # 2) Patch branch.operate to return a `FakeAnalysis` instance
    #    Patch branch.communicate to return "final_answer_mock"
    with patch(
        "lionagi.session.branch.Branch.operate",
        new=AsyncMock(
            return_value=FakeAnalysis(
                **{
                    "analysis": "final_answer_mock",
                    "extension_needed": False,
                    "action_requests": [],
                }
            )
        ),
    ):
        with patch(
            "lionagi.session.branch.Branch.communicate",
            new=AsyncMock(return_value="final_answer_mock"),
        ):
            res = await branch.ReAct(
                instruct={"instruction": "Solve a puzzle with ReAct strategy"},
                interpret=False,
                extension_allowed=False,
            )

    # 3) Confirm we got the final answer
    assert res.analysis == "final_answer_mock"
