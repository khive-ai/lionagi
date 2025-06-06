# tests/branch_ops/test_operate.py

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from lionagi.protocols.generic.event import EventStatus
from lionagi.service.connections.api_calling import APICalling
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.providers.oai_ import (
    OPENAI_CHAT_ENDPOINT_CONFIG,
)
from lionagi.service.imodel import iModel
from lionagi.session.branch import Branch


def make_mocked_branch_for_operate():
    branch = Branch(user="tester_fixture", name="BranchForTests_Operate")

    async def _fake_invoke(**kwargs):
        endpoint = Endpoint(config=OPENAI_CHAT_ENDPOINT_CONFIG)
        fake_call = APICalling(
            payload={"model": "gpt-4o-mini", "messages": []},
            headers={"Authorization": "Bearer test"},
            endpoint=endpoint,
        )
        fake_call.execution.response = '{"foo":"mocked_response_string"}'
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
async def test_operate_no_actions_no_validation():
    """
    branch.operate(...) with invoke_actions=False and skip_validation=True => returns raw string.
    """
    branch = make_mocked_branch_for_operate()
    final = await branch.operate(
        instruction="Just a test", invoke_actions=False, skip_validation=True
    )
    assert final == '{"foo":"mocked_response_string"}'
    assert len(branch.messages) == 2


@pytest.mark.asyncio
async def test_operate_with_validation():
    """
    If we pass a response_format, it should parse "mocked_response_string" into that model.
    """

    class ExampleModel(BaseModel):
        foo: str

    branch = make_mocked_branch_for_operate()

    final = await branch.operate(
        instruction="Expect typed output",
        response_format=ExampleModel,
        invoke_actions=False,
    )
    assert final.foo == "mocked_response_string"
    assert len(branch.messages) == 2
