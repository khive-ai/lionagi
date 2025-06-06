# tests/branch_ops/test_communicate.py

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


def make_mocked_branch_for_communicate():
    """
    Returns a Branch whose chat_model invoke yields "mocked_response_string".
    """
    branch = Branch(user="tester_fixture", name="BranchForTests_Communicate")

    async def _fake_invoke(**kwargs):
        endpoint = Endpoint(config=OPENAI_CHAT_ENDPOINT_CONFIG)
        fake_call = APICalling(
            payload={"model": "gpt-4o-mini", "messages": []},
            headers={"Authorization": "Bearer test"},
            endpoint=endpoint,
        )
        fake_call.execution.response = '{"data":"mocked_response_string"}'
        fake_call.execution.status = EventStatus.COMPLETED
        return fake_call

    async_mock_invoke = AsyncMock(side_effect=_fake_invoke)
    mock_chat_model = iModel(
        provider="openai", model="gpt-4o-mini", api_key="test_key"
    )
    mock_chat_model.invoke = async_mock_invoke

    branch.chat_model = mock_chat_model
    return branch


@pytest.mark.asyncio
async def test_communicate_no_validation():
    """
    If skip_validation=True, branch.communicate(...) should directly return the raw string.
    """
    branch = make_mocked_branch_for_communicate()

    result = await branch.communicate(
        instruction="User says hi", skip_validation=True
    )
    assert result == '{"data":"mocked_response_string"}'

    # If your updated code doesn't store messages, or does so differently, adjust accordingly:
    assert len(branch.messages) == 2


@pytest.mark.asyncio
async def test_communicate_with_model_validation():
    """
    If we provide a request_model, the final response is parsed into that model.
    """

    class MySimpleModel(BaseModel):
        data: str = "default_data"

    branch = make_mocked_branch_for_communicate()

    parsed = await branch.communicate(
        instruction="Send typed output",
        request_model=MySimpleModel,
    )
    # We'll assume your code sets parsed.data = "mocked_response_string"
    assert parsed.data == "mocked_response_string"
    assert len(branch.messages) == 2
