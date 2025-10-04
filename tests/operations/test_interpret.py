# tests/branch_ops/test_interpret.py

from unittest.mock import AsyncMock

import pytest

from lionagi.protocols.generic.event import EventStatus
from lionagi.service.connections.api_calling import APICalling
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.providers.oai_ import _get_oai_config
from lionagi.service.imodel import iModel
from lionagi.service.third_party.openai_models import (
    OpenAIChatCompletionsRequest,
)
from lionagi.session.branch import Branch


def make_mocked_branch_for_interpret():
    branch = Branch(user="tester_fixture", name="BranchForTests_Interpret")

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
    mock_chat_model = iModel(
        provider="openai", model="gpt-4.1-mini", api_key="test_key"
    )
    mock_chat_model.invoke = mock_invoke

    branch.chat_model = mock_chat_model
    return branch


@pytest.mark.asyncio
async def test_interpret_basic():
    """
    branch.interpret(...) => calls branch.communicate(...) with skip_validation,
    returning 'mocked_response_string'.
    """
    branch = make_mocked_branch_for_interpret()

    refined_prompt = await branch.interpret(
        text="User's raw input", domain="some_domain", style="concise"
    )
    assert refined_prompt == "mocked_response_string"
    assert len(branch.messages) == 0
