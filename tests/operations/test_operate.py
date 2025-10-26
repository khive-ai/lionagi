# tests/branch_ops/test_operate.py

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from lionagi.protocols.generic.event import EventStatus
from lionagi.service.connections.api_calling import APICalling
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.providers.oai_ import _get_oai_config
from lionagi.service.imodel import iModel
from lionagi.service.third_party.openai_models import (
    OpenAIChatCompletionsRequest,
)
from lionagi.session.branch import Branch


def make_mocked_branch_for_operate():
    branch = Branch(user="tester_fixture", name="BranchForTests_Operate")

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
        fake_call.execution.response = '{"foo":"mocked_response_string"}'
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


@pytest.mark.asyncio
async def test_operate_with_actions_preserves_response_data():
    """
    Regression test: when operate() returns a structured response with actions,
    the action_responses should be merged with the original response data.

    Previously, only action_responses were returned, losing original data.
    """

    class ResponseModel(BaseModel):
        answer: str
        confidence: float

    # Mock branch with response that includes action requests
    branch = Branch(user="tester", name="ActionTest")

    async def _fake_invoke_with_actions(**kwargs):
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
        # Response with both data AND action requests
        fake_call.execution.response = """{
            "answer": "42",
            "confidence": 0.95,
            "action_required": true,
            "action_requests": [
                {"function": "add", "arguments": {"a": 1, "b": 2}}
            ]
        }"""
        fake_call.execution.status = EventStatus.COMPLETED
        return fake_call

    mock_invoke = AsyncMock(side_effect=_fake_invoke_with_actions)
    mock_chat_model = iModel(
        provider="openai", model="gpt-4.1-mini", api_key="test_key"
    )
    mock_chat_model.invoke = mock_invoke
    branch.chat_model = mock_chat_model

    # Register a simple tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    branch.register_tools([add])

    # Execute with actions=True
    result = await branch.operate(
        instruction="Calculate something",
        response_format=ResponseModel,
        actions=True,
        invoke_actions=True,
    )

    # CRITICAL: Result should have BOTH original response data AND action_responses
    assert hasattr(result, "answer"), "Original 'answer' field missing"
    assert hasattr(result, "confidence"), "Original 'confidence' field missing"
    assert hasattr(
        result, "action_responses"
    ), "action_responses field missing"

    # Verify original data is preserved
    assert result.answer == "42"
    assert result.confidence == 0.95

    # Verify action_responses were added
    assert len(result.action_responses) == 1
    assert result.action_responses[0].function == "add"
