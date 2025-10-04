"""
Centralized Mock Factory for LionAGI Testing

Provides standardized mocks for common testing patterns to reduce code duplication
and ensure consistent mocking strategies across the test suite.

Usage Examples:
    # Create a standard mocked branch
    branch = LionAGIMockFactory.create_mocked_branch()

    # Create with custom response
    branch = LionAGIMockFactory.create_mocked_branch(response="custom response")

    # Create with structured response
    branch = LionAGIMockFactory.create_mocked_branch(
        response={"result": "structured data"}
    )
"""

from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock

from lionagi.protocols.generic.event import EventStatus
from lionagi.service.connections.api_calling import APICalling
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.providers.oai_ import _get_oai_config
from lionagi.service.imodel import iModel
from lionagi.service.third_party.openai_models import (
    OpenAIChatCompletionsRequest,
)
from lionagi.session.branch import Branch


class LionAGIMockFactory:
    """Centralized factory for creating standardized mocks in LionAGI tests."""

    @staticmethod
    def create_mocked_branch(
        name: str = "TestBranch",
        user: str = "tester",
        response: str | dict[str, Any] = "mocked_response_string",
        status: EventStatus = EventStatus.COMPLETED,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
    ) -> Branch:
        """
        Create a Branch with mocked iModel that returns configurable responses.

        This replaces the 20+ line manual mock setup with a single line call.

        Args:
            name: Branch name for identification
            user: User identifier for the branch
            response: Response data to return from mocked API calls
            status: Event status for mocked calls
            provider: AI provider to mock
            model: Model name to use in mock

        Returns:
            Branch instance with fully configured mock iModel
        """
        branch = Branch(user=user, name=name)

        # Create the mock iModel with standardized behavior
        mock_chat_model = LionAGIMockFactory.create_mocked_imodel(
            provider=provider, model=model, response=response, status=status
        )

        # Set both chat and parse models to support all branch operations
        branch.chat_model = mock_chat_model
        branch.parse_model = mock_chat_model

        return branch

    @staticmethod
    def create_mocked_imodel(
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        response: str | dict[str, Any] = "mocked_response_string",
        responses: list[str | dict[str, Any]] | None = None,
        status: EventStatus = EventStatus.COMPLETED,
        api_key: str = "test_key",
    ) -> iModel:
        """
        Create a mocked iModel with configurable responses.

        Args:
            provider: AI provider name
            model: Model name
            response: Single response for all calls (if responses not provided)
            responses: Sequence of responses for multiple calls
            status: Event status for mocked calls
            api_key: API key for model initialization

        Returns:
            iModel instance with mocked invoke method
        """
        mock_model = iModel(provider=provider, model=model, api_key=api_key)

        if responses:
            # Create sequence of responses for multiple calls
            response_iter = iter(responses)

            async def _sequence_invoke(**kwargs):
                try:
                    current_response = next(response_iter)
                except StopIteration:
                    # If we run out of responses, use the last one
                    current_response = responses[-1] if responses else response

                return LionAGIMockFactory.create_api_calling_mock(
                    response_data=current_response, status=status, model=model
                )

            mock_model.invoke = AsyncMock(side_effect=_sequence_invoke)
        else:
            # Single response for all calls
            async def _single_invoke(**kwargs):
                return LionAGIMockFactory.create_api_calling_mock(
                    response_data=response, status=status, model=model
                )

            mock_model.invoke = AsyncMock(side_effect=_single_invoke)

        return mock_model

    @staticmethod
    def create_api_calling_mock(
        response_data: str | dict[str, Any] = "mocked_response_string",
        status: EventStatus = EventStatus.COMPLETED,
        model: str = "gpt-4o-mini",
        endpoint_config: dict[str, Any] | None = None,
    ) -> APICalling:
        """
        Create a mocked APICalling object with standardized structure.

        Args:
            response_data: Data to set in execution.response
            status: Event status for the execution
            model: Model name for payload
            endpoint_config: Custom endpoint configuration

        Returns:
            APICalling instance with mocked response data
        """
        if endpoint_config is None:
            endpoint_config = _get_oai_config(
                name="oai_chat",
                endpoint="chat/completions",
                request_options=OpenAIChatCompletionsRequest,
                kwargs={"model": model},
            )

        endpoint = Endpoint(config=endpoint_config)

        api_call = APICalling(
            payload={"model": model, "messages": []},
            headers={"Authorization": "Bearer test"},
            endpoint=endpoint,
        )

        # Set the response data that the system will actually use
        api_call.execution.response = response_data
        api_call.execution.status = status

        return api_call

    @staticmethod
    def create_mocked_session(
        branches: list[str] | None = None,
        default_branch_response: (
            str | dict[str, Any]
        ) = "mocked_response_string",
    ):
        """
        Create a Session with multiple mocked branches.

        Args:
            branches: List of branch names to create
            default_branch_response: Default response for all branches

        Returns:
            Session instance with mocked branches
        """
        from lionagi.session.session import Session

        session = Session()

        if branches:
            for branch_name in branches:
                branch = LionAGIMockFactory.create_mocked_branch(
                    name=branch_name, response=default_branch_response
                )
                session.branches[branch_name] = branch

        return session

    @staticmethod
    def create_error_response_mock(
        error_message: str = "Mocked API Error",
        error_code: str = "test_error",
        status: EventStatus = EventStatus.FAILED,
    ) -> APICalling:
        """
        Create a mocked APICalling object that represents an error response.

        Args:
            error_message: Error message to include
            error_code: Error code identifier
            status: Event status (typically ERROR)

        Returns:
            APICalling instance configured for error testing
        """
        api_call = LionAGIMockFactory.create_api_calling_mock(
            response_data={
                "error": {"message": error_message, "code": error_code}
            },
            status=status,
        )

        return api_call
