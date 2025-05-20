"""Interface for communicating with LLM services.

The ServiceInterface is responsible for preparing requests to LLM services,
sending them via lionfuncs.network.adapters, and processing the responses.
"""

from typing import Any, Dict, List, Optional, Union

import lionfuncs.async_utils as async_utils
from lionfuncs.errors import LionError, LionNetworkError
from lionfuncs.network.adapters import OpenAIAdapter, AnthropicAdapter

from lionagi.models.message import Message


class ServiceInterface:
    """Interface for communicating with LLM services."""

    def __init__(
        self,
        adapter: Optional[Any] = None,
        adapter_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the ServiceInterface.

        Args:
            adapter: The lionfuncs.network.adapters adapter to use.
                If None, an OpenAIAdapter will be created.
            adapter_config: Configuration for the adapter.
        """
        self.adapter = adapter or OpenAIAdapter(**(adapter_config or {}))
        self.adapter_config = adapter_config or {}

    @classmethod
    def create(
        cls, provider: str, api_key: Optional[str] = None, **kwargs
    ) -> "ServiceInterface":
        """Create a ServiceInterface for a specific provider.

        Args:
            provider: The provider to use (e.g., "openai", "anthropic").
            api_key: The API key to use. If None, it will be loaded from
                environment variables.
            **kwargs: Additional configuration for the adapter.

        Returns:
            A ServiceInterface instance configured for the specified provider.

        Raises:
            LionError: If the provider is not supported.
        """
        # TODO: Implement provider-specific adapter creation
        # 1. Map the provider to the appropriate adapter class
        # 2. Create the adapter with the provided configuration
        # 3. Return a ServiceInterface instance with the adapter
        raise NotImplementedError("This method is not yet implemented.")

    async def prepare_request(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare a request for the LLM service.

        Args:
            messages: The messages to include in the request.
            tools: Optional list of tools to make available.
            **kwargs: Additional parameters for the request.

        Returns:
            The prepared request payload.
        """
        # TODO: Implement request preparation logic
        # 1. Convert messages to the format expected by the adapter
        # 2. Add tools if provided
        # 3. Add any additional parameters
        # 4. Return the prepared request payload
        raise NotImplementedError("This method is not yet implemented.")

    async def call_llm(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call the LLM service with the prepared request.

        Args:
            request: The prepared request payload.

        Returns:
            The raw response from the LLM service.

        Raises:
            LionNetworkError: If there is an error communicating with the service.
        """
        # TODO: Implement LLM call logic
        # 1. Call the adapter with the prepared request
        # 2. Handle any errors
        # 3. Return the raw response
        raise NotImplementedError("This method is not yet implemented.")

    async def process_response(
        self,
        response: Dict[str, Any],
    ) -> Union[Message, Dict[str, Any]]:
        """Process the response from the LLM service.

        Args:
            response: The raw response from the LLM service.

        Returns:
            A Message object or a dictionary containing the processed response.

        Raises:
            LionError: If there is an error processing the response.
        """
        # TODO: Implement response processing logic
        # 1. Extract the relevant information from the response
        # 2. Create a Message object or other appropriate structure
        # 3. Return the processed response
        raise NotImplementedError("This method is not yet implemented.")

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Union[Message, Dict[str, Any]]:
        """Send a chat request to the LLM service.

        This is a convenience method that combines prepare_request,
        call_llm, and process_response.

        Args:
            messages: The messages to include in the request.
            tools: Optional list of tools to make available.
            **kwargs: Additional parameters for the request.

        Returns:
            A Message object or a dictionary containing the processed response.

        Raises:
            LionNetworkError: If there is an error communicating with the service.
            LionError: If there is an error processing the response.
        """
        # TODO: Implement chat logic
        # 1. Prepare the request
        # 2. Call the LLM service
        # 3. Process the response
        # 4. Return the processed response
        raise NotImplementedError("This method is not yet implemented.")