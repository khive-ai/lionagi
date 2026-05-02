from lionagi.service.connections.endpoint import Endpoint

from .._config import NvidiaNimConfigs


@NvidiaNimConfigs.CHAT.register
class NvidiaNimChatEndpoint(Endpoint):
    """NVIDIA NIM chat completion endpoint.

    Get your API key from: https://build.nvidia.com/
    API Documentation: https://docs.nvidia.com/nim/
    """

    def __init__(self, config=None, **kwargs):
        if config is None:
            from lionagi.config import settings

            kwargs.setdefault(
                "api_key", settings.NVIDIA_NIM_API_KEY or "dummy-key-for-testing"
            )
            kwargs.setdefault("kwargs", {"model": "meta/llama3-8b-instruct"})
            kwargs.setdefault("requires_tokens", True)
        super().__init__(config, **kwargs)
