from lionagi.service.connections.endpoint import Endpoint

from .._config import GeminiConfigs


@GeminiConfigs.CHAT.register
class GeminiChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        if config is None:
            from lionagi.config import settings

            kwargs.setdefault(
                "api_key", settings.GEMINI_API_KEY or "dummy-key-for-testing"
            )
            kwargs.setdefault("kwargs", {"model": "gemini-2.5-flash"})
        super().__init__(config, **kwargs)
