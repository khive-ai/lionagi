from lionagi.service.connections.endpoint import Endpoint

from .._config import GroqConfigs


@GroqConfigs.CHAT.register
class GroqChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        if config is None:
            from lionagi.config import settings

            kwargs.setdefault(
                "api_key", settings.GROQ_API_KEY or "dummy-key-for-testing"
            )
            kwargs.setdefault("kwargs", {"model": "llama-3.3-70b-versatile"})
        super().__init__(config, **kwargs)