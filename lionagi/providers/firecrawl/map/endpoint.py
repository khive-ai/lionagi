from pydantic import BaseModel

from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig

from .._config import FirecrawlConfigs


@FirecrawlConfigs.MAP.register
class FirecrawlMapEndpoint(Endpoint):
    def __init__(self, config: EndpointConfig = None, **kwargs):
        if config is None:
            from lionagi.config import settings

            kwargs.setdefault(
                "api_key", settings.FIRECRAWL_API_KEY or "dummy-key-for-testing"
            )
            kwargs.setdefault("timeout", 120)
            kwargs.setdefault("max_retries", 3)
        super().__init__(config=config, **kwargs)

    def create_payload(
        self,
        request: dict | BaseModel,
        extra_headers: dict | None = None,
        **kwargs,
    ):
        payload, headers = super().create_payload(request, extra_headers, **kwargs)
        # Re-serialize with camelCase aliases for the Firecrawl API.
        if self.config.request_options is not None:
            model_cls = self.config.request_options
            try:
                obj = model_cls.model_validate(payload)
                payload = obj.model_dump(by_alias=True, exclude_none=True)
            except Exception:
                pass
        return payload, headers
