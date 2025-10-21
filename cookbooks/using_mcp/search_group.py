import logging
import os
from typing import Any, Literal

from dotenv import load_dotenv
from khivemcp import ServiceGroup, operation

from lionagi import iModel
from lionagi.service.connections.api_calling import APICalling
from lionagi.service.third_party.exa_models import ExaSearchRequest
from lionagi.service.third_party.pplx_models import PerplexityChatRequest

logger = logging.getLogger(__name__)

PROVIDERS = Literal["exa", "perplexity"]


SearchRequestSchema = ExaSearchRequest | PerplexityChatRequest


class SearchServiceGroup(ServiceGroup):
    def __init__(self, config: dict | None = None):
        """
        Initialize the LLMServiceGroup.
        """
        load_dotenv()
        super().__init__(config=config)
        self._pplx_available = bool(os.getenv("PERPLEXITY_API_KEY", False))
        self._exa_available = bool(os.getenv("EXA_API_KEY", False))
        logger.info("[LLMServiceGroup] Initialized.")
        self.imodels = {}

    @operation(
        name="search",
        schema=SearchRequestSchema,
        accepts_context=True,
        parallelizable=True,
    )
    async def search(
        self, ctx, request: ExaSearchRequest | PerplexityChatRequest
    ):
        """Execute search queries using various backends (Perplexity, Exa). Supports parallel batch execution.

        Context provides access to:
        - ctx.request_id: Unique request identifier
        - ctx.meta: Request metadata
        - ctx.access_token: Auth token (if auth enabled)
        """
        # Example: Log request metadata
        logger.info(
            f"Search request from context: {getattr(ctx, 'request_id', 'unknown')}"
        )

        # Type-based routing: check which schema was validated
        if isinstance(request, ExaSearchRequest):
            return await self._exa_search_impl(request)
        elif isinstance(request, PerplexityChatRequest):
            return await self._perplexity_search_impl(request)
        else:
            raise ValueError(f"Unexpected request type: {type(request)}")

    async def _exa_search_impl(self, request: ExaSearchRequest):
        """Internal implementation for Exa search."""
        if not self._exa_available:
            raise RuntimeError("Exa API key not configured")

        if "exa_search" not in self.imodels:
            self.imodels["exa_search"] = iModel(
                provider="exa",
                endpoint="search",
                queue_capacity=5,
                interval=1,
                limit_requests=5,
                api_key="EXA_API_KEY",
            )

        result: APICalling = await self.imodels["exa_search"].invoke(
            **request.model_dump(exclude_none=True),
            is_cached=True,
        )
        return {
            "id": str(result.id),
            "created_at": result.created_datetime.isoformat(),
            "status": result.execution.status.value,
            "duration": result.execution.duration,
            "response": result.execution.response,
            "error": result.execution.error,
        }

    async def _perplexity_search_impl(self, request: PerplexityChatRequest):
        """Internal implementation for Perplexity search."""
        if not self._pplx_available:
            raise RuntimeError("Perplexity API key not configured")

        if "perplexity_search" not in self.imodels:
            self.imodels["perplexity_search"] = iModel(
                provider="perplexity",
                endpoint="chat",
                interval=1,  # 1 second window
                limit_tokens=100000,  # Higher token limit
                api_key="PERPLEXITY_API_KEY",
                limit_requests=100,  # Allow many concurrent requests
            )

        imodel = self.imodels["perplexity_search"]
        result: APICalling = await imodel.invoke(
            **request.model_dump(exclude_none=True),
            is_cached=True,
        )
        return {
            "id": str(result.id),
            "created_at": result.created_datetime.isoformat(),
            "status": result.execution.status.value,
            "duration": result.execution.duration,
            "response": result.execution.response,
            "error": result.execution.error,
        }
