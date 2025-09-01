"""iModel: Model interface abstraction.

Ocean's v0 wisdom: Abstract interface for different AI providers.
Allows swapping between OpenAI, Anthropic, NVIDIA, etc.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional
from dataclasses import dataclass

from ..foundation.contracts import Observable
from ...wire.dto.base import ChatMessage, CompletionRequest, CompletionResponse


@dataclass
class ModelConfig:
    """Configuration for a model"""
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # Provider-specific config
    extra: dict[str, Any] = None


class iModel(ABC):
    """Abstract interface for AI models.
    
    Provides uniform interface regardless of provider.
    This is what allows us to swap between OpenAI, Claude, NIM, etc.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    @abstractmethod
    async def complete(self, 
                      messages: list[ChatMessage],
                      **kwargs) -> CompletionResponse:
        """Get completion from model"""
        pass
    
    @abstractmethod
    async def stream(self,
                    messages: list[ChatMessage],
                    **kwargs) -> AsyncIterator[str]:
        """Stream completion from model"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass
    
    @abstractmethod
    async def validate_request(self, request: CompletionRequest) -> bool:
        """Validate request before sending"""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get provider name (openai, anthropic, nvidia, etc)"""
        pass
    
    @property
    @abstractmethod  
    def supports_functions(self) -> bool:
        """Does this model support function calling?"""
        pass
    
    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        """Does this model support vision inputs?"""
        pass