"""OpenAI dialect - including Azure and NVIDIA NIM compatibility.

Ocean: Connection to NVIDIA NIM team makes this critical.
"""

from typing import Any, Dict
from ..dto.base import CompletionRequest, CompletionResponse

CHAT_PATH = "chat/completions"  # relative to base_url


class OpenAIDialect:
    """OpenAI-compatible dialect.
    
    Works with:
    - OpenAI API
    - Azure OpenAI
    - NVIDIA NIM (Ocean's connections!)
    - Any OpenAI-compatible endpoint
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
    
    def adapt_request(self, request: CompletionRequest) -> Dict[str, Any]:
        """Convert generic request to OpenAI format with full parameter support"""
        data: Dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            data["max_tokens"] = request.max_tokens
        if request.stream:
            data["stream"] = True
        if hasattr(request, 'top_p') and request.top_p is not None:
            data["top_p"] = request.top_p
        if hasattr(request, 'frequency_penalty') and request.frequency_penalty is not None:
            data["frequency_penalty"] = request.frequency_penalty
        if hasattr(request, 'presence_penalty') and request.presence_penalty is not None:
            data["presence_penalty"] = request.presence_penalty
        if hasattr(request, 'stop') and request.stop:
            data["stop"] = request.stop
        return data
    
    def adapt_response(self, response: Dict[str, Any]) -> CompletionResponse:
        """Convert OpenAI response to generic format"""
        return CompletionResponse(
            id=response.get("id", ""),
            model=response.get("model", ""),
            choices=response.get("choices", []),
            usage=response.get("usage", {}),
            created=response.get("created", 0),
        )
    
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for OpenAI-compatible APIs"""
        return {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
        }
