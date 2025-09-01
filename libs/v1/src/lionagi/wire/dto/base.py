"""DTOs using msgspec for wire protocol speed.

msgspec for serialization - 5-10x faster than JSON.
"""

import msgspec
from typing import Any, Optional, List, Dict
from uuid import UUID


class Request(msgspec.Struct):
    """Base request DTO"""
    id: UUID
    method: str
    params: Dict[str, Any] = msgspec.field(default_factory=dict)


class Response(msgspec.Struct):
    """Base response DTO"""
    id: UUID
    result: Any = None
    error: Optional[str] = None


class ChatMessage(msgspec.Struct):
    """Chat message DTO"""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class CompletionRequest(msgspec.Struct):
    """LLM completion request"""
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    
    # Optional parameters
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None


class CompletionResponse(msgspec.Struct):
    """LLM completion response"""
    id: str
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    created: int