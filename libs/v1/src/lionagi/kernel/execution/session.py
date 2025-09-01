"""Session: High-level conversation management.

The main interface developers use for AI interactions.
"""

from typing import Any, Optional, List
from uuid import UUID, uuid4

from ..foundation.contracts import Observable, Observation
from ..safety.ipu import IPU
from ...domain.generic.pile import Pile
from ...domain.generic.event import Event
from ...wire.transport.http import HTTPTransport
from ...wire.dialects.openai import OpenAIDialect
from ...wire.dto.base import ChatMessage, CompletionRequest


class Session(Observable):
    """Managed conversation with full observation chain.
    
    Everything goes through IPU validation and capability checking.
    """
    
    def __init__(self, 
                 model: str = "gpt-4o-mini",
                 dialect: Any = None,
                 ipu: IPU = None):
        self._id = uuid4()
        self.model = model
        self.dialect = dialect or OpenAIDialect()
        self.ipu = ipu or IPU()
        self.messages = Pile()  # Observable collection of ChatMessage
        self.events = Pile()
        self.transport = HTTPTransport(base_url=self.dialect.base_url)
    
    @property
    def id(self) -> UUID:
        return self._id
    
    async def chat(self, message: str, **kwargs) -> str:
        """High-level chat interface using dialect + HTTP transport."""
        # Create event
        event = Event(
            type="chat_request",
            source=self.id,
            data={"message": message, **kwargs}
        )
        self.events.include(event)
        
        # IPU observes and validates
        observation = await self.ipu.observe(event)
        
        if not observation.valid:
            raise ValueError(f"Request denied: {observation.reason}")
        
        # Append user message to history
        user_msg = ChatMessage(role="user", content=message)
        self.messages.include(user_msg)

        # Build request
        req = CompletionRequest(
            model=self.model,
            messages=list(self.messages),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens"),
            stream=False,
        )
        payload = self.dialect.adapt_request(req)
        headers = self.dialect.get_headers()

        # Call provider
        raw = await self.transport.post_json("chat/completions", payload, headers=headers)
        adapted = self.dialect.adapt_response(raw)

        # Extract first assistant message if present
        response = ""
        try:
            response = adapted.choices[0]["message"]["content"]
        except Exception:
            try:
                response = adapted.choices[0].get("text", "")
            except Exception:
                response = ""
        
        # Create response event
        response_event = Event(
            type="chat_response",
            target=self.id,
            data={"response": response},
            causality=(event.id,)
        )
        self.events.include(response_event)
        
        return response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.transport.close()
