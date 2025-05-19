import json # stdlib json
from collections.abc import Callable
from functools import wraps
from typing import Any

# Updated imports
from lionagi.core.async_core import AsyncAdapter # Assuming this path is correct
from lionagi.core_defs import Embedable, Embedding, StructuredLog
from lionagi.core_bases.invokable import Invokable
from lionagi.core_utils import as_async_fn, validate_model_to_dict, sha256_of_dict


class Event(Invokable, Embedable): # Inherits from moved Invokable and Embedable (now in core_defs)
    event_type: str | None = None # Made event_type optional to match __init__

    def __init__(
        self,
        event_invoke_function: Callable,
        event_invoke_args: list[Any],
        event_invoke_kwargs: dict[str, Any],
        event_type: str | None = None, # event_type can be None
    ):
        # super().__init__(response_obj=None) # Invokable's __init__ doesn't take response_obj
        # Invokable inherits Temporal, which has its own __init__ from BaseModel
        # Temporal.__init__ will be called by BaseModel.__init__
        # Embedable.__init__ will be called by BaseModel.__init__
        # Need to call super() for Pydantic model initialization
        super().__init__() # Call BaseModel's init via Temporal/Embedable
        
        self._invoke_function = event_invoke_function
        self._invoke_args = event_invoke_args or []
        self._invoke_kwargs = event_invoke_kwargs or {}
        if event_type is not None:
            self.event_type = event_type
        # Ensure 'id' is set if not already by Temporal's default_factory
        # This should be handled by Pydantic's BaseModel initialization process
        # via Temporal's id field.

    def create_content(self):
        if self.content is not None:
            return self.content

        event = {"request": self.request, "response": self.execution.response}
        # Ensure self.content is a string for Embedable
        self.content = json.dumps(event, default=str, ensure_ascii=False)
        return self.content

    def to_log(self, event_type: str | None = None, hash_content: bool = False) -> StructuredLog:
        if self.content is None:
            self.create_content()

        # Create a StructuredLog object
        log_entry = StructuredLog( # Changed Log to StructuredLog
            id=str(self.id), # Ensure id is string if StructuredLog expects string
            created_at=self.created_at.isoformat(),
            updated_at=self.updated_at.isoformat(),
            event_type=event_type or self.__class__.__name__,
            content=self.content, # content is already a string
            embedding=self.embedding,
            duration=self.execution.duration,
            status=self.execution.status.value.lower(),
            error=self.execution.error,
            # sha256 will be set below if hash_content is True
        )

        if hash_content:
            # sha256_of_dict is already imported from lionagi.core_utils
            log_entry.sha256 = sha256_of_dict({"content": self.content})

        return log_entry


def as_event(
    *,
    request_arg: str | None = None,
    embed_content: bool = False,
    embed_function: Callable[..., Embedding] | None = None,
    adapt: bool = False,
    adapter: type[AsyncAdapter] | None = None,
    event_type: str | None = None,
    **kw,
):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Event: # Returns the new Event class
            request_obj = kwargs.get(request_arg) if request_arg else None
            
            # This logic for stripping self/cls seems fragile.
            # Assuming args[0] is the main request object if not specified by request_arg
            # This might need review based on actual usage of the decorator.
            effective_args = args
            if len(args) > 0 and hasattr(args[0], "__class__") and not request_arg:
                 # Heuristic: if first arg is an instance and request_arg is not set,
                 # it might be 'self' or 'cls' if func is a method.
                 # However, this decorator is more likely for standalone functions.
                 # For simplicity, let's assume args are direct inputs to func.
                 pass


            if request_obj is None and effective_args:
                request_obj = effective_args[0]
            
            event = Event(func, list(effective_args), kwargs, event_type=event_type)
            event.request = validate_model_to_dict(request_obj) # request_obj could be None
            
            await event.invoke() # Populates event.execution and event.response_obj
            
            if event.content is None: # Ensure content is created for embedding
                event.create_content()

            if embed_content and embed_function is not None and event.content is not None:
                async_embed = as_async_fn(embed_function)
                event.embedding = await async_embed(event.content)

            if adapt and adapter is not None:
                await adapter.to_obj(event.to_log(event_type=event.event_type), **kw) # Pass event.event_type

            return event

        return wrapper

    return decorator