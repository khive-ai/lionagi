# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from claude_code_sdk import ClaudeCodeOptions
from pydantic import BaseModel

from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.utils import to_dict


from .models import ClaudeCodeRequest


ENDPOINT_CONFIG = EndpointConfig(
    name="claude_code",
    provider="anthropic",
    base_url="internal",
    endpoint="query",
    api_key="dummy",
    request_options=ClaudeCodeRequest,
)

class ClaudeCodeEndpoint(Endpoint):

    def __init__(self, config=ENDPOINT_CONFIG, **kwargs):
        super().__init__(config=config, **kwargs)

    def create_payload(
        self,
        request: dict | BaseModel,
        **kwargs,
    ):
        request_dict = to_dict(request)
        # Merge stored kwargs from config, then request, then additional kwargs
        request_dict = {**self.config.kwargs, **request_dict, **kwargs}
        messages = request_dict.pop("messages", None)

        resume = request_dict.pop("resume", None)
        continue_conversation = request_dict.pop("continue_conversation", None)

        request_obj = ClaudeCodeRequest.create(
            messages=messages,
            resume=resume,
            continue_conversation=continue_conversation,
            **{k:v for k, v in request_dict.items() if v is not None and k in ClaudeCodeRequest.model_fields},
        )
        request_options = request_obj.as_claude_options()
        payload = {
            "prompt": request_obj.prompt,
            "options": request_options,
        }
        return (payload, {})

    def _stream_claude_code(self, prompt: str, options: ClaudeCodeOptions):
        from claude_code_sdk import query
        return query(prompt=prompt, options=options)

    async def stream(
        self,
        request: dict | BaseModel,
        **kwargs,
    ):
        async for chunk in self._stream_claude_code(**request, **kwargs):
            yield chunk

    def _parse_claude_code_response(self, responses: list) -> dict:
        """Parse Claude Code responses into a clean chat completions-like format.
        
        Claude Code returns a list of messages:
        - SystemMessage: initialization info
        - AssistantMessage(s): actual assistant responses with content blocks
        - UserMessage(s): for tool use interactions
        - ResultMessage: final result with metadata
        
        When Claude Code uses tools, the ResultMessage.result may be None.
        In that case, we need to look at the tool results in UserMessages.
        """
        result_message = None
        model = 'claude-code'
        assistant_text_content = []
        tool_results = []
        
        # Process all messages
        for response in responses:
            class_name = response.__class__.__name__
            
            if class_name == 'SystemMessage' and hasattr(response, 'data'):
                model = response.data.get('model', 'claude-code')
            
            elif class_name == 'AssistantMessage':
                # Extract text content from assistant messages
                if hasattr(response, 'content') and response.content:
                    for block in response.content:
                        if hasattr(block, 'text'):
                            assistant_text_content.append(block.text)
                        elif isinstance(block, dict) and 'text' in block:
                            assistant_text_content.append(block['text'])
            
            elif class_name == 'UserMessage':
                # Extract tool results from user messages
                if hasattr(response, 'content') and response.content:
                    for item in response.content:
                        if isinstance(item, dict) and item.get('type') == 'tool_result':
                            tool_results.append(item.get('content', ''))
            
            elif class_name == 'ResultMessage':
                result_message = response
        
        # Determine the final content
        final_content = ""
        if result_message and hasattr(result_message, 'result') and result_message.result:
            # Use ResultMessage.result if available
            final_content = result_message.result
        elif assistant_text_content:
            # Use assistant text content if available
            final_content = '\n'.join(assistant_text_content)
        elif tool_results:
            # If only tool results are available, use a generic summary
            # (Claude Code typically provides its own summary after tool use)
            final_content = "I've completed the requested task using the available tools."
        
        # Build the clean chat completions response
        result = {
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_content
                },
                "finish_reason": "stop" if not (result_message and hasattr(result_message, 'is_error') and result_message.is_error) else "error"
            }]
        }
        
        # Add usage information if available
        if result_message and hasattr(result_message, 'usage'):
            result['usage'] = result_message.usage

        
        # Add only essential Claude Code metadata
        if result_message:
            if hasattr(result_message, 'cost_usd'):
                result['usage']['cost_usd'] = result_message.cost_usd
            if hasattr(result_message, 'session_id'):
                result['session_id'] = result_message.session_id
            if hasattr(result_message, 'is_error'):
                result['is_error'] = result_message.is_error
            if hasattr(result_message, 'num_turns'):
                result['num_turns'] = result_message.num_turns
        
        return result

    async def _call(
        self,
        payload: dict,
        headers: dict,
        **kwargs,
    ):
        responses = []
        async for chunk in self._stream_claude_code(**payload):
            responses.append(chunk)
        
        # Parse the responses into a consistent format
        return self._parse_claude_code_response(responses)
            
