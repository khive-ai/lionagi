import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

from claude_code_sdk import ClaudeCodeOptions, PermissionMode

class ClaudeCodeRequest(BaseModel):
    prompt: str = Field(description="The prompt for Claude Code")
    allowed_tools: list[str] = Field(default_factory=list, description="List of allowed tools")
    max_thinking_tokens: int = 8000
    mcp_tools: list[str] = list
    mcp_servers: dict[str, Any] = Field(default_factory=dict)
    permission_mode: PermissionMode | None = None
    continue_conversation: bool = False
    resume: str | None = None
    max_turns: int | None = None
    disallowed_tools: list[str] = Field(default_factory=list)
    model: str | None = None
    permission_prompt_tool_name: str | None = None
    cwd: str | Path | None = None
    system_prompt: str | None = None
    append_system_prompt: str | None = None

    def as_claude_options(self) -> ClaudeCodeOptions:
        dict_ = self.model_dump(exclude_unset=True)
        dict_.pop("prompt")
        return ClaudeCodeOptions(**dict_)
    
    @classmethod
    def create(
        cls,
        messages: list[dict],
        resume: str | None = None, 
        continue_conversation: bool = None,
        **kwargs
    ):
        prompt = messages[-1]["content"]
        if isinstance(prompt, dict | list):
            prompt = json.dumps(prompt)
        
        # If resume is provided, set continue_conversation to True
        if resume is not None and continue_conversation is None:
            continue_conversation = True
        
        dict_ = dict(
            prompt=prompt,
            continue_conversation=continue_conversation,
            resume=resume,
        )

        if resume is not None or continue_conversation is not None:
            if messages[0]["role"] == "system":
                dict_["system_prompt"] = messages[0]["content"]
        
        if (a := kwargs.get("system_prompt")) is not None:
            dict_["append_system_prompt"] = a

        if (a := kwargs.get("append_system_prompt")) is not None:
            dict_.setdefault("append_system_prompt", "")
            dict_["append_system_prompt"] += str(a)

        dict_ = {**dict_, **kwargs}
        dict_ = {k:v for k, v in dict_.items() if v is not None}
        return cls(**dict_)
