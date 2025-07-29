# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from lionagi.utils import is_import_installed

HAS_CLAUDE_CODE_SDK = is_import_installed("claude_code_sdk")

# --------------------------------------------------------------------------- constants
ClaudePermission = Literal[
    "default",
    "acceptEdits",
    "bypassPermissions",
    "dangerously-skip-permissions",
]

CLAUDE_CODE_OPTION_PARAMS = {
    "allowed_tools",
    "max_thinking_tokens",
    "mcp_tools",
    "mcp_servers",
    "permission_mode",
    "continue_conversation",
    "resume",
    "max_turns",
    "disallowed_tools",
    "model",
    "permission_prompt_tool_name",
    "cwd",
    "system_prompt",
    "append_system_prompt",
}


# --------------------------------------------------------------------------- request model
class ClaudeCodeRequest(BaseModel):
    # -- conversational bits -------------------------------------------------
    prompt: str = Field(description="The prompt for Claude Code")
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    max_turns: int | None = None
    continue_conversation: bool = False
    resume: str | None = None

    # -- repo / workspace ----------------------------------------------------
    repo: Path = Field(default_factory=Path.cwd, exclude=True)
    ws: str | None = None  # sub-directory under repo
    add_dir: str | None = None  # extra read-only mount
    allowed_tools: list[str] | None = None

    # -- runtime & safety ----------------------------------------------------
    model: Literal["sonnet", "opus"] | str | None = "sonnet"
    max_thinking_tokens: int | None = None
    mcp_tools: list[str] = Field(default_factory=list)
    mcp_servers: dict[str, Any] = Field(default_factory=dict)
    mcp_config: str | Path | None = Field(None, exclude=True)
    permission_mode: ClaudePermission | None = None
    permission_prompt_tool_name: str | None = None
    disallowed_tools: list[str] = Field(default_factory=list)

    # -- internal use --------------------------------------------------------
    auto_finish: bool = Field(
        default=False,
        description="Automatically finish the conversation after the first response",
    )
    verbose_output: bool = Field(default=False)
    cli_display_theme: Literal["light", "dark"] = "dark"

    # ------------------------ validators & helpers --------------------------
    @field_validator("permission_mode", mode="before")
    def _norm_perm(cls, v):
        if v in {
            "dangerously-skip-permissions",
            "--dangerously-skip-permissions",
        }:
            return "bypassPermissions"
        return v

    # Workspace path derived from repo + ws
    def cwd(self) -> Path:
        if not self.ws:
            return self.repo

        # Convert to Path object for proper validation
        ws_path = Path(self.ws)

        # Check for absolute paths or directory traversal attempts
        if ws_path.is_absolute():
            raise ValueError(
                f"Workspace path must be relative, got absolute: {self.ws}"
            )

        if ".." in ws_path.parts:
            raise ValueError(
                f"Directory traversal detected in workspace path: {self.ws}"
            )

        # Resolve paths to handle symlinks and normalize
        repo_resolved = self.repo.resolve()
        result = (self.repo / ws_path).resolve()

        # Ensure the resolved path is within the repository bounds
        try:
            result.relative_to(repo_resolved)
        except ValueError:
            raise ValueError(
                f"Workspace path escapes repository bounds. "
                f"Repository: {repo_resolved}, Workspace: {result}"
            )

        return result

    @model_validator(mode="after")
    def _check_perm_workspace(self):
        if self.permission_mode == "bypassPermissions":
            # Use secure path validation with resolved paths
            repo_resolved = self.repo.resolve()
            cwd_resolved = self.cwd().resolve()

            # Check if cwd is within repo bounds using proper path methods
            try:
                cwd_resolved.relative_to(repo_resolved)
            except ValueError:
                raise ValueError(
                    f"With bypassPermissions, workspace must be within repository bounds. "
                    f"Repository: {repo_resolved}, Workspace: {cwd_resolved}"
                )
        return self

    # ------------------------ CLI helpers -----------------------------------
    def as_cmd_args(self) -> list[str]:
        """Build argument list for the *Node* `claude` CLI."""
        args: list[str] = ["-p", self.prompt, "--output-format", "stream-json"]
        if self.allowed_tools:
            args.append("--allowedTools")
            for tool in self.allowed_tools:
                args.append(f'"{tool}"')

        if self.disallowed_tools:
            args.append("--disallowedTools")
            for tool in self.disallowed_tools:
                args.append(f'"{tool}"')

        if self.resume:
            args += ["--resume", self.resume]
        elif self.continue_conversation:
            args.append("--continue")

        if self.max_turns:
            # +1 because CLI counts *pairs*
            args += ["--max-turns", str(self.max_turns + 1)]

        if self.permission_mode == "bypassPermissions":
            args += ["--dangerously-skip-permissions"]

        if self.add_dir:
            args += ["--add-dir", self.add_dir]

        if self.permission_prompt_tool_name:
            args += [
                "--permission-prompt-tool",
                self.permission_prompt_tool_name,
            ]

        if self.mcp_config:
            args += ["--mcp-config", str(self.mcp_config)]

        args += ["--model", self.model or "sonnet", "--verbose"]
        return args

    # ------------------------ SDK helpers -----------------------------------
    def as_claude_options(self):
        from claude_code_sdk import ClaudeCodeOptions

        data = {
            k: v
            for k, v in self.model_dump(exclude_none=True).items()
            if k in CLAUDE_CODE_OPTION_PARAMS
        }
        return ClaudeCodeOptions(**data)

    # ------------------------ convenience constructor -----------------------
    @classmethod
    def create(
        cls,
        messages: list[dict[str, Any]],
        resume: str | None = None,
        continue_conversation: bool | None = None,
        **kwargs,
    ):
        if not messages:
            raise ValueError("messages may not be empty")

        prompt = ""

        # 1. if resume or continue_conversation, use the last message
        if resume or continue_conversation:
            continue_conversation = True
            prompt = messages[-1]["content"]
            if isinstance(prompt, (dict, list)):
                prompt = json.dumps(prompt)

        # 2. else, use entire messages except system message
        else:
            prompts = []
            continue_conversation = False
            for message in messages:
                if message["role"] != "system":
                    content = message["content"]
                    prompts.append(
                        json.dumps(content)
                        if isinstance(content, (dict, list))
                        else content
                    )

            prompt = "\n".join(prompts)

        # 3. assemble the request data
        data: dict[str, Any] = dict(
            prompt=prompt,
            resume=resume,
            continue_conversation=bool(continue_conversation),
        )

        # 4. extract system prompt if available
        if (messages[0]["role"] == "system") and (
            resume or continue_conversation
        ):
            data["system_prompt"] = messages[0]["content"]
        if kwargs.get("append_system_prompt"):
            data["append_system_prompt"] = str(
                kwargs.get("append_system_prompt")
            )

        data.update(kwargs)
        return cls.model_validate(data, strict=False)
