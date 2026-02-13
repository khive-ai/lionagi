# CLI Agent Providers

lionagi integrates with CLI-based AI agents -- Claude Code, Gemini CLI, and
OpenAI Codex -- as iModel providers. This enables agent-to-agent orchestration:
your outer agent uses lionagi to spawn and coordinate inner agents.

## Supported CLI Providers

| Provider String | CLI Tool | Endpoint Class |
|----------------|----------|---------------|
| `claude_code` | Claude Code | `ClaudeCodeCLIEndpoint` |
| `gemini_code` | Gemini CLI | `GeminiCLIEndpoint` |
| `codex` | OpenAI Codex | `CodexCLIEndpoint` |

All CLI providers use `endpoint="query_cli"` and require no real API key
(`api_key` is ignored; the CLI tool handles auth).

## Quick Setup

### Claude Code

```python
from lionagi import Branch, iModel

cc_model = iModel(
    provider="claude_code",
    model="sonnet",                    # model hint for Claude Code
    cwd="/path/to/project",            # working directory for the agent
    permission_mode="bypassPermissions",  # or "default"
    allowed_tools=["Read", "Grep", "Glob"],
    verbose_output=True,
)

branch = Branch(chat_model=cc_model, name="claude_agent")
response = await branch.communicate("Analyze the codebase structure")
```

### Gemini CLI

```python
gemini_model = iModel(
    provider="gemini_code",
    model="gemini-2.5-pro",
    cwd="/path/to/project",
)

branch = Branch(chat_model=gemini_model, name="gemini_agent")
response = await branch.communicate("Review this module for issues")
```

### Codex

```python
codex_model = iModel(
    provider="codex",
    model="codex-mini",
    cwd="/path/to/project",
)

branch = Branch(chat_model=codex_model, name="codex_agent")
response = await branch.communicate("Refactor the authentication module")
```

## iModel Configuration Reference

### Common Parameters (all CLI providers)

```python
iModel(
    provider="claude_code",          # Required: provider string
    model="sonnet",                  # Model selection hint
    cwd="/path/to/project",          # Working directory for the agent
    verbose_output=True,             # Show agent output during execution
    cli_include_summary=False,       # Include cost/usage summary
    auto_finish=False,               # Auto-complete if agent doesn't finish
)
```

### Claude Code-Specific Parameters

```python
iModel(
    provider="claude_code",
    model="sonnet",
    permission_mode="bypassPermissions",  # "default" | "acceptEdits" | "bypassPermissions"
    allowed_tools=["Read", "Grep", "Glob", "Bash"],
    disallowed_tools=["Write"],           # Block specific tools
    cli_display_theme="dark",             # "dark" | "light"
    max_turns=10,                         # Max conversation turns
    max_thinking_tokens=16000,            # Thinking budget
    continue_conversation=True,           # Continue previous session
    system_prompt="You are a security auditor.",
    append_system_prompt="Focus on OWASP Top 10.",  # Appended to existing system
    mcp_tools=["browser", "filesystem"],  # MCP tool names to enable
    mcp_config="/path/to/.mcp.json",      # MCP config file
    add_dir="/shared/reference",          # Extra read-only directory
)
```

### Gemini CLI-Specific Parameters

```python
iModel(
    provider="gemini_code",
    model="gemini-2.5-pro",
    sandbox=True,                         # Safety sandboxing (default True)
    approval_mode="auto_edit",            # "suggest" | "auto_edit" | "full_auto"
    # yolo=True,                          # Auto-approve all -- emits safety warning
    debug=False,
    include_directories=["/extra/src"],   # Additional directories to include
    system_prompt="You are a code analyst.",
)
```

### Codex CLI-Specific Parameters

```python
iModel(
    provider="codex",
    model="gpt-5.3-codex",
    full_auto=True,                       # Auto-approve with workspace-write sandbox
    sandbox="workspace-write",            # "read-only" | "workspace-write" | "danger-full-access"
    # bypass_approvals=True,              # Skip ALL approvals -- use with caution
    skip_git_repo_check=False,
    output_schema="/path/to/schema.json", # JSON Schema for structured output
    include_plan_tool=True,               # Enable planning tool
    images=["screenshot.png"],            # Attach images
    config_overrides={"key": "value"},    # Custom config -c flags
    system_prompt="You are a test engineer.",
)
```

## Context Management

**CLI providers manage their own context.** Unlike API providers where Branch's
MessageManager controls what the LLM sees via `progression=`, CLI agents
(Claude Code, Gemini CLI, Codex) maintain their own conversation history
internally through session resume. This is a fundamental architectural
difference:

| Aspect | API Providers | CLI Providers |
|--------|--------------|---------------|
| Context owner | Branch (MessageManager + Progression) | The CLI agent itself |
| History mechanism | Messages stored in Pile, windowed via `progression=` | Session resume via `--resume` flag |
| Compression | Narrow the Progression (sliding window) | Agent handles its own context management |
| State across calls | Controlled by Branch | Controlled by session_id |

**Do not use `progression=` windowing with CLI providers** -- the agent already
manages its own conversation context. Branch still records messages for logging
and inspection, but the CLI agent's session is the source of truth for what
context it sees.

### Session Lifecycle

CLI session state is managed automatically via `session_id`:

1. First call creates a new session. The CLI returns a `session_id` in the
   `system` event.
2. lionagi stores the `session_id` on the endpoint.
3. Subsequent calls on the same iModel pass `--resume` with that ID.
4. If the resumed session gets a new ID (the CLI may reassign), lionagi
   updates to the new ID automatically.
5. `Branch.clone()` creates a fresh copy with no shared session state.
6. `iModel.copy(share_session=True)` carries over the session ID.

```python
# Fresh copy, independent session
new_model = cc_model.copy()

# Copy that resumes the same CLI session
resumed_model = cc_model.copy(share_session=True)
```

## Multi-Agent Orchestration

### Workspace Isolation

Give each agent its own working directory to prevent file conflicts:

```python
def create_agent(role: str, subdir: str) -> iModel:
    return iModel(
        provider="claude_code",
        model="sonnet",
        cwd=f"/project/.agents/{subdir}",
        permission_mode="bypassPermissions",
        allowed_tools=["Read", "Grep", "Glob", "Bash"],
    )

orchestrator = create_agent("orchestrator", "orc")
researcher_1 = create_agent("researcher", "res1")
researcher_2 = create_agent("researcher", "res2")
```

### Fan-Out Pattern

```python
from lionagi import Branch, Session, Builder
from lionagi.operations.fields import Instruct

session = Session()
builder = Builder("research")

# Orchestrator generates sub-tasks
orc_branch = Branch(chat_model=orchestrator, name="orchestrator")
session.include_branches([orc_branch])

root = builder.add_operation(
    "operate",
    branch=orc_branch,
    instruction="Break this task into 3 research sub-tasks",
    response_format=TaskList,  # your Pydantic model
)

result = await session.flow(builder.get_graph())

# Fan out to researcher agents
for i, task in enumerate(result["operation_results"][root].tasks):
    agent = create_agent("researcher", f"res_{i}")
    res_branch = Branch(chat_model=agent, name=f"researcher_{i}")
    session.include_branches([res_branch])

    builder.add_operation(
        "communicate",
        branch=res_branch,
        instruction=task.description,
        depends_on=[root],
    )

# Execute all research in parallel
final = await session.flow(builder.get_graph(), max_concurrent=3)
```

## Cost Tracking

CLI providers include cost data in model responses:

```python
from lionagi.protocols.messages import AssistantResponse

# After a communicate/operate call, inspect the last message
last_msg = branch.messages[-1]
if isinstance(last_msg, AssistantResponse):
    cost = last_msg.model_response.get("total_cost_usd", 0)
    result_text = last_msg.model_response.get("result", "")
    summary = last_msg.model_response.get("summary", "")
```

## Event Handlers (Claude Code only)

Claude Code supports streaming event handlers for fine-grained control:

```python
cc_model = iModel(provider="claude_code", model="sonnet")

# Access the endpoint to set handlers
cc_model.endpoint.update_handlers(
    on_text=lambda chunk: print(f"Text: {chunk.text}"),
    on_tool_use=lambda chunk: print(f"Tool: {chunk}"),
    on_thinking=lambda chunk: print(f"Thinking: {chunk.text}"),
    on_final=lambda session: print(f"Done: {session.result}"),
)
```

Available handler keys by provider:

| Provider | Handlers |
|----------|----------|
| `claude_code` | `on_thinking`, `on_text`, `on_tool_use`, `on_tool_result`, `on_system`, `on_final` |
| `gemini_code` | `on_text`, `on_tool_use`, `on_tool_result`, `on_final` |
| `codex` | `on_text`, `on_tool_use`, `on_tool_result`, `on_final` |

## Async Context Manager

Both `Branch` and `iModel` support async context managers:

```python
async with iModel(provider="claude_code", model="sonnet") as model:
    async with Branch(chat_model=model) as branch:
        result = await branch.communicate("Analyze this code")
    # Branch dumps logs on exit

# iModel executor stops on exit
```

## Error Handling

```python
try:
    result = await branch.communicate("Complex analysis task")
except ValueError as e:
    if "Failed to invoke API call" in str(e):
        # CLI tool likely timed out or crashed
        # Default timeout is 18000 seconds (5 hours)
        pass
```

## Key Differences: CLI vs API Providers

| Aspect | API Providers | CLI Providers |
|--------|--------------|---------------|
| Auth | API key required | CLI tool handles auth |
| Context owner | Branch (MessageManager + Progression) | The CLI agent itself (session resume) |
| Latency | Low (HTTP) | Higher (subprocess) |
| Session state | Stateless | Persistent session_id (auto-updated) |
| Concurrency | High (100 queue capacity) | Low (default 3) |
| Clone behavior | Shared endpoint | Fresh endpoint copy |
| Timeout | Provider default | 18000s (5 hours) |
| Tool calling | Via function schemas | Via CLI tool's built-in tools |
| `progression=` | Controls LLM context window | Not applicable (agent manages own context) |
