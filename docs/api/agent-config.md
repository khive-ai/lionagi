# `AgentConfig` and `create_agent()`

```python
from lionagi.agent import AgentConfig, create_agent, PermissionPolicy
from lionagi.agent.hooks import guard_destructive, guard_paths, log_tool_use
```

`AgentConfig` captures what a coding agent needs — model, tools, hooks, permissions, system
prompt — in a single serializable object. `create_agent()` wires it into a ready-to-use `Branch`.

---

## `AgentConfig`

```python
@dataclass
class AgentConfig
```

Source: `lionagi/agent/config.py`

### Fields

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `name` | `str` | `"agent"` | Human label for the agent |
| `model` | `str \| None` | `None` | Model spec: `"provider/model"` or bare alias |
| `effort` | `str \| None` | `None` | Override effort level (e.g. `"high"`, `"xhigh"`) |
| `system_prompt` | `str` | `""` | System prompt text |
| `tools` | `list[str]` | `[]` | Tool presets to register: `"coding"`, `"reader"`, `"editor"`, `"bash"`, `"search"` |
| `hook_handlers` | `dict[str, list[Callable]]` | `{}` | Phase-keyed hooks (`"pre:bash"`, `"post:*"`, `"error:editor"`) |
| `permissions` | `dict \| PermissionPolicy` | `{}` | Permission rules; see `PermissionPolicy` |
| `mcp_servers` | `list[str] \| None` | `None` | MCP server names to load from `.mcp.json` |
| `mcp_config_path` | `str \| None` | `None` | Explicit path to `.mcp.json` (overrides auto-discovery) |
| `max_extensions` | `int` | `20` | Max tool-use rounds per agent invocation |
| `yolo` | `bool` | `False` | Auto-approve all tool calls (pass-through to provider kwargs) |
| `lion_system` | `bool` | `True` | Prepend lionagi system preamble to `system_prompt` |
| `cwd` | `str \| None` | `None` | Working directory for tools and MCP discovery |
| `extra` | `dict` | `{}` | Additional YAML fields preserved on round-trip |

### Hook methods

```python
config.pre("bash", handler)       # register a pre-hook for the bash tool
config.post("editor", handler)    # register a post-hook for the editor tool
config.on_error("*", handler)     # register an error hook for all tools
```

- `pre` hooks: `async (tool_name: str, action: str, args: dict) -> dict | None`
  Return a modified `args` dict to rewrite the call, or raise `PermissionError` to block.
- `post` hooks: `async (tool_name: str, action: str, args: dict, result: dict) -> dict | None`
  Return a modified `result` dict, or `None` to pass through unchanged.
- Tool name `"*"` matches all tools.

### Preset methods

#### `AgentConfig.coding()`

```python
@classmethod
def coding(
    cls,
    name: str = "coder",
    model: str | None = None,
    effort: str | None = "high",
    system_prompt: str | None = None,
    cwd: str | None = None,
    **kwargs,
) -> AgentConfig
```

Preset for a coding agent. Registers `tools=["coding"]` (reader, editor, bash, search, context,
subagent) and uses the built-in coding system prompt when `system_prompt` is not provided.

```python
config = AgentConfig.coding(model="openai/gpt-4.1", cwd="/Users/me/project")
```

#### `AgentConfig.from_yaml()`

```python
@classmethod
def from_yaml(cls, path: str | Path) -> AgentConfig
```

Load config from a YAML file. Hook callables are code-only and are not serialized.

```yaml
# example .lionagi/agents/coder/coder.yaml
name: coder
model: openai/gpt-4.1
effort: high
tools: [coding]
system_prompt: |
  You are a coding agent...
permissions:
  mode: rules
  allow:
    reader: ["*"]
    search: ["*"]
    bash: ["git *", "cargo *", "uv *"]
  deny:
    bash: ["rm -rf *", "sudo *"]
```

#### `AgentConfig.to_yaml()`

```python
def to_yaml(self, path: str | Path) -> None
```

Save config fields to YAML. `hook_handlers` (callables) are omitted.

---

## `create_agent()`

```python
async def create_agent(
    config: AgentConfig,
    *,
    load_settings: bool = True,
    project_dir: str | None = None,
    trust_project_settings: bool = False,
    trusted_hook_modules: set[str] | frozenset[str] | None = None,
) -> Branch
```

Source: `lionagi/agent/factory.py`

Creates a fully configured `Branch` from an `AgentConfig`. Wires: settings → hooks →
system prompt → model → tools → MCP.

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `config` | `AgentConfig` | — | Agent configuration |
| `load_settings` | `bool` | `True` | Load hooks from `~/.lionagi/settings.yaml` |
| `project_dir` | `str \| None` | `None` | Project root for settings resolution; auto-detected if `None` |
| `trust_project_settings` | `bool` | `False` | Also load `.lionagi/settings.yaml` from the project dir |
| `trusted_hook_modules` | `set[str] \| None` | `None` | Python modules allowed for import-based hooks; defaults to `{"lionagi.agent.hooks"}` |

Returns a `Branch` ready for use with all tools registered and hooks attached.

```python
config = AgentConfig.coding(model="openai/gpt-4.1")
branch = await create_agent(config)
response = await branch.chat("Refactor the auth module")
```

**Settings loading order** (project-local wins):
1. `~/.lionagi/settings.yaml` — always loaded when `load_settings=True`
2. `.lionagi/settings.yaml` — loaded only when `trust_project_settings=True`

---

## `PermissionPolicy`

```python
@dataclass
class PermissionPolicy
```

Source: `lionagi/agent/permissions.py`

Per-tool allow/deny/escalate rules evaluated before each tool call. Three modes:

| Mode | Behavior |
|------|----------|
| `"allow_all"` | All tool calls permitted (default) |
| `"deny_all"` | All tool calls blocked |
| `"rules"` | Check deny → allow → escalate lists; default deny if no rule matches |

### Fields

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `mode` | `str` | `"allow_all"` | `"allow_all"` \| `"deny_all"` \| `"rules"` |
| `allow` | `dict[str, list[str]]` | `{}` | Tool → list of fnmatch patterns that permit the call |
| `deny` | `dict[str, list[str]]` | `{}` | Tool → list of fnmatch patterns that block the call |
| `escalate` | `dict[str, list[str]]` | `{}` | Tool → list of patterns that trigger `on_escalate` |
| `on_escalate` | `Callable \| None` | `None` | Async callable invoked on escalation; return `True` to allow, a `dict` to rewrite args |

Tool names in `allow`/`deny`/`escalate` are normalized: `"bash_tool"` → `"bash"`, etc.
`"*"` as a tool key applies to all tools.

### Preset class methods

```python
PermissionPolicy.allow_all()   # mode="allow_all"
PermissionPolicy.deny_all()    # mode="deny_all"

# reader + search allowed; editor + bash denied
PermissionPolicy.read_only()

# reader + editor + search allowed; dangerous bash commands denied; other bash → escalate
PermissionPolicy.safe()
```

### `from_dict()`

```python
@classmethod
def from_dict(cls, data: dict) -> PermissionPolicy
```

Build from a plain dict (e.g. loaded from YAML):

```python
policy = PermissionPolicy.from_dict({
    "mode": "rules",
    "allow": {"reader": ["*"], "bash": ["git *", "uv *"]},
    "deny": {"bash": ["rm *", "sudo *"]},
})
```

### Pattern matching

For the `bash` tool, patterns are matched against the command string.
For `editor` and `reader`, patterns are matched against the file path.
Shell control operators (`;`, `&&`, `||`, `|`, backticks, `$()`, redirects) in bash commands
are blocked unconditionally before pattern matching — they cannot be allow-listed.

### Using with `AgentConfig`

```python
# Dict form (round-trips through YAML)
config.permissions = {
    "mode": "rules",
    "allow": {"reader": ["*"], "bash": ["git *"]},
    "deny": {"bash": ["rm *"]},
}

# Object form (code-only)
config.permissions = PermissionPolicy.safe()
```

---

## Built-in hooks

Source: `lionagi/agent/hooks.py`

### `guard_destructive`

```python
async def guard_destructive(tool_name: str, action: str, args: dict) -> dict | None
```

Pre-hook for `bash`. Raises `PermissionError` when the command matches a destructive pattern:
`rm -rf`, `git push --force`, `git reset --hard`, `git clean -fd`, `DROP TABLE`,
`DROP DATABASE`, `TRUNCATE TABLE`, `mkfs`, `dd if=`, writes to `/dev/sd*`.

```python
config.pre("bash", guard_destructive)
```

### `guard_paths()`

```python
def guard_paths(
    allowed_paths: list[str] | None = None,
    denied_paths: list[str] | None = None,
) -> Callable
```

Factory that returns a pre-hook restricting file access by path. Applied to `reader` and `editor`.

- `allowed_paths`: if set, any path outside these roots raises `PermissionError`.
- `denied_paths`: patterns (absolute paths, filenames, or substrings) that are always blocked.

```python
config.pre("reader", guard_paths(allowed_paths=["/Users/me/project/"]))
config.pre("editor", guard_paths(denied_paths=[".env", "*.key"]))
```

### `log_tool_use`

```python
async def log_tool_use(tool_name: str, action: str, args: dict, result: dict) -> dict | None
```

Post-hook for any tool. Logs `tool=<name> action=<action> success=<bool>` at `INFO` level
via the standard `logging` module. Returns `None` (does not modify result).

```python
config.post("*", log_tool_use)
```

### `auto_format_python`

```python
async def auto_format_python(tool_name: str, action: str, args: dict, result: dict) -> dict | None
```

Post-hook for `editor`. Runs `ruff format <file_path>` on successfully edited `.py` files.

```python
config.post("editor", auto_format_python)
```

---

Next: [`SandboxSession`](sandbox.md) — isolated worktree execution
