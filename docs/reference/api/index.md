# API Reference

Complete reference for the lionagi public API. All classes documented here are
importable from the top-level package:

```python
from lionagi import Branch, Session, iModel, Builder, Operation
```

Provider API keys are read from environment variables (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, etc.) or can be passed explicitly.

---

## Branch

The primary API surface for lionagi. A `Branch` manages a single conversation
thread with message history, tool registration, and LLM operations.

`Branch` inherits from `Element` (UUID identity + timestamp + metadata) and
supports the async context manager protocol.

### Constructor

```python
Branch(
    *,
    user: SenderRecipient = None,
    name: str | None = None,
    messages: Pile[RoledMessage] = None,
    system: System | JsonValue = None,
    system_sender: SenderRecipient = None,
    chat_model: iModel | dict = None,
    parse_model: iModel | dict = None,
    imodel: iModel = None,                    # deprecated, alias of chat_model
    tools: FuncTool | list[FuncTool] = None,
    log_config: DataLoggerConfig | dict = None,
    system_datetime: bool | str = None,
    system_template = None,
    system_template_context: dict = None,
    logs: Pile[Log] = None,
    use_lion_system_message: bool = False,
    **kwargs,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user` | `SenderRecipient \| None` | `None` | Owner or sender context for this branch. |
| `name` | `str \| None` | `None` | Human-readable name for the branch. |
| `messages` | `Pile[RoledMessage] \| None` | `None` | Pre-existing messages to seed the conversation. |
| `system` | `System \| JsonValue \| None` | `None` | System message content for the LLM. |
| `system_sender` | `SenderRecipient \| None` | `None` | Sender attributed to the system message. |
| `chat_model` | `iModel \| dict \| None` | `None` | Primary chat model. Defaults to the provider/model in `AppSettings`. |
| `parse_model` | `iModel \| dict \| None` | `None` | Model for structured parsing. Falls back to `chat_model`. |
| `imodel` | `iModel \| None` | `None` | **Deprecated.** Use `chat_model` instead. |
| `tools` | `FuncTool \| list[FuncTool] \| None` | `None` | Tools (functions or `Tool` objects) to register. |
| `log_config` | `DataLoggerConfig \| dict \| None` | `None` | Logging configuration. |
| `system_datetime` | `bool \| str \| None` | `None` | Include timestamps in system messages. `True` for default format, or a `strftime` string. |
| `system_template` | `Template \| str \| None` | `None` | Jinja2 template for the system message. |
| `system_template_context` | `dict \| None` | `None` | Variables for rendering the system template. |
| `logs` | `Pile[Log] \| None` | `None` | Pre-existing logs. |
| `use_lion_system_message` | `bool` | `False` | If `True`, prepends the default Lion system prompt. |

**Example:**

```python
from lionagi import Branch, iModel

branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
    system="You are a helpful research assistant.",
)
```

### Async Context Manager

`Branch` supports `async with` for automatic log flushing on exit:

```python
async with Branch(system="You are helpful.") as branch:
    result = await branch.communicate("Hello!")
# logs are automatically dumped on exit
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `UUID` | Unique identifier (inherited from `Element`). |
| `created_at` | `float` | Creation timestamp (Unix epoch). |
| `metadata` | `dict` | Arbitrary metadata dictionary. |
| `system` | `System \| None` | The system message, if any. |
| `messages` | `Pile[RoledMessage]` | All messages in the conversation. |
| `logs` | `Pile[Log]` | All activity logs. |
| `chat_model` | `iModel` | The primary chat model (read/write). |
| `parse_model` | `iModel` | The parsing model (read/write). |
| `tools` | `dict[str, Tool]` | Registered tools, keyed by name. |
| `msgs` | `MessageManager` | The underlying message manager. |
| `acts` | `ActionManager` | The underlying action/tool manager. |
| `mdls` | `iModelManager` | The underlying model manager. |

### Methods

---

#### chat

```python
async def chat(
    self,
    instruction: Instruction | JsonValue = None,
    guidance: JsonValue = None,
    context: JsonValue = None,
    sender: ID.Ref = None,
    recipient: ID.Ref = None,
    request_fields: list[str] | dict[str, JsonValue] = None,
    response_format: type[BaseModel] | BaseModel = None,
    progression: Progression | list = None,
    imodel: iModel = None,
    tool_schemas: list[dict] = None,
    images: list = None,
    image_detail: Literal["low", "high", "auto"] = None,
    plain_content: str = None,
    return_ins_res_message: bool = False,
    include_token_usage_to_model: bool = False,
    **kwargs,
) -> tuple[Instruction, AssistantResponse]
```

Low-level LLM invocation using the current conversation history. Messages are
**not** automatically appended to the branch -- use `communicate` or `operate`
for that.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instruction` | `Instruction \| JsonValue` | `None` | Main user instruction text or data. |
| `guidance` | `JsonValue` | `None` | Additional system/user guidance. |
| `context` | `JsonValue` | `None` | Context data for the model. |
| `sender` | `ID.Ref` | `None` | Message sender (defaults to `branch.user`). |
| `recipient` | `ID.Ref` | `None` | Message recipient (defaults to `branch.id`). |
| `request_fields` | `list[str] \| dict` | `None` | Field-level validation hints. |
| `response_format` | `type[BaseModel]` | `None` | Pydantic model for structured responses. |
| `progression` | `Progression \| list` | `None` | Custom message ordering. |
| `imodel` | `iModel` | `None` | Override the default chat model. |
| `tool_schemas` | `list[dict]` | `None` | Tool schemas for function calling. |
| `images` | `list` | `None` | Images to include in the prompt. |
| `image_detail` | `"low" \| "high" \| "auto"` | `None` | Image detail level. |
| `plain_content` | `str` | `None` | Plain text content, overrides other content. |
| `return_ins_res_message` | `bool` | `False` | If `True`, returns `(Instruction, AssistantResponse)`. Otherwise returns response content only. |
| `include_token_usage_to_model` | `bool` | `False` | Include token usage in model messages. |

**Returns:** `tuple[Instruction, AssistantResponse]` -- the instruction and the model's response.

---

#### communicate

```python
async def communicate(
    self,
    instruction: Instruction | JsonValue = None,
    *,
    guidance: JsonValue = None,
    context: JsonValue = None,
    plain_content: str = None,
    sender: SenderRecipient = None,
    recipient: SenderRecipient = None,
    progression: ID.IDSeq = None,
    response_format: type[BaseModel] = None,
    request_fields: dict | list[str] = None,
    chat_model: iModel = None,
    parse_model: iModel = None,
    skip_validation: bool = False,
    images: list = None,
    image_detail: Literal["low", "high", "auto"] = None,
    num_parse_retries: int = 3,
    clear_messages: bool = False,
    include_token_usage_to_model: bool = False,
    **kwargs,
) -> str | BaseModel | dict | None
```

High-level conversational call. Messages **are** automatically added to the
branch. Simpler than `operate` -- no tool invocation.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instruction` | `Instruction \| JsonValue` | `None` | The user's main query. |
| `guidance` | `JsonValue` | `None` | Additional guidance for the model. |
| `context` | `JsonValue` | `None` | Extra context data. |
| `plain_content` | `str` | `None` | Plain text appended to the instruction. |
| `response_format` | `type[BaseModel]` | `None` | Pydantic model for structured output. |
| `request_fields` | `dict \| list[str]` | `None` | Specific fields to extract. |
| `chat_model` | `iModel` | `None` | Override the default chat model. |
| `parse_model` | `iModel` | `None` | Override the default parse model. |
| `skip_validation` | `bool` | `False` | Return raw string without parsing. |
| `images` | `list` | `None` | Images for the context. |
| `image_detail` | `"low" \| "high" \| "auto"` | `None` | Image detail level. |
| `num_parse_retries` | `int` | `3` | Max parse retries (capped at 5). |
| `clear_messages` | `bool` | `False` | Clear stored messages before sending. |
| `include_token_usage_to_model` | `bool` | `False` | Include token usage in model messages. |

**Returns:** `str | BaseModel | dict | None` -- raw string, validated model,
field dict, or `None` on parse failure.

**Example:**

```python
# Simple text response
answer = await branch.communicate("What is the capital of France?")

# Structured response
from pydantic import BaseModel

class Answer(BaseModel):
    city: str
    country: str

result = await branch.communicate(
    "What is the capital of France?",
    response_format=Answer,
)
print(result.city)  # "Paris"
```

---

#### operate

```python
async def operate(
    self,
    *,
    instruct: Instruct = None,
    instruction: Instruction | JsonValue = None,
    guidance: JsonValue = None,
    context: JsonValue = None,
    sender: SenderRecipient = None,
    recipient: SenderRecipient = None,
    progression: Progression = None,
    chat_model: iModel = None,
    invoke_actions: bool = True,
    tool_schemas: list[dict] = None,
    images: list = None,
    image_detail: Literal["low", "high", "auto"] = None,
    parse_model: iModel = None,
    skip_validation: bool = False,
    tools: ToolRef = None,
    operative: Operative = None,
    response_format: type[BaseModel] = None,
    actions: bool = False,
    reason: bool = False,
    call_params: AlcallParams = None,
    action_strategy: Literal["sequential", "concurrent"] = "concurrent",
    verbose_action: bool = False,
    field_models: list[FieldModel] = None,
    exclude_fields: list | dict | None = None,
    handle_validation: Literal["raise", "return_value", "return_none"] = "return_value",
    include_token_usage_to_model: bool = False,
    **kwargs,
) -> list | BaseModel | None | dict | str
```

Full orchestration with optional tool invocation and structured response
parsing. Messages **are** automatically added to the conversation.

**Key Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instruct` | `Instruct` | `None` | Instruction bundle (instruction + guidance + context). |
| `instruction` | `Instruction \| JsonValue` | `None` | Direct instruction (alternative to `instruct`). |
| `guidance` | `JsonValue` | `None` | Additional guidance. |
| `context` | `JsonValue` | `None` | Context data. |
| `chat_model` | `iModel` | `None` | Override chat model. |
| `invoke_actions` | `bool` | `True` | Automatically invoke tools from LLM response. |
| `tools` | `ToolRef` | `None` | Tools to make available. |
| `response_format` | `type[BaseModel]` | `None` | Pydantic model for the response. |
| `actions` | `bool` | `False` | Signal that function-calling is expected. |
| `reason` | `bool` | `False` | Request chain-of-thought reasoning. |
| `action_strategy` | `"sequential" \| "concurrent"` | `"concurrent"` | Tool invocation strategy. |
| `skip_validation` | `bool` | `False` | Skip response parsing/validation. |
| `handle_validation` | `"raise" \| "return_value" \| "return_none"` | `"return_value"` | Behavior on parse failure. |

**Returns:** `list | BaseModel | None | dict | str` -- parsed response,
raw text, or `None` depending on validation settings.

**Example:**

```python
from pydantic import BaseModel

class AnalysisReport(BaseModel):
    summary: str
    key_findings: list[str]
    confidence: float

report = await branch.operate(
    instruction="Analyze the quarterly revenue data.",
    context={"revenue": [100, 120, 115, 140]},
    response_format=AnalysisReport,
    reason=True,
)
print(report.summary)
```

---

#### parse

```python
async def parse(
    self,
    text: str,
    handle_validation: Literal["raise", "return_value", "return_none"] = "return_value",
    max_retries: int = 3,
    request_type: type[BaseModel] = None,
    operative: Operative = None,
    similarity_algo: str = "jaro_winkler",
    similarity_threshold: float = 0.85,
    fuzzy_match: bool = True,
    handle_unmatched: Literal["ignore", "raise", "remove", "fill", "force"] = "force",
    fill_value: Any = None,
    fill_mapping: dict[str, Any] | None = None,
    strict: bool = False,
    suppress_conversion_errors: bool = False,
    response_format: type[BaseModel] = None,
) -> BaseModel | dict | str | None
```

Parses raw text into a structured Pydantic model using the parse model.
Supports fuzzy key matching for malformed LLM output. Does **not** append
messages to the conversation.

**Key Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *(required)* | Raw text to parse. |
| `request_type` | `type[BaseModel]` | `None` | Target Pydantic model. |
| `response_format` | `type[BaseModel]` | `None` | Alias for `request_type`. |
| `handle_validation` | `"raise" \| "return_value" \| "return_none"` | `"return_value"` | Behavior on parse failure. |
| `max_retries` | `int` | `3` | Retry count for failed parses. |
| `fuzzy_match` | `bool` | `True` | Attempt fuzzy key matching. |
| `similarity_threshold` | `float` | `0.85` | Threshold for fuzzy matching (0.0-1.0). |
| `handle_unmatched` | `"ignore" \| "raise" \| "remove" \| "fill" \| "force"` | `"force"` | Policy for unrecognized fields. |
| `strict` | `bool` | `False` | Raise on ambiguous types/fields. |

**Returns:** `BaseModel | dict | str | None`

---

#### ReAct

```python
async def ReAct(
    self,
    instruct: Instruct | dict[str, Any],
    interpret: bool = False,
    interpret_domain: str | None = None,
    interpret_style: str | None = None,
    interpret_sample: str | None = None,
    interpret_model: str | None = None,
    interpret_kwargs: dict | None = None,
    tools: Any = None,
    tool_schemas: Any = None,
    response_format: type[BaseModel] | BaseModel = None,
    intermediate_response_options: list[BaseModel] | BaseModel = None,
    intermediate_listable: bool = False,
    reasoning_effort: Literal["low", "medium", "high"] = None,
    extension_allowed: bool = True,
    max_extensions: int | None = 3,
    response_kwargs: dict | None = None,
    display_as: Literal["json", "yaml"] = "yaml",
    return_analysis: bool = False,
    analysis_model: iModel | None = None,
    verbose: bool = False,
    verbose_length: int = None,
    include_token_usage_to_model: bool = True,
    **kwargs,
) -> Any | tuple[Any, list]
```

Multi-step Reason + Act loop. Iteratively generates chain-of-thought analysis,
invokes tools, and optionally extends the reasoning for additional steps.
Messages are automatically added to the branch.

**Key Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instruct` | `Instruct \| dict` | *(required)* | Instruction with `instruction`, `guidance`, and `context` keys. |
| `interpret` | `bool` | `False` | Pre-process the instruction through `interpret()`. |
| `interpret_domain` | `str` | `None` | Domain hint for interpretation (e.g., `"finance"`). |
| `interpret_style` | `str` | `None` | Style hint (e.g., `"concise"`, `"detailed"`). |
| `tools` | `Any` | `None` | Tools to use. `None` defaults to all registered tools. |
| `response_format` | `type[BaseModel]` | `None` | Final output schema. |
| `extension_allowed` | `bool` | `True` | Allow multi-step expansion. |
| `max_extensions` | `int` | `3` | Max expansion steps (capped at 5). |
| `return_analysis` | `bool` | `False` | Also return intermediate analysis objects. |
| `analysis_model` | `iModel` | `None` | Override model for analysis steps. |
| `reasoning_effort` | `"low" \| "medium" \| "high"` | `None` | Reasoning depth hint. |
| `verbose` | `bool` | `False` | Log detailed analysis and action info. |

**Returns:**

- If `return_analysis=False`: the final output (string, dict, or BaseModel).
- If `return_analysis=True`: `tuple[final_output, list[ReActAnalysis]]`.

**Example:**

```python
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

branch = Branch(
    system="You are a research assistant.",
    tools=[search],
)

result = await branch.ReAct(
    instruct={"instruction": "Find the population of Tokyo"},
    max_extensions=2,
)
```

---

#### ReActStream

```python
async def ReActStream(
    self,
    instruct: Instruct | dict[str, Any],
    # ... same parameters as ReAct ...
    **kwargs,
) -> AsyncGenerator
```

Streaming variant of `ReAct`. Yields intermediate `ReActAnalysis` objects as
each reasoning step completes.

**Returns:** `AsyncGenerator` yielding analysis results per step.

**Example:**

```python
async for step in branch.ReActStream(
    instruct={"instruction": "Research and compare Python web frameworks"},
    verbose=True,
):
    print(f"Step completed: {step}")
```

---

#### act

```python
async def act(
    self,
    action_request: list | ActionRequest | BaseModel | dict,
    *,
    strategy: Literal["concurrent", "sequential"] = "concurrent",
    verbose_action: bool = False,
    suppress_errors: bool = True,
    call_params: AlcallParams = None,
) -> list[ActionResponse]
```

Directly invokes tool actions without an LLM call. Useful for executing
tool calls programmatically.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action_request` | `list \| ActionRequest \| BaseModel \| dict` | *(required)* | Tool call request(s) to execute. |
| `strategy` | `"concurrent" \| "sequential"` | `"concurrent"` | Execution strategy. |
| `verbose_action` | `bool` | `False` | Log action details. |
| `suppress_errors` | `bool` | `True` | Suppress exceptions from tool calls. |

**Returns:** `list[ActionResponse]`

---

#### interpret

```python
async def interpret(
    self,
    text: str,
    domain: str | None = None,
    style: str | None = None,
    interpret_model = None,
    **kwargs,
) -> str
```

Rewrites raw user input into a clearer, more structured LLM prompt. Does
**not** add messages to the conversation.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *(required)* | Raw user input to rewrite. |
| `domain` | `str` | `None` | Domain hint (e.g., `"finance"`, `"devops"`). |
| `style` | `str` | `None` | Style hint (e.g., `"concise"`, `"detailed"`). |

**Returns:** `str` -- the refined prompt.

**Example:**

```python
refined = await branch.interpret(
    "how do i do marketing stuff",
    domain="marketing",
    style="detailed",
)
# => "Explain step-by-step how to set up a marketing analytics pipeline..."
```

---

#### register_tools

```python
def register_tools(
    self,
    tools: FuncTool | list[FuncTool] | LionTool,
    update: bool = False,
) -> None
```

Registers one or more tools (functions or `Tool` objects) in the branch.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tools` | `FuncTool \| list[FuncTool] \| LionTool` | *(required)* | Tool(s) to register. Can be plain functions, `Tool` instances, or `LionTool` subclasses. |
| `update` | `bool` | `False` | Overwrite existing tools with the same name. |

**Example:**

```python
def multiply(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y

branch.register_tools([multiply])
```

---

#### get_operation

```python
def get_operation(self, operation: str) -> Callable | None
```

Looks up an operation by name. First checks for a method on the Branch, then
falls back to the operation registry.

**Parameters:**

- `operation` (`str`): Operation name (e.g., `"chat"`, `"communicate"`).

**Returns:** `Callable | None`

---

#### clone

```python
def clone(self, sender: ID.Ref = None) -> Branch
```

Creates a synchronous copy of the branch, including messages, system config,
tools, and models. API-backed models are shared; CLI-backed models get fresh
copies.

**Parameters:**

- `sender` (`ID.Ref`, optional): New sender ID for all cloned messages.

**Returns:** `Branch` -- a new branch instance.

---

#### aclone

```python
async def aclone(self, sender: ID.Ref = None) -> Branch
```

Async variant of `clone`. Acquires the message lock before cloning.

**Returns:** `Branch`

---

#### to_dict

```python
def to_dict(self) -> dict
```

Serializes the branch to a dictionary including messages, logs, models, system
message, log config, and metadata.

**Returns:** `dict`

---

#### from_dict

```python
@classmethod
def from_dict(cls, data: dict) -> Branch
```

Deserializes a `Branch` from a dictionary produced by `to_dict`.

**Parameters:**

- `data` (`dict`): Serialized branch data.

**Returns:** `Branch`

---

#### to_df

```python
def to_df(self, *, progression: Progression = None) -> pd.DataFrame
```

Converts branch messages to a pandas DataFrame.

**Parameters:**

- `progression` (`Progression`, optional): Custom message ordering.

**Returns:** `pd.DataFrame`

---

#### dump_logs / adump_logs

```python
def dump_logs(self, clear: bool = True, persist_path = None) -> None
async def adump_logs(self, clear: bool = True, persist_path = None) -> None
```

Writes logs to disk. If `clear=True` (default), clears the in-memory log after
writing.

---

## Session

Manages multiple `Branch` instances and executes graph-based workflows. Creates
a default branch automatically on initialization.

### Constructor

```python
Session(
    branches: Pile[Branch] = ...,
    default_branch: Branch | None = None,
    name: str = "Session",
    user: SenderRecipient | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `branches` | `Pile[Branch]` | auto | Collection of branches. A default empty `Pile` is created. |
| `default_branch` | `Branch \| None` | `None` | Primary branch. Created automatically if `None`. |
| `name` | `str` | `"Session"` | Session name. |
| `user` | `SenderRecipient \| None` | `None` | User/owner of the session. Propagated to branches. |

**Example:**

```python
from lionagi import Session, iModel

session = Session()
# session.default_branch is ready to use
result = await session.default_branch.communicate("Hello!")
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `branches` | `Pile[Branch]` | All branches in the session. |
| `default_branch` | `Branch` | The active default branch. |
| `name` | `str` | Session name. |
| `user` | `SenderRecipient \| None` | Session owner. |

### Methods

---

#### flow

```python
async def flow(
    self,
    graph: Graph,
    *,
    context: dict[str, Any] | None = None,
    parallel: bool = True,
    max_concurrent: int = 5,
    verbose: bool = False,
    default_branch: Branch | ID.Ref | None = None,
    alcall_params: Any = None,
) -> dict[str, Any]
```

Executes a graph-based workflow using multi-branch orchestration. Independent
operations run in parallel; dependent operations run sequentially.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `Graph` | *(required)* | Workflow graph containing `Operation` nodes. |
| `context` | `dict` | `None` | Initial context passed to operations. |
| `parallel` | `bool` | `True` | Run independent operations concurrently. |
| `max_concurrent` | `int` | `5` | Max concurrent branches. |
| `verbose` | `bool` | `False` | Enable verbose logging. |
| `default_branch` | `Branch \| ID.Ref` | `None` | Branch override (defaults to `self.default_branch`). |

**Returns:** `dict[str, Any]` -- execution results with completed operations
and final context.

**Example:**

```python
from lionagi import Session, Builder

session = Session()
builder = Builder()

step1 = builder.add_operation("communicate", instruction="Brainstorm ideas")
step2 = builder.add_operation(
    "communicate",
    instruction="Evaluate the ideas",
    depends_on=[step1],
    inherit_context=True,
)

result = await session.flow(builder.get_graph())
```

---

#### new_branch

```python
def new_branch(
    self,
    system: System | JsonValue = None,
    system_sender: SenderRecipient = None,
    system_datetime: bool | str = None,
    user: SenderRecipient = None,
    name: str | None = None,
    imodel: iModel | None = None,
    messages: Pile[RoledMessage] = None,
    progress: Progression = None,
    tool_manager: ActionManager = None,
    tools: Tool | Callable | list = None,
    as_default_branch: bool = False,
    **kwargs,
) -> Branch
```

Creates a new branch and includes it in the session.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system` | `System \| JsonValue` | `None` | System message for the new branch. |
| `name` | `str` | `None` | Branch name. |
| `imodel` | `iModel` | `None` | Model for the branch. |
| `tools` | `Tool \| Callable \| list` | `None` | Tools to register. |
| `as_default_branch` | `bool` | `False` | Make this the session's default branch. |

**Returns:** `Branch`

---

#### get_branch

```python
def get_branch(self, branch: ID.Ref | str, default: Any = ..., /) -> Branch
```

Retrieves a branch by UUID or name. Raises `ItemNotFoundError` if not found
and no default is given.

---

#### split / asplit

```python
def split(self, branch: ID.Ref) -> Branch
async def asplit(self, branch: ID.Ref) -> Branch
```

Clones a branch and includes the clone in the session.

**Returns:** `Branch` -- the newly created clone.

---

#### remove_branch

```python
def remove_branch(self, branch: ID.Ref, delete: bool = False) -> None
```

Removes a branch from the session. If the removed branch was the default, the
first remaining branch becomes the new default.

---

#### change_default_branch

```python
def change_default_branch(self, branch: ID.Ref) -> None
```

Sets a different branch as the session default.

---

#### register_operation

```python
def register_operation(
    self, operation: str, func: Callable, *, update: bool = False
) -> None
```

Registers a custom operation callable, accessible from all branches in the
session.

---

#### operation (decorator)

```python
@session.operation(name=None, update=False)
async def my_custom_op():
    ...
```

Decorator to register a function as a named operation on the session.

---

#### concat_messages

```python
def concat_messages(
    self,
    branches: ID.RefSeq = None,
    exclude_clone: bool = False,
    exclude_load: bool = False,
) -> Pile[RoledMessage]
```

Concatenates messages from multiple branches into a single `Pile`.

---

#### to_df

```python
def to_df(
    self,
    branches: ID.RefSeq = None,
    exclude_clone: bool = False,
    exlcude_load: bool = False,
) -> pd.DataFrame
```

Converts session messages across branches into a DataFrame.

---

## iModel

Unified provider interface for LLM API calls with rate limiting, retry logic,
and hook support. Supports OpenAI, Anthropic, Gemini, Ollama, NVIDIA NIM,
Perplexity, Groq, OpenRouter, and Claude Code CLI.

### Constructor

```python
iModel(
    provider: str = None,
    base_url: str = None,
    endpoint: str | Endpoint = "chat",
    api_key: str = None,
    queue_capacity: int | None = None,
    capacity_refresh_time: float = 60,
    interval: float | None = None,
    limit_requests: int = None,
    limit_tokens: int = None,
    concurrency_limit: int | None = None,
    streaming_process_func: Callable = None,
    provider_metadata: dict | None = None,
    hook_registry: HookRegistry | dict | None = None,
    exit_hook: bool = False,
    id: UUID | str = None,
    created_at: float | None = None,
    **kwargs,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | `None` | Provider name: `"openai"`, `"anthropic"`, `"gemini"`, `"ollama"`, `"claude_code"`, etc. Can also be inferred from `model` if given as `"provider/model"`. |
| `base_url` | `str` | `None` | Custom API base URL. |
| `endpoint` | `str \| Endpoint` | `"chat"` | Endpoint type or a pre-built `Endpoint` instance. |
| `api_key` | `str` | `None` | API key. Falls back to environment variables. |
| `queue_capacity` | `int` | `None` | Max queued requests before execution. |
| `capacity_refresh_time` | `float` | `60` | Seconds between queue capacity resets. |
| `interval` | `float` | `None` | Request processing interval. Defaults to `capacity_refresh_time`. |
| `limit_requests` | `int` | `None` | Max requests per cycle. |
| `limit_tokens` | `int` | `None` | Max tokens per cycle. |
| `concurrency_limit` | `int` | `None` | Max concurrent streaming requests (CLI only). |
| `streaming_process_func` | `Callable` | `None` | Custom function to process streaming chunks. |
| `provider_metadata` | `dict` | `None` | Provider-specific metadata (e.g., session IDs). |
| `hook_registry` | `HookRegistry \| dict` | `None` | Pre/post-invocation hooks. |
| `exit_hook` | `bool` | `False` | Enable exit hooks on invocation. |
| `**kwargs` | | | Provider-specific parameters (e.g., `model`, `temperature`, `max_tokens`). |

**Example:**

```python
from lionagi import iModel

# OpenAI
model = iModel(provider="openai", model="gpt-4.1-mini", temperature=0.7)

# Anthropic
model = iModel(provider="anthropic", model="claude-sonnet-4-20250514")

# Shorthand: provider/model
model = iModel(model="openai/gpt-4.1-mini")

# Ollama (local)
model = iModel(provider="ollama", model="llama3")
```

### Async Context Manager

```python
async with iModel(provider="openai", model="gpt-4.1-mini") as model:
    result = await model.invoke(messages=[...])
# executor is stopped and resources released on exit
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `UUID` | Unique model instance identifier. |
| `created_at` | `float` | Creation timestamp. |
| `endpoint` | `Endpoint` | The configured endpoint. |
| `executor` | `RateLimitedAPIExecutor` | Rate-limited request executor. |
| `is_cli` | `bool` | Whether this model uses a CLI endpoint. |
| `model_name` | `str` | The model name string (e.g., `"gpt-4.1-mini"`). |
| `request_options` | `type[BaseModel] \| None` | Request schema for the endpoint. |
| `hook_registry` | `HookRegistry` | Registered hooks. |

### Methods

---

#### invoke

```python
async def invoke(self, api_call: APICalling = None, **kw) -> APICalling
```

Performs a rate-limited API call. Starts the executor if needed, enqueues
the request, and waits for completion.

**Parameters:**

- `api_call` (`APICalling`, optional): Pre-built API call. If `None`, one is
  created from `**kw`.
- `**kw`: Request parameters (e.g., `messages`, `temperature`).

**Returns:** `APICalling` -- the completed call with `.response` populated.

**Raises:** `ValueError` on invocation failure.

---

#### stream

```python
async def stream(self, api_call=None, **kw) -> AsyncGenerator
```

Performs a streaming API call, yielding chunks as they arrive.

**Parameters:**

- `api_call`: Pre-built API call. If `None`, one is created with `stream=True`.
- `**kw`: Request parameters.

**Yields:** Processed chunks (via `streaming_process_func` if set) or raw
chunks. The final yield is the completed `APICalling` object.

---

#### create_api_calling

```python
def create_api_calling(
    self,
    include_token_usage_to_model: bool = False,
    **kwargs,
) -> APICalling
```

Builds an `APICalling` event from keyword arguments using the configured
endpoint.

**Returns:** `APICalling`

---

#### create_event

```python
async def create_event(
    self,
    create_event_type: type[Event] = APICalling,
    create_event_exit_hook: bool = None,
    create_event_hook_timeout: float = 10.0,
    create_event_hook_params: dict = None,
    pre_invoke_event_exit_hook: bool = None,
    pre_invoke_event_hook_timeout: float = 30.0,
    pre_invoke_event_hook_params: dict = None,
    post_invoke_event_exit_hook: bool = None,
    post_invoke_event_hook_timeout: float = 30.0,
    post_invoke_event_hook_params: dict = None,
    **kwargs,
) -> APICalling
```

Creates an API call event with optional pre/post-invocation hooks. Used
internally by `invoke()` and `stream()`.

---

#### process_chunk

```python
async def process_chunk(self, chunk) -> Any
```

Processes a single streaming chunk. Override this or provide
`streaming_process_func` for custom chunk handling.

---

#### copy

```python
def copy(self, share_session: bool = False) -> iModel
```

Creates a new `iModel` with the same configuration but a fresh ID and
executor.

**Parameters:**

- `share_session` (`bool`): If `True`, carries over CLI session state.

**Returns:** `iModel`

---

#### close

```python
async def close(self) -> None
```

Stops the executor and releases resources.

---

#### to_dict / from_dict

```python
def to_dict(self) -> dict
@classmethod
def from_dict(cls, data: dict) -> iModel
```

Serialization and deserialization. The dict includes `id`, `created_at`,
`endpoint` config, `processor_config`, and `provider_metadata`.

---

## Builder

`OperationGraphBuilder` (aliased as `Builder`) constructs directed acyclic
graphs of operations for execution by `Session.flow()`. Supports incremental
build-execute-expand cycles.

### Constructor

```python
Builder(name: str = "DynamicGraph")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"DynamicGraph"` | Name for the graph. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `graph` | `Graph` | The underlying directed graph. |
| `last_operation_id` | `str \| None` | ID of the most recently added operation. |

### Methods

---

#### add_operation

```python
def add_operation(
    self,
    operation: str,
    node_id: str | None = None,
    depends_on: list[str] | None = None,
    inherit_context: bool = False,
    branch = None,
    **parameters,
) -> str
```

Adds an operation node to the graph. If no `depends_on` is specified, the node
is linked sequentially from the current head nodes.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `operation` | `str` | *(required)* | Branch operation name: `"chat"`, `"communicate"`, `"operate"`, `"ReAct"`, etc. |
| `node_id` | `str` | `None` | Reference ID for later lookup. |
| `depends_on` | `list[str]` | `None` | Node IDs this operation depends on. |
| `inherit_context` | `bool` | `False` | Inherit conversation context from the primary dependency. |
| `branch` | `Branch \| ID.Ref` | `None` | Specific branch to run on. |
| `**parameters` | | | Operation parameters (e.g., `instruction`, `response_format`). |

**Returns:** `str` -- UUID of the created operation node.

**Example:**

```python
from lionagi import Builder

builder = Builder()
step1 = builder.add_operation("communicate", instruction="List 5 ideas")
step2 = builder.add_operation(
    "communicate",
    instruction="Evaluate these ideas",
    depends_on=[step1],
    inherit_context=True,
)
```

---

#### add_aggregation

```python
def add_aggregation(
    self,
    operation: str,
    node_id: str | None = None,
    source_node_ids: list[str] | None = None,
    inherit_context: bool = False,
    inherit_from_source: int = 0,
    branch = None,
    **parameters,
) -> str
```

Adds a node that aggregates results from multiple source nodes (defaults to
current head nodes).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `operation` | `str` | *(required)* | The aggregation operation. |
| `source_node_ids` | `list[str]` | `None` | Nodes to aggregate from. Defaults to current heads. |
| `inherit_context` | `bool` | `False` | Inherit context from one source. |
| `inherit_from_source` | `int` | `0` | Index of the source to inherit from. |

**Returns:** `str` -- UUID of the aggregation node.

**Example:**

```python
# Fan-out then aggregate
nodes = []
for topic in ["AI", "ML", "NLP"]:
    nodes.append(builder.add_operation("communicate", instruction=f"Research {topic}"))

synthesis = builder.add_aggregation(
    "communicate",
    source_node_ids=nodes,
    instruction="Synthesize findings",
)
```

---

#### expand_from_result

```python
def expand_from_result(
    self,
    items: list[Any],
    source_node_id: str,
    operation: str,
    strategy: ExpansionStrategy = ExpansionStrategy.CONCURRENT,
    inherit_context: bool = False,
    chain_context: bool = False,
    **shared_params,
) -> list[str]
```

Expands the graph based on execution results. Creates new operation nodes for
each item in `items`, linked from the source node.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `items` | `list[Any]` | *(required)* | Items to expand (e.g., from a previous result). |
| `source_node_id` | `str` | *(required)* | Node that produced the items. |
| `operation` | `str` | *(required)* | Operation to apply to each item. |
| `strategy` | `ExpansionStrategy` | `CONCURRENT` | `CONCURRENT` or `SEQUENTIAL`. |
| `inherit_context` | `bool` | `False` | Inherit context from source. |
| `chain_context` | `bool` | `False` | Chain context between sequential expansions. |

**Returns:** `list[str]` -- IDs of the new nodes.

---

#### add_conditional_branch

```python
def add_conditional_branch(
    self,
    condition_check_op: str,
    true_op: str,
    false_op: str | None = None,
    **check_params,
) -> dict[str, str]
```

Adds a conditional branching structure: a check node with `true` and optional
`false` paths.

**Returns:** `dict` with keys `"check"`, `"true"`, and optionally `"false"`,
each mapping to a node ID.

---

#### get_graph

```python
def get_graph(self) -> Graph
```

Returns the current graph for execution by `Session.flow()`.

**Returns:** `Graph`

---

#### mark_executed

```python
def mark_executed(self, node_ids: list[str]) -> None
```

Marks nodes as executed, useful for incremental build-execute cycles.

---

#### get_unexecuted_nodes

```python
def get_unexecuted_nodes(self) -> list[Operation]
```

Returns all operations not yet marked as executed.

---

#### get_node_by_reference

```python
def get_node_by_reference(self, reference_id: str) -> Operation | None
```

Looks up a node by its `node_id` reference (set via `add_operation`).

---

#### visualize_state

```python
def visualize_state(self) -> dict[str, Any]
```

Returns a summary dict with `total_nodes`, `executed_nodes`,
`unexecuted_nodes`, `current_heads`, `expansions`, and `edges`.

---

#### visualize

```python
def visualize(self, title: str = "Operation Graph", figsize=(14, 10)) -> None
```

Renders the graph visually using matplotlib.

---

## Operation

A graph node representing a single branch operation. Extends both `Node` and
`Event`, tracking execution status, timing, and results.

### Constructor

```python
Operation(
    operation: str,
    parameters: dict[str, Any] | BaseModel = {},
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `operation` | `str` | *(required)* | Operation name: `"chat"`, `"operate"`, `"communicate"`, `"parse"`, `"ReAct"`, `"interpret"`, `"act"`, `"ReActStream"`, or a custom name. |
| `parameters` | `dict \| BaseModel` | `{}` | Parameters passed to the operation. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `branch_id` | `UUID \| None` | ID of the branch that executed this operation. |
| `graph_id` | `UUID \| None` | ID of the graph containing this operation. |
| `request` | `dict` | Parameters as a dictionary. |
| `response` | `Any` | Execution result (populated after `invoke`). |

### Methods

---

#### invoke

```python
async def invoke(self, branch: Branch) -> None
```

Executes the operation on the given branch. Updates `execution.status`,
`execution.response`, `execution.duration`, and `execution.error`.

**Parameters:**

- `branch` (`Branch`): The branch to execute on.

---

### Factory Function

```python
from lionagi.operations.node import create_operation

op = create_operation(
    operation="communicate",
    parameters={"instruction": "Hello"},
)
```

---

## Pile

Thread-safe, async-compatible ordered collection keyed by UUID. Items are
accessed by UUID (not by list index). Supports JSON, CSV, and DataFrame
adapters.

### Constructor

```python
Pile(
    collections: list[Element] = None,
    item_type: set[type] = None,
    order: list[UUID] = None,
    strict_type: bool = False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collections` | `list[Element]` | `None` | Initial items. |
| `item_type` | `set[type]` | `None` | Allowed element types. |
| `order` | `list[UUID]` | `None` | Explicit ordering. |
| `strict_type` | `bool` | `False` | Disallow subtypes if `True`. |

### Key Interface

| Method/Operator | Description |
|-----------------|-------------|
| `pile[uuid]` | Get item by UUID. |
| `pile[0]`, `pile[1:3]` | Get by index or slice. |
| `pile.include(item)` | Add item if not present. |
| `pile.exclude(item)` | Remove item if present. |
| `pile.pop(key)` | Remove and return item. |
| `pile.get(key, default)` | Get item with fallback. |
| `pile.update(items)` | Update/add items. |
| `pile.insert(index, item)` | Insert at position. |
| `pile.clear()` | Remove all items. |
| `pile.keys()` | All UUIDs in order. |
| `pile.values()` | All items in order. |
| `pile.items()` | `(UUID, item)` pairs in order. |
| `len(pile)` | Number of items. |
| `item in pile` | Membership test. |
| `pile.is_empty()` | Check emptiness. |
| `pile.to_df(columns=...)` | Convert to pandas DataFrame. |
| `pile.dump(path, "json")` | Export to JSON, CSV, or Parquet. |
| `pile \| other` | Union. |
| `pile & other` | Intersection. |
| `pile ^ other` | Symmetric difference. |

All mutating methods have async variants prefixed with `a` (e.g., `apop`,
`ainclude`, `aexclude`, `aclear`, `aupdate`, `aget`).

Pile also works as an async context manager for locked access:

```python
async with pile:
    pile.include(item)
```

---

## Progression

Ordered sequence of UUIDs, decoupled from storage. Multiple progressions can
index the same `Pile` without copying data. Internally backed by
`collections.deque` for O(1) `popleft()` and `append()`.

### Constructor

```python
Progression(
    order: deque[UUID] = deque(),
    name: str | None = None,
)
```

### Key Interface

| Method/Operator | Description |
|-----------------|-------------|
| `prog[0]`, `prog[1:3]` | Index/slice access. |
| `prog.append(item)` | Append UUID(s). |
| `prog.insert(index, item)` | Insert at position. |
| `prog.remove(item)` | Remove first occurrence. |
| `prog.pop(index=-1)` | Remove and return by index. |
| `prog.popleft()` | O(1) remove and return from front. |
| `prog.include(item)` | Add if not present. |
| `prog.exclude(item)` | Remove if present. |
| `prog.clear()` | Remove all. |
| `prog.index(item)` | Find position. |
| `prog.count(item)` | Count occurrences. |
| `prog.extend(other)` | Extend from another Progression. |
| `len(prog)` | Length. |
| `item in prog` | O(1) membership test (via internal set). |
| `prog + other` | Concatenation (new Progression). |
| `prog - other` | Difference (new Progression). |

---

## Supporting Types

These types are used throughout the API but do not typically need to be
instantiated directly.

### Instruct

Instruction bundle used by `operate` and `ReAct`:

```python
from lionagi.operations.fields import Instruct

instruct = Instruct(
    instruction="Analyze the data",
    guidance="Focus on trends",
    context={"data": [1, 2, 3]},
    reason=True,
)
```

### Element

Base class for all identifiable objects. Provides `id` (UUID), `created_at`
(float), and `metadata` (dict).

```python
from lionagi import Element

el = Element()
print(el.id)          # UUID
print(el.created_at)  # float timestamp
```

### Node

`Element` with arbitrary `content` and optional embedding vector. Used as the
base for graph nodes.

### Graph

Directed graph of `Node` and `Edge` objects with adjacency tracking.

### ExpansionStrategy

Enum for `Builder.expand_from_result`:

```python
from lionagi.operations.builder import ExpansionStrategy

ExpansionStrategy.CONCURRENT             # Run expanded ops in parallel
ExpansionStrategy.SEQUENTIAL             # Run expanded ops in sequence
ExpansionStrategy.SEQUENTIAL_CONCURRENT_CHUNK
ExpansionStrategy.CONCURRENT_SEQUENTIAL_CHUNK
```

---

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
export OLLAMA_BASE_URL="http://localhost:11434"  # default
```

### AppSettings

Default model settings are controlled through `lionagi.config.settings`:

- `LIONAGI_CHAT_PROVIDER` -- default: `"openai"`
- `LIONAGI_CHAT_MODEL` -- default: `"gpt-4.1-mini"`

Override by setting environment variables or passing `chat_model` to `Branch`.

---

## Quick Reference

### Common Patterns

**Simple conversation:**

```python
from lionagi import Branch

branch = Branch(system="You are helpful.")
answer = await branch.communicate("What is 2 + 2?")
```

**Structured output:**

```python
from pydantic import BaseModel
from lionagi import Branch

class Answer(BaseModel):
    value: int
    explanation: str

result = await branch.communicate(
    "What is 2 + 2?",
    response_format=Answer,
)
```

**Tool use with ReAct:**

```python
def calculator(expression: str) -> float:
    """Evaluate a math expression."""
    return eval(expression)

branch = Branch(system="You solve math problems.", tools=[calculator])
result = await branch.ReAct({"instruction": "What is 15% of 847?"})
```

**Graph workflow:**

```python
from lionagi import Session, Builder

session = Session()
builder = Builder()

# Parallel research
ids = [
    builder.add_operation("communicate", instruction=f"Research {t}")
    for t in ["AI safety", "AI alignment", "AI governance"]
]

# Aggregate
builder.add_aggregation(
    "communicate",
    source_node_ids=ids,
    instruction="Synthesize a unified report",
)

result = await session.flow(builder.get_graph())
```
