![PyPI - Version](https://img.shields.io/pypi/v/lionagi?labelColor=233476aa&color=231fc935)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lionagi?color=blue)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
[![codecov](https://codecov.io/github/khive-ai/lionagi/graph/badge.svg?token=FAE47FY26T)](https://codecov.io/github/khive-ai/lionagi)

[Documentation](https://khive-ai.github.io/lionagi/) |
[Discord](https://discord.gg/JDj9ENhUE8) |
[PyPI](https://pypi.org/project/lionagi/)

# LION - Language InterOperable Network

## An AGentic Intelligence SDK

LionAGI is a robust framework for orchestrating multi-step AI operations with
precise control. Bring together multiple models, advanced ReAct reasoning, tool
integrations, and custom validations in a single coherent pipeline.

## Why LionAGI?

- **Structured**: Validate and type all LLM interactions with Pydantic.
- **Expandable**: Integrate multiple providers (OpenAI, Anthropic, Perplexity,
  custom) with minimal friction.
- **Controlled**: Use built-in safety checks, concurrency strategies, and advanced
  multi-step flows like ReAct.
- **Transparent**: Debug easily with real-time logging, message introspection, and
  tool usage tracking.

## Installation

```
uv add lionagi  # recommended to use pyproject and uv for dependency management

pip install lionagi # or install directly
```

## Quick Start

```python
from lionagi import Branch, iModel

# Pick a model
gpt4o = iModel(provider="openai", model="gpt-4o-mini")

# Create a Branch (conversation context)
hunter = Branch(
  system="you are a hilarious dragon hunter who responds in 10 words rhymes.",
  chat_model=gpt4o,
)

# Communicate asynchronously
response = await hunter.communicate("I am a dragon")
print(response)
```

```
You claim to be a dragon, oh what a braggin'!
```

### Structured Responses

Use Pydantic to keep outputs structured:

```python
from pydantic import BaseModel

class Joke(BaseModel):
    joke: str

res = await hunter.operate(
    instruction="Tell me a short dragon joke",
    response_format=Joke
)
print(type(res))
print(res.joke)
```

```
<class '__main__.Joke'>
With fiery claws, dragons hide their laughter flaws!
```

### ReAct and Tools

LionAGI supports advanced multi-step reasoning with ReAct. Tools let the LLM
invoke external actions:

```
pip install "lionagi[reader]"
```

```python
from lionagi.tools.types import ReaderTool

# Define model first
gpt4o = iModel(provider="openai", model="gpt-4o-mini")

branch = Branch(chat_model=gpt4o, tools=[ReaderTool])
result = await branch.ReAct(
    instruct={
      "instruction": "Summarize my PDF and compare with relevant papers.",
      "context": {"paper_file_path": "/path/to/paper.pdf"},
    },
    extension_allowed=True,     # allow multi-round expansions
    max_extensions=5,
    verbose=True,      # see step-by-step chain-of-thought
)
print(result)
```

The LLM can now open the PDF, read in slices, fetch references, and produce a
final structured summary.

### MCP (Model Context Protocol) Integration

LionAGI supports Anthropic's Model Context Protocol for seamless tool integration:

```
pip install "lionagi[mcp]"
```

```python
from lionagi import load_mcp_tools

# Load tools from any MCP server
tools = await load_mcp_tools(".mcp.json", ["search", "memory"])

# Use with ReAct reasoning
branch = Branch(chat_model=gpt4o, tools=tools)
result = await branch.ReAct(
    instruct={"instruction": "Research recent AI developments"},
    tools=["search_exa_search"],
    max_extensions=3
)
```

- **Dynamic Discovery**: Auto-discover and register tools from MCP servers
- **Type Safety**: Full Pydantic validation for tool interactions
- **Connection Pooling**: Efficient resource management with automatic reuse

### Observability & Debugging

- Inspect messages:

```python
df = branch.to_df()
print(df.tail())
```

- Action logs show each tool call, arguments, and outcomes.
- Verbose ReAct provides chain-of-thought analysis (helpful for debugging
  multi-step flows).

### Example: Multi-Model Orchestration

```python
from lionagi import Branch, iModel

# Define models for multi-model orchestration
gpt4o = iModel(provider="openai", model="gpt-4o-mini")
sonnet = iModel(
  provider="anthropic",
  model="claude-3-5-sonnet-20241022",
  max_tokens=1000,                    # max_tokens is required for anthropic models
)

branch = Branch(chat_model=gpt4o)
analysis = await branch.communicate("Analyze these stats", chat_model=sonnet) # Switch mid-flow
```

Seamlessly route to different models in the same workflow.

### CLI Agent Integration

LionAGI integrates with coding agent CLIs as providers, enabling multi-agent orchestration across models:

| Provider | CLI | Models |
|----------|-----|--------|
| `claude_code` | [Claude Code](https://docs.anthropic.com/en/docs/claude-code/sdk) | sonnet, opus, haiku |
| `codex` | [OpenAI Codex](https://github.com/openai/codex) | gpt-5.3-codex-spark, gpt-5.4 |
| `gemini_code` | [Gemini CLI](https://github.com/google-gemini/gemini-cli) | gemini-3.1-* (unstable) |

```python
from lionagi import iModel, Branch

# Use any CLI agent as a model
agent = Branch(chat_model=iModel(provider="claude_code", model="sonnet"))
response = await agent.communicate("Explain the architecture of this codebase")

# Switch providers mid-flow
codex = iModel(provider="codex", model="gpt-5.3-codex-spark")
response2 = await agent.communicate("Compare with your analysis", chat_model=codex)
```

See the [CLI Guide](docs/cli_guide.md) for the `li` command-line tool that wraps these providers with fan-out orchestration, session persistence, and effort control.

### CLI — `li`

LionAGI ships a command-line tool `li` for spawning agents and orchestrating multi-agent fan-out patterns directly from your terminal. See the full [CLI Guide](docs/cli_guide.md) for details.

```bash
# Single agent
li agent claude/sonnet "Explain the observer pattern"
li agent codex/gpt-5.3-codex-spark "Review this function for bugs" --yolo

# Fan-out: orchestrator decomposes task, N workers run in parallel, optional synthesis
li o fanout claude/sonnet "What are the key design patterns in this codebase?" -n 3 --with-synthesis

# Heterogeneous workers + different synthesis model
li o fanout claude/sonnet "Analyze error handling approaches" \
    --workers "claude/sonnet, codex/gpt-5.3-codex-spark" \
    --with-synthesis claude/opus-4-7-high

# Resume any conversation
li agent -r <branch-id> "follow up on your analysis"
```

### optional dependencies

```
"lionagi[reader]" - Reader tool for any unstructured data and web pages
"lionagi[ollama]" - Ollama model support for local inference
"lionagi[rich]" - Rich output formatting for better console display
"lionagi[schema]" - Convert pydantic schema to make the Model class persistent
"lionagi[postgres]" - Postgres database support for storing and retrieving structured data
"lionagi[graph]" - Graph display for visualizing complex workflows
"lionagi[sqlite]" - SQLite database support for lightweight data storage (also need `postgres` option)
```

## Community & Contributing

We welcome issues, ideas, and pull requests:

- Discord: Join to chat or get help
- Issues / PRs: GitHub

### Citation

```
@software{Li_LionAGI_2023,
  author = {Haiyang Li},
  month = {12},
  year = {2023},
  title = {LionAGI: Towards Automated General Intelligence},
  url = {https://github.com/lion-agi/lionagi},
}
```

**🦁 LionAGI**

> Because real AI orchestration demands more than a single prompt. Try it out
> and discover the next evolution in structured, multi-model, safe AI.
