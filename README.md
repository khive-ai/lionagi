![PyPI - Version](https://img.shields.io/pypi/v/lionagi?labelColor=233476aa&color=231fc935)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lionagi?color=blue)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)

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

res = await hunter.communicate(
    "Tell me a short dragon joke",
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

### Claude Code Integration

LionAGI now supports Anthropic's Claude Code [Python SDK](https://github.com/anthropics/claude-code-sdk-python), and [CLI SDK](https://docs.anthropic.com/en/docs/claude-code/sdk) enabling autonomous coding capabilities with persistent session management. The CLI endpoint
directly connects to claude code, and is recommended, you can either use it via a [proxy server](https://github.com/khive-ai/lionagi/tree/main/cookbooks/claude_proxy) or directly with `query_cli` endpoint, provided you have already logged onto claude code cli in your terminal.

```python
from lionagi import iModel, Branch

def create_cc_model():
  return iModel(
      provider="claude_code",
      endpoint="query_cli",
      model="sonnet",
      verbose_output=True,  # Enable detailed output for debugging
  )

# Start a coding session
orchestrator = Branch(chat_model=create_cc_model())
response = await orchestrator.communicate("Explain the architecture of protocols, operations, and branch")

# continue the session with more queries
response2 = await orchestrator.communicate("how do these parts form lionagi system")
```

### Fan out fan in pattern orchestration with claude code

```python
# use structured outputs with claude code
from lionagi.fields import LIST_INSTRUCT_FIELD_MODEL, Instruct

response3 = await orchestrator.operate(
  instruct=Instruct(
    instruction="create 4 research questions for parallel discovery",
    guidance="put into `instruct_models` field as part of your structured result message",
    context="I'd like to create an orchestration system for AI agents using lionagi"
  ),
  field_models=[LIST_INSTRUCT_FIELD_MODEL],
)

len(response3.instruct_models)  # should be 4

async def handle_instruct(instruct):
  sub_branch = Branch(
    system="You are an diligent research expert.",
    chat_model=create_cc_model(),
  )
  return await sub_branch.operate(instruct=instruct)

# run in parallel across all instruct models
from lionagi.utils import alcall
responses = await alcall(response3.instruct_models, handle_instruct)

# now hand these reports back to the orchestrator
final_response = await orchestrator.communicate(
  "please synthesize these research findings into a final report",
  context=responses,
)
```

Key features:
- **Auto-Resume Sessions**: Conversations automatically continue from where they left off
- **Tool Permissions**: Fine-grained control over which tools Claude can access
- **Streaming Support**: Real-time feedback during code generation
- **Seamless Integration**: Works with existing LionAGI workflows

### optional dependencies

```
"lionagi[reader]" - Reader tool for any unstructured data and web pages
"lionagi[ollama]" - Ollama model support for local inference
"lionagi[claude-code]" - Claude code python SDK integration (cli endpoint does not require this)
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
