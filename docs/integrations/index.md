# Integrations

Connect lionagi with external services, tools, and frameworks.

## Core

- **[LLM Providers](llm-providers.md)** -- OpenAI, Anthropic, Gemini, Ollama, and more
- **[Tools](tools.md)** -- Turn any Python function into an LLM-callable tool
- **[MCP Servers](mcp-servers.md)** -- Connect to Model Context Protocol tool servers

## Data

- **[Databases](databases.md)** -- Serializing Branch state; pydapter for storage adapters
- **[Vector Stores](vector-stores.md)** -- Using embeddings and vector search with tools

## External Frameworks

lionagi does not have native integrations with other AI frameworks, but you can wrap any framework's functionality as a lionagi tool:

- **[DSPy](dspy-optimization.md)** -- Wrap DSPy modules as tools
- **[LlamaIndex](llamaindex-rag.md)** -- Wrap LlamaIndex query engines as tools

## Integration Pattern

Any Python function (sync or async) can become a tool:

```python
branch = Branch(tools=[your_function])
result = await branch.operate(instruction="...", actions=True)
```

This works with any external library -- database clients, HTTP APIs, vector stores, or other AI frameworks. See [Tools](tools.md) for details.
