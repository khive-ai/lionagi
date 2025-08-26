# Integrations

Connect LionAGI with external services and frameworks.

## Available Integrations

### **Core Infrastructure**
- **[LLM Providers](llm-providers.md)** - OpenAI, Anthropic, local models
- **[Vector Stores](vector-stores.md)** - Qdrant, Pinecone, Chroma
- **[Databases](databases.md)** - PostgreSQL, MongoDB, Redis
- **[Tools](tools.md)** - External APIs and services  
- **[MCP Servers](mcp-servers.md)** - Model Context Protocol integration

### **AI Framework Orchestration**
- **[LlamaIndex RAG](llamaindex-rag.md)** - Wrap RAG capabilities as operations
- **[DSPy Optimization](dspy-optimization.md)** - Embed prompt optimization

## Meta-Orchestration

LionAGI's key advantage: **orchestrate any existing AI framework** without code changes.

```python
# Your existing framework code runs unchanged
builder.add_operation(operation=your_existing_workflow)
```

This works with:
- LangChain workflows
- CrewAI crews  
- AutoGen conversations
- Custom Python functions
- External APIs

## Integration Patterns

**Wrapper Operations**: Embed external tools as LionAGI operations
**Multi-Framework**: Coordinate multiple frameworks in single workflow  
**Gradual Migration**: Keep existing code while gaining orchestration benefits