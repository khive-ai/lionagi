# Integrations

!!! info "For Production Systems"
    This section helps you connect LionAGI to real-world systems - databases, APIs, tools, and other AI frameworks.

Connect LionAGI with external services and frameworks to build comprehensive AI systems.

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

LionAGI's key advantage: **orchestrate any existing AI framework** without code
changes.

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

- **Wrapper Operations**: Embed external tools as LionAGI operations
- **Multi-Framework**: Coordinate multiple frameworks in single workflow
- **Gradual Migration**: Keep existing code while gaining orchestration benefits

## When You Need Integrations

!!! success "Use Integrations When:"
    - **Persistent data**: Need to store workflow results in databases
    - **External knowledge**: RAG systems with vector stores and knowledge bases
    - **Tool augmentation**: Agents need access to APIs, calculators, or specialized services
    - **Framework combination**: Want to orchestrate existing LangChain/CrewAI workflows
    - **Production deployment**: Need monitoring, logging, and enterprise infrastructure

## Getting Started

!!! tip "Integration Strategy"
    **Start simple**: Begin with [LLM Providers](llm-providers.md) and [Tools](tools.md)  
    **Add persistence**: Connect [Databases](databases.md) for workflow state  
    **Scale up**: Add [Vector Stores](vector-stores.md) for knowledge-intensive workflows  
    **Orchestrate**: Integrate existing frameworks with [meta-orchestration patterns](#meta-orchestration)
