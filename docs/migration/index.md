# Migration Guides

Migrate from other AI frameworks to LionAGI.

## Available Migration Guides

- **[From LangChain](from-langchain.md)** - Migrate LangChain workflows to
  LionAGI
- **[From CrewAI](from-crewai.md)** - Convert CrewAI crews to LionAGI workflows
- **[From AutoGen](from-autogen.md)** - Adapt AutoGen conversations for LionAGI

## Migration Philosophy

**Zero-Disruption Migration**: Keep your existing code and gradually adopt
LionAGI orchestration.

```python
# Your existing workflow runs unchanged
async def existing_workflow(input_data):
    return await your_current_implementation(input_data)

# Orchestrate with LionAGI
builder.add_operation(operation=existing_workflow)
```

## Migration Strategies

### **Gradual Adoption**

1. Wrap existing workflows as custom operations
2. Add LionAGI orchestration around them
3. Gradually convert individual components
4. Gain orchestration benefits without disruption

### **Hybrid Workflows**

- Mix existing framework code with native LionAGI operations
- Coordinate multiple frameworks in single workflow
- Best of all worlds approach

### **Full Migration**

- Translate framework patterns to LionAGI equivalents
- Leverage LionAGI's superior orchestration capabilities
- Gain performance and simplicity benefits

## Why Migrate?

- **Parallel Execution**: LionAGI runs operations concurrently by default
- **Simpler Code**: Less boilerplate, cleaner abstractions
- **Framework Agnostic**: Orchestrate any tool, not just one ecosystem
- **Production Ready**: Built-in monitoring, error handling, performance control
