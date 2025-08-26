# ReAct with RAG Pattern

Tool-augmented reasoning with retrieval for knowledge-intensive tasks.

## When to Use This Pattern

Use ReAct with RAG when:

- Tasks require external knowledge or data
- Multi-step reasoning is needed
- Information must be gathered and synthesized
- Complex problem-solving requires both thinking and acting

## Pattern Structure

ReAct cycles through: **Reason** → **Act** → **Observe** → **Repeat**

## Basic Implementation

```python
from lionagi import Branch, Session, Builder, iModel

def search_knowledge(query: str) -> dict:
    """Your knowledge retrieval function"""
    return {"query": query, "results": [...]}

# ReAct-enabled Branch
researcher = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-sonnet-20240229"),
    system="Research and reason step by step using available tools.",
    tools=[search_knowledge]
)

session = Session()
builder = Builder()

# ReAct operation
task = builder.add_operation(
    "ReAct",
    branch=researcher,
    instruct={
        "instruction": "Research neural networks in machine learning",
        "context": "Provide comprehensive analysis with specific details"
    },
    max_extensions=3  # Reasoning steps limit
)

result = await session.flow(builder.get_graph())
```

## Multi-Tool ReAct

```python
def search_papers(query: str) -> dict:
    """Search academic sources"""
    return {"source": "papers", "results": [...]}

def search_docs(query: str) -> dict:
    """Search documentation"""
    return {"source": "docs", "results": [...]}

# Multi-tool researcher
researcher = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-opus-20240229"),
    system="Expert researcher with access to multiple knowledge sources.",
    tools=[search_papers, search_docs]
)

# Complex research task
task = builder.add_operation(
    "ReAct",
    branch=researcher,
    instruct={
        "instruction": "Compare transformer architectures and performance metrics",
        "guidance": "Use academic sources, then analyze technical documentation"
    },
    max_extensions=5,
    reason=True  # Include reasoning in output
)
```

## File-Based RAG

```python
from lionagi.tools.file.reader import ReaderTool

# Document analyst
analyst = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-sonnet-20240229"),
    system="Analyze documents systematically, extracting key information.",
    tools=[ReaderTool]
)

# Document analysis task
analysis = builder.add_operation(
    "ReAct",
    branch=analyst,
    instruct={
        "instruction": "Analyze project documentation for core architecture",
        "guidance": "Start with README, then extract key technical details"
    },
    max_extensions=4
)
```

## Key Patterns

### Tool Selection Strategy

```python
# Phase-based tool usage
information_gathering = ["search", "retrieve", "extract"]
analysis_phase = ["analyze", "calculate", "compare"] 
synthesis_phase = ["verify", "synthesize", "conclude"]
```

### Reasoning Control

```python
instruct = {
    "instruction": "Clear task definition",
    "guidance": "Step-by-step process guidance", 
    "context": "Background for decision making"
}
```

### Performance Limits

```python
max_extensions=5  # Prevent infinite reasoning loops
reason=True      # Include reasoning traces
```

## Best Practices

- **Tool Design**: Return structured data with consistent formats
- **Reasoning Guidance**: Provide clear step-by-step instructions
- **Context Management**: Keep context focused and relevant
- **Error Handling**: Set reasonable limits on reasoning steps

ReAct with RAG enables systematic information gathering and reasoning for
complex, knowledge-intensive tasks.
