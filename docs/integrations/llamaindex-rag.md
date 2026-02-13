# LlamaIndex Integration

lionagi does not have a native LlamaIndex integration. The two frameworks serve different purposes: LlamaIndex focuses on RAG pipelines and document indexing, while lionagi focuses on multi-model orchestration and tool calling.

## Using LlamaIndex with lionagi

You can wrap LlamaIndex query engines as lionagi tools:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from lionagi import Branch

# Your LlamaIndex setup
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Wrap as a lionagi tool
def rag_search(question: str) -> str:
    """Search the document index for relevant information.

    Args:
        question: The question to search for.
    """
    response = query_engine.query(question)
    return str(response)

branch = Branch(tools=[rag_search])

result = await branch.ReAct(
    instruct={"instruction": "What are the key findings in the research papers?"},
    max_extensions=2,
)
```

Both frameworks can share the same LLM API keys and run in the same process.
