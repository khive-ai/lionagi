# Vector Stores

lionagi does not include built-in vector store integration. However, its data primitives support embedding vectors, and you can connect any vector store through tools.

## Node Embeddings

The `Node` class (used throughout lionagi's graph system) has an optional `embedding` field:

```python
from lionagi.protocols.graph.node import Node

node = Node(
    content="Some text to embed",
    embedding=[0.1, 0.2, 0.3, ...],  # Your embedding vector
)
```

This lets you store embeddings alongside content in lionagi's graph structures, but querying and indexing are left to your vector store of choice.

## Connecting a Vector Store as a Tool

The recommended approach is to wrap your vector store operations as tool functions:

```python
from lionagi import Branch

async def vector_search(query: str, top_k: int = 5) -> str:
    """Search the vector database for relevant documents.

    Args:
        query: The search query.
        top_k: Number of results to return.
    """
    # Your vector store client (Qdrant, Pinecone, Chroma, etc.)
    results = await your_vector_client.search(query, limit=top_k)
    return "\n".join(r.text for r in results)

branch = Branch(
    tools=[vector_search],
    system="Search the knowledge base before answering questions."
)

result = await branch.ReAct(
    instruct={"instruction": "What are the latest findings on climate change?"},
    max_extensions=2,
)
```

This approach works with any vector store and keeps lionagi decoupled from specific database choices.

## Related

- [pydapter](https://github.com/khive-ai/pydapter) includes a Qdrant adapter for vector storage
- [Tools](tools.md) covers how to create and register custom tools
