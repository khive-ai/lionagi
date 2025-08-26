# Vector Store Integration

Integrate LionAGI with vector databases for RAG, memory, and semantic search.

## Basic Vector Store Pattern

```python
from lionagi import Branch, iModel
import numpy as np

# Simple vector store interface
class VectorStore:
    def __init__(self):
        self.vectors = {}
        self.metadata = {}
    
    def add(self, doc_id: str, vector: np.ndarray, metadata: dict):
        self.vectors[doc_id] = vector
        self.metadata[doc_id] = metadata
    
    def search(self, query_vector: np.ndarray, top_k: int = 5):
        # Cosine similarity search
        similarities = {}
        for doc_id, vector in self.vectors.items():
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities[doc_id] = similarity
        
        # Return top-k results
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(doc_id, self.metadata[doc_id], score) for doc_id, score in sorted_results]

# Vector search tool for agents
def vector_search(query: str, top_k: int = 3) -> str:
    """Search vector store for relevant documents"""
    # Mock implementation - replace with actual embedding + search
    return f"Found {top_k} relevant documents for: {query}"

# RAG-enabled branch
rag_branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    tools=[vector_search],
    system="You have access to a vector database. Use vector_search to find relevant information before answering."
)

# Usage
response = await rag_branch.ReAct(
    instruct={"instruction": "What are the latest developments in AI safety?"},
    max_extensions=2
)
```

## Qdrant Integration

```python
import asyncio
from typing import List, Dict, Any

class QdrantAdapter:
    """Qdrant vector database adapter for LionAGI"""
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.host = host
        self.port = port
        # Initialize Qdrant client here
    
    async def create_collection(self, collection_name: str, vector_size: int = 1536):
        """Create Qdrant collection"""
        # Implementation would create actual Qdrant collection
        print(f"Created collection: {collection_name}")
    
    async def add_documents(self, collection: str, documents: List[Dict[str, Any]]):
        """Add documents with embeddings to collection"""
        # Implementation would add to Qdrant
        return f"Added {len(documents)} documents to {collection}"
    
    async def search(self, collection: str, query_vector: List[float], limit: int = 5):
        """Vector similarity search"""
        # Mock results - replace with actual Qdrant search
        return [
            {"id": "doc1", "score": 0.95, "payload": {"text": "Sample document 1"}},
            {"id": "doc2", "score": 0.87, "payload": {"text": "Sample document 2"}},
        ]

# Qdrant-powered research agent
async def create_qdrant_researcher():
    """Create research agent with Qdrant vector search"""
    
    qdrant = QdrantAdapter()
    
    async def qdrant_search(query: str) -> str:
        """Qdrant search tool for agents"""
        # Generate embedding for query (mock)
        query_vector = [0.1] * 1536  # Replace with actual embedding
        
        results = await qdrant.search("research_papers", query_vector)
        
        # Format results for agent
        formatted = "\n".join([
            f"- {r['payload']['text']} (score: {r['score']:.2f})"
            for r in results
        ])
        
        return f"Vector search results:\n{formatted}"
    
    researcher = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        tools=[qdrant_search],
        system="Research specialist with access to Qdrant vector database"
    )
    
    return researcher

# Usage
researcher = await create_qdrant_researcher()
result = await researcher.ReAct(
    instruct={"instruction": "Find papers on transformer architectures"},
    max_extensions=2
)
```

## ChromaDB Local Vector Store

```python
class ChromaAdapter:
    """ChromaDB adapter for local vector storage"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        # Initialize ChromaDB client
    
    def add_texts(self, collection_name: str, texts: List[str], metadatas: List[Dict] = None):
        """Add texts to ChromaDB collection"""
        # Implementation would use ChromaDB
        return f"Added {len(texts)} texts to {collection_name}"
    
    def similarity_search(self, collection_name: str, query: str, k: int = 5):
        """Search for similar texts"""
        # Mock results
        return [
            {"text": f"Result {i}", "metadata": {"source": f"doc_{i}"}, "distance": 0.1 * i}
            for i in range(k)
        ]

# Local RAG with ChromaDB
async def local_rag_workflow(documents: List[str], query: str):
    """Local RAG workflow using ChromaDB"""
    
    chroma = ChromaAdapter()
    
    # Add documents to vector store
    chroma.add_texts("knowledge_base", documents)
    
    # Create search tool
    def search_knowledge(query: str) -> str:
        results = chroma.similarity_search("knowledge_base", query, k=3)
        return "\n".join([f"- {r['text']}" for r in results])
    
    # Create RAG agent
    rag_agent = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        tools=[search_knowledge],
        system="Answer questions using the knowledge base. Always search first."
    )
    
    # Query with RAG
    response = await rag_agent.ReAct(
        instruct={"instruction": query},
        max_extensions=2
    )
    
    return response

# Usage
docs = [
    "LionAGI is a graph-based multi-agent framework",
    "Branches represent individual AI agents with memory",
    "Builder pattern creates complex agent workflows"
]
answer = await local_rag_workflow(docs, "How does LionAGI work?")
```

## Pinecone Cloud Integration

```python
class PineconeAdapter:
    """Pinecone cloud vector database adapter"""
    
    def __init__(self, api_key: str, environment: str):
        self.api_key = api_key
        self.environment = environment
        # Initialize Pinecone client
    
    async def upsert_vectors(self, index_name: str, vectors: List[Dict]):
        """Upload vectors to Pinecone index"""
        # Implementation would use Pinecone API
        return f"Upserted {len(vectors)} vectors to {index_name}"
    
    async def query(self, index_name: str, vector: List[float], top_k: int = 5):
        """Query Pinecone index"""
        # Mock results
        return {
            "matches": [
                {"id": f"vec_{i}", "score": 0.9 - i*0.1, "metadata": {"text": f"Result {i}"}}
                for i in range(top_k)
            ]
        }

# Production RAG with Pinecone
async def production_rag_system():
    """Production-ready RAG system with Pinecone"""
    
    pinecone = PineconeAdapter("your-api-key", "us-west1-gcp")
    
    async def pinecone_search(query: str) -> str:
        """Production vector search"""
        # Generate embedding (mock)
        query_vector = [0.1] * 1536
        
        results = await pinecone.query("production-index", query_vector)
        
        context = "\n".join([
            f"- {match['metadata']['text']} (relevance: {match['score']:.2f})"
            for match in results["matches"]
        ])
        
        return f"Relevant context:\n{context}"
    
    # Production RAG agent
    production_agent = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        tools=[pinecone_search],
        system="Production AI assistant with cloud vector search capabilities"
    )
    
    return production_agent

# Usage
agent = await production_rag_system()
response = await agent.ReAct(
    instruct={"instruction": "Explain quantum computing applications"},
    max_extensions=3
)
```

## Memory-Enhanced Agents

```python
from lionagi import types

class AgentMemory:
    """Vector-based agent memory system"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.conversation_history = []
    
    async def remember(self, content: str, metadata: dict = None):
        """Store memory with vector embedding"""
        # Generate embedding and store
        vector = self.generate_embedding(content)
        memory_id = f"memory_{len(self.conversation_history)}"
        
        self.vector_store.add(memory_id, vector, {
            "content": content,
            "timestamp": metadata.get("timestamp"),
            "type": metadata.get("type", "conversation")
        })
        
        self.conversation_history.append(memory_id)
    
    async def recall(self, query: str, k: int = 3):
        """Recall relevant memories"""
        query_vector = self.generate_embedding(query)
        results = self.vector_store.search(query_vector, k)
        
        return [
            {"content": r[1]["content"], "relevance": r[2]}
            for r in results
        ]
    
    def generate_embedding(self, text: str):
        """Generate embedding for text"""
        # Mock embedding - replace with actual model
        return np.random.random(1536)

# Memory-enhanced agent
async def create_memory_agent():
    """Agent with vector-based long-term memory"""
    
    vector_store = VectorStore()
    memory = AgentMemory(vector_store)
    
    async def remember_conversation(content: str) -> str:
        await memory.remember(content, {"type": "conversation"})
        return f"Remembered: {content[:50]}..."
    
    async def recall_relevant(query: str) -> str:
        memories = await memory.recall(query)
        if memories:
            return "Relevant memories:\n" + "\n".join([
                f"- {m['content'][:100]}... (relevance: {m['relevance']:.2f})"
                for m in memories
            ])
        return "No relevant memories found"
    
    memory_agent = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        tools=[remember_conversation, recall_relevant],
        system="AI assistant with long-term vector memory. Remember important information and recall when relevant."
    )
    
    return memory_agent

# Usage
agent = await create_memory_agent()
await agent.communicate("I prefer technical explanations over simplified ones")
response = await agent.communicate("Explain machine learning")  # Will recall preference
```

## Choosing Vector Stores

**Local Development:**

- ChromaDB: Easy setup, good for prototyping
- FAISS: High performance, research-grade
- In-memory: Simple testing and demos

**Production Cloud:**

- Pinecone: Managed service, great scaling
- Weaviate: Open source, GraphQL API
- Qdrant: Rust-based, high performance

**Hybrid Approaches:**

- Local + Cloud: Development locally, production in cloud
- Multi-store: Different stores for different use cases
- Fallback: Local backup when cloud unavailable
