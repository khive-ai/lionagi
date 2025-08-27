# Database Integration

Persisting agent state and data across sessions.

## PostgreSQL

### Setup

Install PostgreSQL dependencies:

```bash
uv add lionagi[postgres]
```

### Basic Usage

```python
# Note: Database integration requires custom implementation
# This is a conceptual example - actual implementation varies
from pydapter.extras.async_postgres_ import AsyncPostgresAdapter
from lionagi import Branch, iModel
import asyncio

async def main():
    # Database connection using pydapter
    adapter = AsyncPostgresAdapter(
        host="localhost",
        port=5432,
        database="lionagi_db",
        username="your_user",
        password="your_password"
    )
    
    # Create branch with model
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    
    # Chat and manually persist if needed
    result = await branch.communicate("Hello, I'm testing database integration")
    print(result)
    
    # Custom persistence logic would go here
    # await adapter.insert("conversations", {"message": result, "timestamp": datetime.now()})

asyncio.run(main())
```

### Advanced Configuration

```python
from pydapter import PostgresAdapter
from lionagi.models import Note
from datetime import datetime

# Custom schema for agent conversations
class ConversationLog(Note):
    session_id: str
    timestamp: datetime
    user_input: str
    agent_response: str
    model_used: str
    token_count: int

# Database operations
async def store_conversation(adapter, log: ConversationLog):
    await adapter.insert(log)

async def retrieve_session_history(adapter, session_id: str):
    query = "SELECT * FROM conversation_log WHERE session_id = $1 ORDER BY timestamp"
    return await adapter.select(query, session_id)
```

### Troubleshooting

- **Connection Issues**: Verify PostgreSQL is running: `pg_isready -h localhost`
- **Permission Denied**: Check user permissions:
  `GRANT ALL ON DATABASE lionagi_db TO your_user;`
- **SSL Errors**: Add `sslmode=require` to connection string for cloud databases

## SQLite

### Setup

Install SQLite dependencies:

```bash
uv add lionagi[sqlite]
```

### Basic Usage

```python
# Note: Automatic persistence requires custom implementation
from lionagi import Branch, iModel
import aiosqlite
import asyncio
from datetime import datetime

async def main():
    # Create branch with SQLite for manual persistence
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    
    # Chat without automatic persistence
    response = await branch.communicate(
        "Remember this: I prefer technical explanations."
    )
    
    # Manual persistence example
    async with aiosqlite.connect("lionagi_sessions.db") as db:
        await db.execute(
            "INSERT INTO conversations (message, timestamp) VALUES (?, ?)",
            (str(response), datetime.now())
        )
        await db.commit()
    
    print(response)

asyncio.run(main())
```

### Custom SQLite Operations

```python
import aiosqlite
from lionagi.models import FieldModel
from datetime import datetime

class AgentMemory(FieldModel):
    memory_id: str
    content: str
    importance: float
    created_at: datetime
    accessed_count: int = 0

async def setup_memory_database():
    async with aiosqlite.connect("agent_memory.db") as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                importance REAL,
                created_at TIMESTAMP,
                accessed_count INTEGER DEFAULT 0
            )
        """)
        await db.commit()

async def store_memory(memory: AgentMemory):
    async with aiosqlite.connect("agent_memory.db") as db:
        await db.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?)",
            (memory.memory_id, memory.content, memory.importance, 
             memory.created_at, memory.accessed_count)
        )
        await db.commit()

async def retrieve_important_memories(threshold: float = 0.8):
    async with aiosqlite.connect("agent_memory.db") as db:
        async with db.execute(
            "SELECT * FROM memories WHERE importance >= ? ORDER BY importance DESC",
            (threshold,)
        ) as cursor:
            return await cursor.fetchall()
```

### Troubleshooting

- **Database Locked**: Ensure proper connection closing with `async with`
- **No such table**: Run schema creation before operations
- **Disk Full**: Check available disk space: `df -h`

## Neo4j

### Setup

```bash
# Install Neo4j driver
uv add neo4j
```

### Basic Usage

```python
from neo4j import AsyncGraphDatabase
from lionagi.session import Session
from lionagi.models import FieldModel
from typing import List

class AgentRelationship(FieldModel):
    from_agent: str
    to_agent: str
    relationship_type: str
    strength: float
    created_at: str

class Neo4jAgentGraph:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    
    async def close(self):
        await self.driver.close()
    
    async def create_agent_node(self, agent_id: str, properties: dict):
        async with self.driver.session() as session:
            await session.run(
                "CREATE (a:Agent {id: $agent_id, name: $name, type: $type})",
                agent_id=agent_id, **properties
            )
    
    async def create_relationship(self, rel: AgentRelationship):
        async with self.driver.session() as session:
            await session.run("""
                MATCH (a:Agent {id: $from_agent}), (b:Agent {id: $to_agent})
                CREATE (a)-[:RELATES {type: $rel_type, strength: $strength, created_at: $created_at}]->(b)
            """, 
                from_agent=rel.from_agent, 
                to_agent=rel.to_agent,
                rel_type=rel.relationship_type,
                strength=rel.strength,
                created_at=rel.created_at
            )
    
    async def find_connected_agents(self, agent_id: str) -> List[dict]:
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (a:Agent {id: $agent_id})-[r:RELATES]-(b:Agent)
                RETURN b.id as connected_agent, r.type as relationship, r.strength as strength
                ORDER BY r.strength DESC
            """, agent_id=agent_id)
            return [record.data() for record in await result.consume()]

# Usage example
async def main():
    graph = Neo4jAgentGraph("bolt://localhost:7687", "neo4j", "password")
    
    # Create agent nodes
    await graph.create_agent_node("researcher_001", {
        "name": "Research Agent",
        "type": "researcher"
    })
    
    await graph.create_agent_node("writer_001", {
        "name": "Writer Agent", 
        "type": "writer"
    })
    
    # Create relationship
    rel = AgentRelationship(
        from_agent="researcher_001",
        to_agent="writer_001",
        relationship_type="PROVIDES_DATA",
        strength=0.9,
        created_at="2025-01-01T12:00:00Z"
    )
    await graph.create_relationship(rel)
    
    # Find connections
    connections = await graph.find_connected_agents("researcher_001")
    print(connections)
    
    await graph.close()
```

### Troubleshooting

- **Connection Refused**: Check Neo4j service: `systemctl status neo4j`
- **Authentication Failed**: Verify username/password in Neo4j browser
- **Cypher Query Errors**: Test queries in Neo4j browser first

## MongoDB

### Setup

```bash
uv add motor  # Async MongoDB driver
```

### Basic Usage

```python
from motor.motor_asyncio import AsyncIOMotorClient
from lionagi.models import FieldModel
from lionagi.session import Session
from datetime import datetime
from typing import Optional, List

class AgentDocument(FieldModel):
    agent_id: str
    session_id: str
    messages: List[dict]
    metadata: dict
    created_at: datetime
    updated_at: datetime

class MongoAgentStore:
    def __init__(self, connection_string: str, database_name: str):
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client[database_name]
        self.agents = self.db.agents
        self.sessions = self.db.sessions
    
    async def close(self):
        self.client.close()
    
    async def store_agent_document(self, doc: AgentDocument):
        result = await self.agents.replace_one(
            {"agent_id": doc.agent_id, "session_id": doc.session_id},
            doc.model_dump(),
            upsert=True
        )
        return str(result.upserted_id) if result.upserted_id else None
    
    async def get_agent_sessions(self, agent_id: str) -> List[AgentDocument]:
        cursor = self.agents.find({"agent_id": agent_id})
        docs = await cursor.to_list(length=100)
        return [AgentDocument(**doc) for doc in docs]
    
    async def search_by_content(self, query: str) -> List[AgentDocument]:
        # Text search across message content
        cursor = self.agents.find({
            "$text": {"$search": query}
        })
        docs = await cursor.to_list(length=50)
        return [AgentDocument(**doc) for doc in docs]

# Usage with Branch instead of fabricated Session API
async def main():
    store = MongoAgentStore(
        "mongodb://localhost:27017",
        "lionagi_agents"
    )
    
    # Create text index for search
    await store.agents.create_index([
        ("messages.content", "text"),
        ("metadata.tags", "text")
    ])
    
    # Use with branch
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    
    response = await branch.communicate("Explain quantum computing briefly")
    
    # Extract messages manually for storage
    messages = []
    for msg in getattr(branch, 'messages', []):
        messages.append({
            'content': str(msg),
            'timestamp': datetime.now().isoformat()
        })
    
    # Store branch data
    doc = AgentDocument(
        agent_id="quantum_expert",
        session_id=f"session_{datetime.now().timestamp()}",
        messages=messages,
        metadata={"topic": "quantum_computing", "complexity": "beginner"},
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    await store.store_agent_document(doc)
    
    # Search previous sessions
    quantum_sessions = await store.search_by_content("quantum")
    print(f"Found {len(quantum_sessions)} sessions about quantum topics")
    
    await store.close()
```

### Troubleshooting

- **Connection Timeout**: Check MongoDB service: `systemctl status mongod`
- **Authentication Failed**: Verify connection string format
- **Index Errors**: Ensure unique index constraints are met

## Database Patterns

### Session State Management

```python
# Custom session state management example
from lionagi import Branch
from lionagi.models import FieldModel
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

class BranchState(FieldModel):
    branch_id: str
    system_prompt: str
    message_history: List[dict]
    metadata: Dict[str, Any]
    last_updated: datetime

class DatabaseBranchManager:
    def __init__(self, adapter):
        self.adapter = adapter
    
    async def save_branch(self, branch: Branch, branch_id: str):
        # Extract messages manually (example structure)
        messages = []
        for msg in getattr(branch, 'messages', []):
            messages.append({
                'role': getattr(msg, 'role', 'unknown'),
                'content': getattr(msg, 'content', str(msg))
            })
        
        state = BranchState(
            branch_id=branch_id,
            system_prompt=getattr(branch, 'system', ''),
            message_history=messages,
            metadata={'model': str(getattr(branch, 'chat_model', None))},
            last_updated=datetime.now()
        )
        
        await self.adapter.upsert("branch_states", state.model_dump())
    
    async def load_branch_data(self, branch_id: str) -> Optional[BranchState]:
        state_data = await self.adapter.get("branch_states", branch_id)
        if not state_data:
            return None
            
        return BranchState(**state_data)
    
    async def cleanup_old_branches(self, days_old: int = 30):
        cutoff_date = datetime.now() - timedelta(days=days_old)
        await self.adapter.delete_where(
            "branch_states", 
            "last_updated < %s", 
            cutoff_date
        )
```

### Agent Knowledge Base Schema

```python
class Knowledge(FieldModel):
    knowledge_id: str
    domain: str  # e.g., "programming", "science", "history"
    content: str
    confidence: float  # 0.0 to 1.0
    source: str
    validated: bool = False
    created_by: str  # agent_id
    created_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None

class AgentKnowledgeBase:
    async def add_knowledge(self, knowledge: Knowledge):
        # Validate and store new knowledge
        if knowledge.confidence < 0.7:
            knowledge.validated = False
        
        await self.adapter.insert("knowledge", knowledge.model_dump())
    
    async def query_knowledge(self, domain: str, query: str, limit: int = 10):
        # Semantic search within domain
        results = await self.adapter.semantic_search(
            table="knowledge",
            query_text=query,
            filters={"domain": domain, "validated": True},
            limit=limit
        )
        
        # Update access statistics
        for result in results:
            await self.adapter.increment(
                "knowledge", 
                result["knowledge_id"], 
                "access_count"
            )
        
        return results
```

### Performance Optimization

```python
# Connection pooling for high-throughput scenarios  
# Note: Use pydapter for actual PostgreSQL connections
from pydapter.extras.async_postgres_ import AsyncPostgresAdapter
import asyncpg

class OptimizedDatabaseManager:
    def __init__(self, connection_string: str, pool_size: int = 20):
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.pool = None
    
    async def initialize(self):
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=5,
            max_size=self.pool_size,
            command_timeout=60
        )
    
    async def batch_insert(self, table: str, records: List[dict]):
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Efficient batch insert
                await conn.executemany(
                    f"INSERT INTO {table} VALUES ($1, $2, $3, $4)",
                    [(r['id'], r['data'], r['timestamp'], r['agent_id']) 
                     for r in records]
                )
    
    async def close(self):
        if self.pool:
            await self.pool.close()
```

### Best Practices

1. **Use Connection Pooling**: For production environments with multiple agents
2. **Implement Proper Indexing**: Create indexes on frequently queried fields
3. **Regular Cleanup**: Implement automated cleanup of old sessions and logs
4. **Backup Strategy**: Regular backups for persistent agent knowledge
5. **Monitor Performance**: Track query performance and optimize slow queries
6. **Handle Failures**: Implement retry logic for database operations
7. **Data Validation**: Validate data before storing in the database
8. **Concurrency Control**: Use transactions for operations that modify multiple
   records
