# Data Persistence

Direct Node-to-database patterns with automatic schema creation.

## Basic PostgreSQL Setup

```python
# Install lionagi with postgres support
# uv add "lionagi[postgres]"

from pydantic import BaseModel
from typing import Literal
from lionagi import types
from lionagi.adapters.async_postgres_adapter import LionAGIAsyncPostgresAdapter

# Connection string format
dsn = "postgresql+asyncpg://postgres:postgres@127.0.0.1:54322/postgres"

# Define your data model
class StudentInfo(BaseModel):
    name: str
    age: int
    grade: Literal["A", "B", "C", "D", "F"]

# Create Node with content
class Student(types.Node):
    content: StudentInfo

# Register the adapter
Student.register_async_adapter(LionAGIAsyncPostgresAdapter)
```

## Save Operations

```python
# Create student objects
students = [
    Student(content=StudentInfo(name="Adam Smith", grade="A", age=20)),
    Student(content=StudentInfo(name="Bob Johnson", grade="B", age=22)),
    Student(content=StudentInfo(name="Charlie Brown", grade="C", age=21)),
]

# Save to database (table created automatically)
records = []
for student in students:
    result = await student.adapt_to_async(
        obj_key="lionagi_async_pg",
        dsn=dsn,
        table="students",
    )
    records.append(result)

print(f"Saved {len(records)} records")
# Output: Saved 3 records
```

## Query Operations

```python
# Fetch single record
student = await Student.adapt_from_async(
    {"dsn": dsn, "table": "students"},
    obj_key="lionagi_async_pg",
)
print(f"Retrieved: {student.content.name}, Grade: {student.content.grade}")

# Fetch multiple with limit
students = await Student.adapt_from_async(
    {"dsn": dsn, "table": "students", "limit": 2},
    obj_key="lionagi_async_pg",
    many=True,
)
print(f"Retrieved {len(students)} students")

# Fetch with conditions
adam = await Student.adapt_from_async(
    {"dsn": dsn, "table": "students", "selectors": {"id": str(student.id)}},
    obj_key="lionagi_async_pg",
)
```

## Update Operations

```python
# Modify and update
adam.content.age = 22

# Update in database
result = await adam.adapt_to_async(
    "lionagi_async_pg",
    dsn=dsn,
    table="students",
    operation="update",
    where={"id": str(adam.id)},
)

# Verify update
updated_adam = await Student.adapt_from_async(
    {"dsn": dsn, "table": "students", "selectors": {"id": str(adam.id)}},
    obj_key="lionagi_async_pg",
)
print(f"Updated age: {updated_adam.content.age}")
```

## Conversation Persistence

```python
from lionagi import Branch

# Create conversation data model
class ConversationState(BaseModel):
    branch_id: str
    conversation_data: dict
    message_count: int
    last_updated: float

class ConversationNode(types.Node):
    content: ConversationState

ConversationNode.register_async_adapter(LionAGIAsyncPostgresAdapter)

async def save_conversation(branch: Branch):
    """Save branch state to database"""
    import time
    
    conversation = ConversationNode(
        content=ConversationState(
            branch_id=str(branch.id),
            conversation_data=branch.to_dict(),
            message_count=len(branch.messages),
            last_updated=time.time()
        )
    )
    
    await conversation.adapt_to_async(
        obj_key="lionagi_async_pg",
        dsn=dsn,
        table="conversations",
    )
    return conversation

async def load_conversation(branch_id: str) -> Branch:
    """Load branch state from database"""
    conversation = await ConversationNode.adapt_from_async(
        {"dsn": dsn, "table": "conversations", "selectors": {"branch_id": branch_id}},
        obj_key="lionagi_async_pg",
    )
    
    return Branch.from_dict(conversation.content.conversation_data)

# Usage
branch = Branch(system="You are a helpful assistant")
await branch.communicate("Hello, how are you?")

# Save conversation
saved_conv = await save_conversation(branch)
print(f"Saved conversation with {saved_conv.content.message_count} messages")

# Load conversation
loaded_branch = await load_conversation(str(branch.id))
print(f"Loaded conversation with {len(loaded_branch.messages)} messages")
```

## SQLite Pattern

```python
# For local development with SQLite
sqlite_dsn = "sqlite+aiosqlite:///./conversations.db"

# Same Node patterns work with SQLite
class LocalStudent(types.Node):
    content: StudentInfo

# Use async SQLite adapter (if available) or regular sync operations
async def save_local(student: LocalStudent):
    return await student.adapt_to_async(
        obj_key="lionagi_async_sqlite",  # Adapter key for SQLite
        dsn=sqlite_dsn,
        table="local_students",
    )
```

## Supabase Integration

```python
# For Supabase (managed PostgreSQL)
# Set up project: supabase init && supabase start

supabase_dsn = "postgresql+asyncpg://postgres:postgres@127.0.0.1:54322/postgres"

class SupabaseData(BaseModel):
    title: str
    content: str
    created_by: str

class SupabaseNode(types.Node):
    content: SupabaseData

SupabaseNode.register_async_adapter(LionAGIAsyncPostgresAdapter)

async def supabase_operations():
    # Create
    node = SupabaseNode(
        content=SupabaseData(
            title="Research Paper",
            content="AI safety considerations...",
            created_by="researcher_001"
        )
    )
    
    # Save to Supabase
    result = await node.adapt_to_async(
        obj_key="lionagi_async_pg",
        dsn=supabase_dsn,
        table="research_papers",
    )
    
    # Query by creator
    papers = await SupabaseNode.adapt_from_async(
        {
            "dsn": supabase_dsn, 
            "table": "research_papers",
            "selectors": {"created_by": "researcher_001"}
        },
        obj_key="lionagi_async_pg",
        many=True,
    )
    
    return papers

# Usage
papers = await supabase_operations()
print(f"Found {len(papers)} research papers")
```

## Production Patterns

```python
import asyncio
from contextlib import asynccontextmanager

class DatabaseManager:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool = None
    
    @asynccontextmanager
    async def get_connection(self):
        """Connection pool management"""
        try:
            # Connection logic here
            yield self.dsn
        except Exception as e:
            print(f"Database error: {e}")
            raise
        finally:
            # Cleanup
            pass

# Error handling wrapper
async def safe_db_operation(operation, **kwargs):
    """Wrapper for database operations with error handling"""
    try:
        return await operation(**kwargs)
    except Exception as e:
        print(f"Database operation failed: {e}")
        return None

# Batch operations
async def batch_save(nodes: list[types.Node], table: str):
    """Save multiple nodes efficiently"""
    tasks = []
    for node in nodes:
        task = safe_db_operation(
            node.adapt_to_async,
            obj_key="lionagi_async_pg",
            dsn=dsn,
            table=table,
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    successful = [r for r in results if r is not None]
    
    print(f"Saved {len(successful)}/{len(nodes)} records")
    return successful

# Usage
students = [
    Student(content=StudentInfo(name=f"Student {i}", grade="A", age=20+i))
    for i in range(10)
]
results = await batch_save(students, "batch_students")
```

## Key Patterns

**Node Creation:**

1. Define Pydantic model for data structure
2. Create Node class with content field
3. Register appropriate async adapter

**Database Operations:**

- `adapt_to_async()` for save/update operations
- `adapt_from_async()` for query operations
- Automatic table creation and schema management
- Support for PostgreSQL, SQLite, and Supabase

**Production Considerations:**

- Connection pooling for performance
- Error handling with graceful degradation
- Batch operations for efficiency
- Proper async/await patterns throughout
