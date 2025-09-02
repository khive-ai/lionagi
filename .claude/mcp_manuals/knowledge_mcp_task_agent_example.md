# Task Agent Example: Using Knowledge MCP

This example shows how a Task agent researching "async patterns in Rust" would
use the Knowledge MCP instead of writing a text summary.

## Traditional Approach (BAD)

```text
I researched async patterns in Rust and found that Tokio is the most popular runtime.
It uses work-stealing for efficiency. The async/await syntax makes code readable.
Key findings: ...
```

## Knowledge MCP Approach (GOOD)

```python
# Task Agent: researcher_async_patterns_001
# Domain: async-programming

# 1. Create references for all sources
tokio_ref = mcp__knowledge__create_reference(
    source_type="Documentation",
    title="Tokio Official Documentation",
    created_by="researcher_async_patterns_001",
    url="https://tokio.rs/tokio/tutorial",
    author="Tokio Contributors"
)["reference_id"]

rust_book_ref = mcp__knowledge__create_reference(
    source_type="Book",
    title="The Rust Programming Language - Async Chapter",
    created_by="researcher_async_patterns_001",
    url="https://doc.rust-lang.org/book/ch16-00-concurrency.html",
    author="Steve Klabnik and Carol Nichols"
)["reference_id"]

benchmark_ref = mcp__knowledge__create_reference(
    source_type="Article",
    title="Async Runtime Benchmarks 2024",
    created_by="researcher_async_patterns_001",
    url="https://example.com/benchmarks",
    metadata={"year": 2024, "methodology": "standardized workloads"}
)["reference_id"]

# 2. Create entities for discovered concepts
tokio_entity = mcp__knowledge__create_entity(
    name="Tokio",
    entity_type="Technology",
    created_by="researcher_async_patterns_001",
    properties={
        "category": "async-runtime",
        "language": "rust",
        "version": "1.35",
        "features": ["work-stealing", "multi-threaded", "io-uring"]
    },
    confidence=0.95,
    references=[tokio_ref, benchmark_ref]
)["entity_id"]

async_await_entity = mcp__knowledge__create_entity(
    name="Async/Await Pattern",
    entity_type="Concept",
    created_by="researcher_async_patterns_001",
    properties={
        "paradigm": "asynchronous-programming",
        "introduced": "Rust 1.39"
    },
    confidence=0.9,
    references=[rust_book_ref]
)["entity_id"]

work_stealing_entity = mcp__knowledge__create_entity(
    name="Work Stealing Scheduler",
    entity_type="Concept",
    created_by="researcher_async_patterns_001",
    properties={
        "type": "scheduling-algorithm",
        "benefits": ["load-balancing", "cpu-efficiency"]
    },
    confidence=0.85,
    references=[tokio_ref]
)["entity_id"]

# 3. Map relationships between entities
mcp__knowledge__create_relationship(
    rel_type="Implements",
    source_id=tokio_entity,
    target_id=async_await_entity,
    created_by="researcher_async_patterns_001",
    properties={"since": "0.2.0"},
    confidence=0.95
)

mcp__knowledge__create_relationship(
    rel_type="Uses",
    source_id=tokio_entity,
    target_id=work_stealing_entity,
    created_by="researcher_async_patterns_001",
    properties={"purpose": "multi-core efficiency"},
    confidence=0.9,
    references=[tokio_ref]
)

# 4. Record observations
obs1 = mcp__knowledge__observe(
    content="Tokio's work-stealing scheduler improves CPU utilization by 40% in I/O-bound workloads compared to single-threaded executors",
    context="performance-analysis",
    confidence=0.85,
    entity_id=tokio_entity,
    references=[benchmark_ref],
    created_by="researcher_async_patterns_001"
)["note_id"]

obs2 = mcp__knowledge__observe(
    content="The async/await syntax in Rust provides zero-cost abstractions, compiling to state machines",
    context="language-design",
    confidence=0.9,
    entity_id=async_await_entity,
    references=[rust_book_ref],
    created_by="researcher_async_patterns_001"
)["note_id"]

obs3 = mcp__knowledge__observe(
    content="Work stealing reduces thread contention by allowing idle threads to take tasks from busy threads' queues",
    context="algorithm-analysis",
    confidence=0.88,
    entity_id=work_stealing_entity,
    created_by="researcher_async_patterns_001"
)["note_id"]

# 5. Ask questions for further investigation
q1 = mcp__knowledge__ask_question(
    question="How does Tokio's work-stealing compare to Go's goroutine scheduler?",
    domain="async-programming",
    urgency=0.6,
    entity_id=tokio_entity,
    created_by="researcher_async_patterns_001"
)["note_id"]

q2 = mcp__knowledge__ask_question(
    question="What are the memory overhead implications of async/await state machines?",
    domain="memory-systems",
    urgency=0.7,
    entity_id=async_await_entity,
    created_by="researcher_async_patterns_001"
)["note_id"]

# 6. Record insights based on observations
insight1 = mcp__knowledge__record_insight(
    insight="Tokio's combination of work-stealing and async/await provides best-in-class performance for Rust async applications",
    connections=[obs1, obs2, obs3],
    strength=0.85,
    created_by="researcher_async_patterns_001"
)["note_id"]

# 7. Create a session-level insight
session_insight = mcp__knowledge__record_insight(
    insight="Rust's async ecosystem achieves performance parity with C++ while providing memory safety guarantees",
    connections=[insight1, obs2],
    strength=0.9,
    created_by="researcher_async_patterns_001"
)["note_id"]

# 8. Reinforce critical findings
mcp__knowledge__reinforce_note(obs1)  # Performance data
mcp__knowledge__reinforce_note(insight1)  # Key insight

# 9. Query to verify knowledge accumulation
all_insights = mcp__knowledge__query(
    "MATCH (n:Note) WHERE n.note_type = 'insight' AND n.created_by = 'researcher_async_patterns_001' RETURN n"
)

tokio_knowledge = mcp__knowledge__get_entity(tokio_entity)
```

## Benefits of Knowledge MCP Approach

1. **Structured**: Information is organized in a queryable graph
2. **Traceable**: Every claim links back to sources via references
3. **Reusable**: Other agents can query and build upon this knowledge
4. **Temporal**: Notes have decay rates and can be reinforced
5. **Contextual**: Observations attach to specific entities
6. **Discoverable**: Insights connect to supporting evidence

## What NOT to Do

```python
# DON'T: Create entities without references
mcp__knowledge__create_entity("Some Tech", "Technology", agent_id)  # No references!

# DON'T: Record isolated observations
mcp__knowledge__observe("Something interesting")  # Not attached to entity!

# DON'T: Create generic entities
mcp__knowledge__create_entity("Thing", "Thing", agent_id)  # Use specific types!

# DON'T: Forget to record your sources
# Always create references FIRST
```

## Query Examples for Other Agents

```python
# Find all Rust async technologies
mcp__knowledge__query(
    "MATCH (t:Technology)-[:Implements]->(c:Concept) "
    "WHERE c.name = 'Async/Await Pattern' AND t.properties.language = 'rust' "
    "RETURN t, c"
)

# Get all performance observations about Tokio
mcp__knowledge__query(
    "MATCH (e:Entity {name: 'Tokio'})<-[:attaches_to]-(n:Note) "
    "WHERE n.note_type.context = 'performance-analysis' "
    "RETURN n ORDER BY n.confidence DESC"
)

# Find unanswered questions about async programming
mcp__knowledge__query(
    "MATCH (n:Note) "
    "WHERE n.note_type.type = 'question' AND n.note_type.domain = 'async-programming' "
    "RETURN n ORDER BY n.note_type.urgency DESC"
)
```

Remember: The goal is to build a knowledge graph that other agents can query and
extend, not to write documents that only humans can read!
