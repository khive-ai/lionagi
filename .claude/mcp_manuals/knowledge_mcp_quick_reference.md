# Knowledge MCP Quick Reference

## ⚡ Quick Status (2025-06-29)

- ✅ **Deduplication**: Case-insensitive name matching prevents duplicates
- ✅ **Entity operations**: Create, get, observe all working
- ⚠️ **Queries**: `MATCH (n) RETURN n` works, type-specific queries limited
- ✅ **Relationships**: Creation working, queries in progress

## Essential Commands

### 1. Create Reference (Always First!)

```python
ref_id = mcp__knowledge__create_reference(
    source_type="Documentation",  # Book, Article, CodeRepository, Website
    title="Source Title",
    created_by="your_agent_id",
    url="https://optional.url",
    author="Optional Author"
)["reference_id"]
```

### 2. Create Entity (With Deduplication)

```python
# Handle potential duplicates
try:
    entity_id = mcp__knowledge__create_entity(
        name="Entity Name",  # Case-insensitive duplicate check!
        entity_type="Technology",  # Person, Agent, Project, Concept, Technology, Issue, Session, Organization
        created_by="your_agent_id",
        properties={"key": "value"},  # Optional
        confidence=0.9,  # 0.0-1.0
        references=[ref_id]  # Link to sources
    )["entity_id"]
except Exception as e:
    if "Duplicate entity" in str(e):
        # Extract existing ID from error message
        existing_id = str(e).split("ID: ")[1].strip()
        entity_id = existing_id
```

### 3. Create Relationship

```python
rel_id = mcp__knowledge__create_relationship(
    rel_type="Uses",  # Creates, Uses, Implements, Fixes, DependsOn, WorksOn
    source_id=entity1_id,
    target_id=entity2_id,
    created_by="your_agent_id",
    confidence=0.8
)["relationship_id"]
```

### 4. Record Observations

```python
# Simple observation
mcp__knowledge__observe(
    content="What you observed",
    context="architecture",  # performance, security, etc.
    confidence=0.9,
    entity_id=entity_id,  # Attach to entity
    created_by="your_agent_id"
)

# Ask a question
mcp__knowledge__ask_question(
    question="What needs investigation?",
    domain="distributed-systems",
    urgency=0.7,  # 0.0-1.0
    entity_id=entity_id,
    created_by="your_agent_id"
)

# Record an insight
mcp__knowledge__record_insight(
    insight="Pattern or discovery",
    connections=[note1_id, note2_id],  # Related notes
    strength=0.85,
    created_by="your_agent_id"
)
```

### 5. Query Knowledge (Current Status)

```python
# ✅ WORKING: Get all entities
all_entities = mcp__knowledge__query("MATCH (n) RETURN n")["entities"]

# ✅ WORKING: Get entity with notes
entity_data = mcp__knowledge__get_entity(entity_id)

# ⚠️ LIMITED: Type-specific queries (use alternatives below)
# mcp__knowledge__query("MATCH (n:Technology) RETURN n")  # Returns empty

# Alternative: Filter all entities by type
tech_entities = [e for e in all_entities if e["entity_type"] == "TECHNOLOGY"]

# ⚠️ NOT YET: Relationship queries
# mcp__knowledge__query("MATCH (a)-[r:Uses]->(b) RETURN a, r, b")
```

### 6. Reinforce Important Notes

```python
mcp__knowledge__reinforce_note(note_id)
```

## Common Patterns

### Research Pattern

```python
# 1. Cite source
ref = create_reference("Paper Title", "Article", agent_id, url="...")

# 2. Create concept
concept = create_entity("New Concept", "Concept", agent_id, references=[ref])

# 3. Record findings
observe("Key finding about concept", entity_id=concept["entity_id"])
```

### Technology Analysis Pattern

```python
# 1. Create tech entity
tech = create_entity("Rust", "Technology", agent_id)

# 2. Create project using it
proj = create_entity("Lion", "Project", agent_id)

# 3. Link them
create_relationship("Uses", proj["entity_id"], tech["entity_id"], agent_id)

# 4. Record observations
observe("Lion uses Rust for performance", entity_id=proj["entity_id"])
```

### Problem Investigation Pattern

```python
# 1. Create issue
issue = create_entity("Performance Problem", "Issue", agent_id)

# 2. Ask clarifying questions
ask_question("What causes the bottleneck?", urgency=0.9, entity_id=issue["entity_id"])

# 3. Record observations
observe("CPU usage spikes during X", entity_id=issue["entity_id"])

# 4. Record insight when found
record_insight("Root cause: inefficient algorithm", strength=0.9)
```

## Remember

- **NEW**: Duplicate entities are automatically rejected (case-insensitive)
- **NEW**: Use `MATCH (n) RETURN n` for queries, filter results as needed
- Always create references first for traceability
- Attach notes to entities for context
- Use specific entity types (not generic)
- Connect insights to source observations
- Reinforce important notes to prevent decay
- This is append-only - no updates allowed

## Common Errors & Solutions

### Duplicate Entity Error

```python
# Error: "Duplicate entity: Entity 'Test Name' already exists with ID: xxx"
# Solution: Use the existing entity ID from the error message
```

### Empty Query Results

```python
# Issue: MATCH (n:Technology) returns empty
# Solution: Use MATCH (n) and filter results
all_entities = mcp__knowledge__query("MATCH (n) RETURN n")["entities"]
tech_only = [e for e in all_entities if e["entity_type"] == "TECHNOLOGY"]
```
