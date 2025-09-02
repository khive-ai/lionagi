# Knowledge MCP Usage Guide

## Overview

The Knowledge MCP is an append-only knowledge graph system that enables Task
agents to accumulate structured knowledge instead of writing unstructured
summaries. It implements a temporal episodic buffer through notes that attach to
entities and relationships, supporting AGI development through proper knowledge
representation.

## Important Updates (2025-06-29)

### Deduplication Features

The Knowledge MCP now includes robust deduplication:

- **Case-insensitive exact matching**: "Test Entity" and "test entity" are
  detected as duplicates
- **Automatic rejection**: Duplicate entities return an error with the existing
  entity ID
- **Name normalization**: Built into the validation process

### Query Functionality Status

- ✅ `MATCH (n) RETURN n` - Works! Returns all entities (limited to 100)
- ✅ `MATCH (n:Entity) RETURN n` - Works! Generic entity label
- ⚠️ `MATCH (n:Technology) RETURN n` - Currently not working due to SQL query
  issue
- ✅ `get_entity(id)` - Works perfectly for direct entity retrieval

**Recommendation**: Use `get_entity()` for specific entities or
`MATCH (n) RETURN n` for all entities until type-specific queries are fixed.

## Core Concepts

### 1. References

References provide evidence, traceability, and observability without storing
copyrighted content.

#### Reference Types & Templates

**Academic Paper**

```python
ref_id = mcp__knowledge__create_reference(
    source_type="Paper",
    title="Attention Is All You Need",
    created_by="research_agent_001",
    url="https://arxiv.org/abs/1706.03762",
    author="Vaswani et al.",
    publication_date="2017-06-12T00:00:00Z",
    metadata={
        "arxiv_id": "1706.03762",
        "venue": "NeurIPS 2017",
        "citations": "50000+",
        "field": "machine_learning"
    }
)["reference_id"]
```

**Code Repository**

```python
ref_id = mcp__knowledge__create_reference(
    source_type="CodeRepository",
    title="Lion-AGI Framework",
    created_by="code_analyst_002",
    url="https://github.com/ohdearquant/lion",
    author="Ocean",
    metadata={
        "language": "python",
        "stars": "1500",
        "license": "MIT",
        "commit": "abc123def",
        "version": "2.0.0"
    },
    content_hash="sha256:abcdef123456..."  # For integrity
)["reference_id"]
```

**Internal Documentation**

```python
ref_id = mcp__knowledge__create_reference(
    source_type="Documentation",
    title="KB System Architecture Design",
    created_by="architect_agent_003",
    author="Ocean",
    metadata={
        "internal": true,
        "version": "1.0",
        "category": "architecture",
        "team": "core"
    },
    archived=true  # Indicates we have a local copy
)["reference_id"]
```

**Blog Post/Article**

```python
ref_id = mcp__knowledge__create_reference(
    source_type="BlogPost",
    title="Building AGI with Event-Driven Architecture",
    created_by="content_agent_004",
    url="https://blog.example.com/agi-event-driven",
    author="Jane Doe",
    publication_date="2025-01-15T10:00:00Z",
    metadata={
        "tags": ["agi", "event-driven", "architecture"],
        "reading_time": "15 min",
        "comments": 42
    }
)["reference_id"]
```

**Book Reference**

```python
ref_id = mcp__knowledge__create_reference(
    source_type="Book",
    title="Designing Data-Intensive Applications",
    created_by="research_agent_005",
    author="Martin Kleppmann",
    publication_date="2017-03-01T00:00:00Z",
    metadata={
        "isbn": "978-1449373320",
        "publisher": "O'Reilly Media",
        "pages": 616,
        "chapter_referenced": "Chapter 11: Stream Processing"
    }
)["reference_id"]
```

#### Best Practices for References

1. **Always include content_hash** for external sources you've processed
2. **Use archived=true** only when you have a local copy
3. **Include version/commit info** for code repositories
4. **Add publication_date** when available for temporal context
5. **Use rich metadata** to enable better search and filtering later

### 2. Entities

Entities represent concepts, people, projects, technologies, and more.

#### Entity Type Templates

**Technology Entity**

```python
try:
    entity_id = mcp__knowledge__create_entity(
        name="Tokio Runtime",
        entity_type="Technology",
        created_by="tech_analyst_001",
        properties={
            "category": "async-runtime",
            "language": "rust",
            "version": "1.35.0",
            "license": "MIT",
            "use_cases": ["web-servers", "network-services", "concurrent-systems"],
            "performance_characteristics": {
                "concurrency_model": "work-stealing",
                "memory_overhead": "low",
                "cpu_efficiency": "high"
            }
        },
        confidence=0.95,
        references=[rust_book_ref, tokio_docs_ref, benchmark_ref]
    )["entity_id"]
except Exception as e:
    if "Duplicate entity" in str(e):
        entity_id = str(e).split("ID: ")[1].strip()
        # Optionally attach new references to existing entity
        mcp__knowledge__observe(
            f"Additional reference found for Tokio: {rust_book_ref}",
            entity_id=entity_id,
            confidence=0.9
        )
```

**Person Entity**

```python
try:
    entity_id = mcp__knowledge__create_entity(
        name="Ocean Li",
        entity_type="Person",
        created_by="org_analyst_002",
        properties={
            "role": "Creator/Founder",
            "expertise": ["AGI", "distributed-systems", "lion-framework"],
            "affiliations": ["Lion Ecosystem"],
            "contributions": {
                "projects": ["lion-agi", "kb-system"],
                "papers": ["Event-Driven AGI Architecture"],
                "patents": []
            },
            "contact": {
                "github": "ohdearquant",
                "preferred_communication": "github-issues"
            }
        },
        confidence=1.0,
        references=[github_profile_ref, project_readme_ref]
    )["entity_id"]
except Exception as e:
    if "Duplicate entity" in str(e):
        entity_id = str(e).split("ID: ")[1].strip()
```

**Project Entity**

```python
try:
    entity_id = mcp__knowledge__create_entity(
        name="KB Multi-Agent Orchestration Framework",
        entity_type="Project",
        created_by="project_analyst_003",
        properties={
            "status": "active",
            "start_date": "2024-06-01",
            "repository": "https://github.com/ohdearquant/kb",
            "team_size": 5,
            "tech_stack": ["python", "rust", "postgresql", "neo4j"],
            "architecture": {
                "pattern": "event-driven",
                "components": ["orchestrator", "task-agents", "knowledge-mcp"],
                "deployment": "distributed"
            },
            "metrics": {
                "issues_processed": 150,
                "agent_invocations": 5000,
                "knowledge_entities": 2500
            }
        },
        confidence=0.9,
        references=[repo_ref, architecture_doc_ref]
    )["entity_id"]
except Exception as e:
    if "Duplicate entity" in str(e):
        entity_id = str(e).split("ID: ")[1].strip()
```

**Concept Entity**

```python
try:
    entity_id = mcp__knowledge__create_entity(
        name="Temporal Episodic Buffer",
        entity_type="Concept",
        created_by="research_agent_004",
        properties={
            "domain": "cognitive-architecture",
            "definition": "A temporary storage system that integrates information from multiple sources",
            "related_concepts": ["working-memory", "event-buffer", "episodic-memory"],
            "applications": ["AGI", "memory-systems", "knowledge-graphs"],
            "theoretical_basis": {
                "origin": "Baddeley's Working Memory Model",
                "year_proposed": 2000,
                "key_properties": ["multi-modal", "temporary", "integrative"]
            },
            "implementation_considerations": {
                "storage": "append-only",
                "decay_rate": "configurable",
                "capacity": "unlimited with decay"
            }
        },
        confidence=0.85,
        references=[cog_sci_paper_ref, implementation_ref]
    )["entity_id"]
except Exception as e:
    if "Duplicate entity" in str(e):
        entity_id = str(e).split("ID: ")[1].strip()
```

**Issue Entity**

```python
try:
    entity_id = mcp__knowledge__create_entity(
        name="Query Performance Degradation #1234",
        entity_type="Issue",
        created_by="debug_agent_005",
        properties={
            "status": "investigating",
            "severity": "medium",
            "reported_date": "2025-06-28",
            "symptoms": [
                "Cypher queries returning empty results",
                "Type-specific queries not working",
                "Generic queries functioning normally"
            ],
            "affected_components": ["query.rs", "storage.rs"],
            "investigation": {
                "root_cause": "SQL generation for entity types",
                "attempted_fixes": ["Added get_entities_by_type method"],
                "workaround": "Use MATCH (n) RETURN n and filter"
            },
            "impact": {
                "users_affected": "all",
                "functionality_impact": "partial",
                "data_integrity": "unaffected"
            }
        },
        confidence=0.9,
        references=[github_issue_ref, debug_log_ref]
    )["entity_id"]
except Exception as e:
    if "Duplicate entity" in str(e):
        entity_id = str(e).split("ID: ")[1].strip()
```

#### Entity Creation Best Practices

1. **Rich Properties**: Include comprehensive metadata for better searchability
2. **Consistent Naming**: Use official names, include version numbers where
   relevant
3. **Deduplication Handling**: Always wrap in try/except for duplicate handling
4. **Reference Linking**: Connect to multiple references for credibility
5. **Confidence Scores**:
   - 1.0: Absolute certainty (e.g., self-reported data)
   - 0.9-0.95: High confidence (multiple sources agree)
   - 0.8-0.9: Good confidence (single reliable source)
   - 0.7-0.8: Moderate confidence (inferred or partial data)
   - <0.7: Low confidence (use sparingly)

**Deduplication Strategy**:

```python
def create_or_get_entity(name, entity_type, properties, references, agent_id):
    """Helper function for entity creation with deduplication"""
    try:
        result = mcp__knowledge__create_entity(
            name=name,
            entity_type=entity_type,
            created_by=agent_id,
            properties=properties,
            confidence=0.9,
            references=references
        )
        entity_id = result["entity_id"]
        print(f"Created new entity: {entity_id}")
        return entity_id, True  # True = newly created
    except Exception as e:
        if "Duplicate entity" in str(e):
            entity_id = str(e).split("ID: ")[1].strip()
            print(f"Using existing entity: {entity_id}")
            # Optionally enrich existing entity with new observation
            if references:
                mcp__knowledge__observe(
                    f"Additional references found: {references}",
                    entity_id=entity_id,
                    confidence=0.9,
                    created_by=agent_id
                )
            return entity_id, False  # False = existing entity
        else:
            raise  # Re-raise other exceptions
```

### 3. Relationships

Relationships connect entities with typed edges.

#### Relationship Type Templates

**Technology Stack Relationship (Uses)**

```python
rel_id = mcp__knowledge__create_relationship(
    rel_type="Uses",
    source_id=kb_project_id,  # KB System project
    target_id=rust_tech_id,   # Rust technology
    created_by="architect_agent_001",
    properties={
        "since": "2024-01",
        "version": "1.75.0",
        "purpose": "performance-critical-components",
        "components": ["knowledge-mcp", "query-engine"],
        "satisfaction": "high",
        "alternatives_considered": ["go", "c++"],
        "decision_rationale": "memory safety + performance"
    },
    confidence=0.95,
    references=[architecture_decision_ref]
)["relationship_id"]
```

**Developer Contribution (WorksOn)**

```python
rel_id = mcp__knowledge__create_relationship(
    rel_type="WorksOn",
    source_id=ocean_person_id,
    target_id=kb_project_id,
    created_by="org_agent_002",
    properties={
        "role": "creator/lead",
        "start_date": "2024-01-01",
        "contribution_areas": ["architecture", "orchestration", "vision"],
        "time_allocation": "full-time",
        "key_decisions": [
            "event-driven architecture",
            "multi-agent orchestration",
            "knowledge-first approach"
        ]
    },
    confidence=1.0,
    references=[project_readme_ref, commit_history_ref]
)["relationship_id"]
```

**System Dependencies (DependsOn)**

```python
rel_id = mcp__knowledge__create_relationship(
    rel_type="DependsOn",
    source_id=knowledge_id,
    target_id=postgresql_id,
    created_by="infra_agent_003",
    properties={
        "dependency_type": "runtime",
        "version_constraint": ">=14.0",
        "features_used": ["jsonb", "arrays", "full-text-search"],
        "criticality": "high",
        "fallback_options": ["sqlite-for-testing"],
        "performance_requirements": {
            "connections": 100,
            "storage": "10GB",
            "query_rate": "1000/sec"
        }
    },
    confidence=0.9,
    references=[system_requirements_ref]
)["relationship_id"]
```

**Implementation Relationship (Implements)**

```python
rel_id = mcp__knowledge__create_relationship(
    rel_type="Implements",
    source_id=kb_system_id,
    target_id=event_driven_pattern_id,
    created_by="architect_agent_004",
    properties={
        "implementation_details": {
            "event_store": "github-issues",
            "event_types": ["research-requested", "decision-ready", "implementation-complete"],
            "processing_model": "async-multi-agent"
        },
        "conformance_level": "full",
        "adaptations": [
            "github-as-event-queue",
            "label-based-state-machine",
            "deliverable-driven-progression"
        ],
        "benefits_realized": [
            "full-audit-trail",
            "transparent-progress",
            "webhook-automation"
        ]
    },
    confidence=0.85,
    references=[architecture_doc_ref, implementation_guide_ref]
)["relationship_id"]
```

**Problem Resolution (Fixes)**

```python
rel_id = mcp__knowledge__create_relationship(
    rel_type="Fixes",
    source_id=pr_1234_id,  # Pull request entity
    target_id=issue_5678_id,  # Issue entity
    created_by="dev_agent_005",
    properties={
        "fix_type": "bugfix",
        "root_cause": "sql-query-generation",
        "changes_made": [
            "Added get_entities_by_type method",
            "Updated query execution logic",
            "Added proper type mapping"
        ],
        "testing": {
            "unit_tests_added": 5,
            "integration_tests": 2,
            "manual_verification": "completed"
        },
        "impact": "restored type-specific queries",
        "backport_needed": false
    },
    confidence=0.9,
    references=[pr_ref, issue_ref, test_results_ref]
)["relationship_id"]
```

**Conceptual Creation (Creates)**

```python
rel_id = mcp__knowledge__create_relationship(
    rel_type="Creates",
    source_id=ocean_person_id,
    target_id=lion_framework_id,
    created_by="history_agent_006",
    properties={
        "creation_date": "2023-06-01",
        "motivation": "need for intelligent orchestration",
        "initial_vision": "AGI through multi-agent coordination",
        "evolution": [
            {"date": "2023-06", "milestone": "initial concept"},
            {"date": "2023-09", "milestone": "first prototype"},
            {"date": "2024-01", "milestone": "production ready"},
            {"date": "2024-06", "milestone": "KB integration"}
        ],
        "intellectual_property": "copyright Ocean Li"
    },
    confidence=1.0,
    references=[project_history_ref, announcement_ref]
)["relationship_id"]
```

#### Relationship Best Practices

1. **Rich Properties**: Include context, rationale, and timeline information
2. **Bidirectional Consideration**: Think about whether reverse relationship is
   needed
3. **Temporal Data**: Always include dates/versions when relevant
4. **Confidence Scoring**: Based on source reliability and information
   completeness
5. **Reference Support**: Link to evidence supporting the relationship

**Common Relationship Patterns**:

```python
# Technology Stack Pattern
project -> Uses -> technology
project -> DependsOn -> infrastructure
project -> Implements -> pattern/standard

# Team Pattern  
person -> WorksOn -> project
person -> Creates -> concept/project
person -> Maintains -> component

# Problem Resolution Pattern
pr/commit -> Fixes -> issue
issue -> Blocks -> feature
feature -> DependsOn -> component

# Knowledge Flow Pattern
paper -> Introduces -> concept
concept -> Influences -> implementation
implementation -> Validates -> theory
```

### 4. Notes (Temporal Episodic Buffer)

Notes are the primary mechanism for accumulating observations, questions,
insights, corrections, and errors. They have temporal decay and can be
reinforced.

#### Note Type Templates

**Observation Note - Performance Analysis**

```python
note_id = mcp__knowledge__create_note(
    content="Tokio's work-stealing scheduler improves CPU utilization by 40% in I/O-bound workloads. Benchmark shows 10k req/s with 100ms p99 latency on 4-core machine.",
    note_type={
        "type": "observation",
        "context": "performance-analysis",
        "confidence": 0.85
    },
    created_by="performance_agent_004",
    attachments=[{
        "target_type": "entity",
        "entity_id": tokio_entity_id,
        "relevance": 0.95
    }],
    references=[benchmark_ref_id, test_setup_ref],
    session_id="perf_analysis_2025_01_29",
    decay_rate=0.1  # Low decay - performance data stays relevant
)["note_id"]
```

**Observation Note - Architecture Discovery**

```python
note_id = mcp__knowledge__create_note(
    content="KB system uses GitHub issues as an event queue, enabling transparent audit trail and webhook automation. Each issue represents an event in the state machine.",
    note_type={
        "type": "observation",
        "context": "architecture-analysis",
        "confidence": 0.95
    },
    created_by="architect_agent_001",
    attachments=[
        {"target_type": "entity", "entity_id": kb_system_id, "relevance": 1.0},
        {"target_type": "entity", "entity_id": github_pattern_id, "relevance": 0.9}
    ],
    references=[architecture_doc_ref],
    session_id="architecture_review_2025_06",
    decay_rate=0.05  # Very low decay - architectural decisions are long-lived
)["note_id"]
```

**Question Note - Urgent Investigation**

```python
note_id = mcp__knowledge__create_note(
    content="Why are type-specific Cypher queries returning empty results while generic queries work? Is this a SQL generation issue or a deeper architectural problem?",
    note_type={
        "type": "question",
        "urgency": 0.9,  # High urgency - blocking functionality
        "domain": "query-engine"
    },
    created_by="debug_agent_002",
    attachments=[
        {"target_type": "entity", "entity_id": knowledge_id, "relevance": 1.0},
        {"target_type": "entity", "entity_id": query_issue_id, "relevance": 0.95}
    ],
    references=[debug_log_ref],
    session_id="debugging_session_2025_06_29",
    decay_rate=0.3  # High decay - urgent questions should be answered quickly
)["note_id"]
```

**Question Note - Research Direction**

```python
note_id = mcp__knowledge__create_note(
    content="How can we implement temporal consistency in a distributed knowledge graph while maintaining eventual consistency guarantees?",
    note_type={
        "type": "question",
        "urgency": 0.5,  # Medium urgency - important but not blocking
        "domain": "distributed-systems"
    },
    created_by="research_agent_003",
    attachments=[
        {"target_type": "entity", "entity_id": temporal_consistency_id, "relevance": 0.9},
        {"target_type": "entity", "entity_id": distributed_kg_id, "relevance": 0.85}
    ],
    references=[research_paper_ref],
    session_id="research_planning_2025_06",
    decay_rate=0.15  # Moderate decay - research questions have medium lifespan
)["note_id"]
```

**Insight Note - Pattern Discovery**

```python
note_id = mcp__knowledge__create_note(
    content="Event sourcing combined with temporal episodic buffers creates natural memory consolidation. The append-only nature preserves history while decay handles relevance.",
    note_type={
        "type": "insight",
        "connections": [obs_note_1_id, obs_note_2_id, question_note_id],
        "strength": 0.9  # High confidence in this insight
    },
    created_by="analyst_agent_004",
    attachments=[
        {"target_type": "entity", "entity_id": event_sourcing_id, "relevance": 0.95},
        {"target_type": "entity", "entity_id": memory_consolidation_id, "relevance": 0.9}
    ],
    references=[event_sourcing_ref, cog_sci_ref],
    session_id="pattern_analysis_2025_06",
    decay_rate=0.08  # Low decay - insights are valuable long-term
)["note_id"]

# Reinforce important insights
mcp__knowledge__reinforce_note(note_id)
```

**Insight Note - Cross-Domain Connection**

```python
note_id = mcp__knowledge__create_note(
    content="The KB system's event-driven architecture mirrors biological memory consolidation: immediate capture (working memory) → structured accumulation (knowledge graph) → decay/reinforcement (temporal dynamics)",
    note_type={
        "type": "insight",
        "connections": [memory_arch_note, kb_design_note, neurosci_note],
        "strength": 0.85
    },
    created_by="innovation_agent_005",
    attachments=[
        {"target_type": "entity", "entity_id": kb_system_id, "relevance": 0.9},
        {"target_type": "relationship", "relationship_id": kb_implements_pattern_id, "relevance": 0.85}
    ],
    references=[neuroscience_paper_ref, architecture_doc_ref],
    session_id="cross_domain_analysis_2025",
    decay_rate=0.05  # Very low decay - fundamental insights
)["note_id"]
```

**Correction Note - Data Fix**

```python
note_id = mcp__knowledge__create_note(
    content="Tokio version was incorrectly recorded as 1.35.0, actual version in use is 1.36.0. Updated entity properties.",
    note_type={
        "type": "correction",
        "target_id": tokio_entity_id,
        "field": "properties.version"
    },
    created_by="validation_agent_006",
    attachments=[
        {"target_type": "entity", "entity_id": tokio_entity_id, "relevance": 1.0}
    ],
    references=[cargo_lock_ref],
    session_id="dependency_audit_2025_06",
    decay_rate=0.2  # Moderate decay - corrections are historical record
)["note_id"]
```

**Agent Error Note - Recovery Record**

```python
note_id = mcp__knowledge__create_note(
    content="Failed to parse Cypher query 'MATCH (n:Technology) RETURN n' - SQL generation returned empty result set. Root cause: entity_type uppercase mismatch in storage.rs line 325.",
    note_type={
        "type": "agent_error",
        "error_type": "data_error",
        "severity": "medium"  # Not critical but affects functionality
    },
    created_by="debug_agent_007",
    attachments=[
        {"target_type": "entity", "entity_id": knowledge_id, "relevance": 0.9},
        {"target_type": "entity", "entity_id": query_bug_id, "relevance": 1.0}
    ],
    references=[error_log_ref, stack_trace_ref],
    session_id="error_investigation_2025_06_29",
    decay_rate=0.25  # Higher decay - errors should be fixed and forgotten
)["note_id"]
```

#### Note Creation Best Practices

1. **Rich Content**: Include specific details, numbers, and context
2. **Proper Attachments**: Link to relevant entities and relationships
3. **Session Coherence**: Use consistent session_id for related work
4. **Decay Rate Strategy**:
   - 0.05-0.1: Long-term valuable (insights, architecture)
   - 0.1-0.2: Medium-term relevant (observations, corrections)
   - 0.2-0.3: Short-term important (questions, errors)
5. **Reference Support**: Always link to evidence
6. **Reinforce Important Notes**: Use reinforce_note() for critical insights

#### Convenience Functions Usage

**Simple Observation**

```python
# Use convenience function for quick observations
mcp__knowledge__observe(
    content="PostgreSQL query performance degrades with >1M entities",
    context="performance-testing",
    confidence=0.8,
    entity_id=postgresql_entity_id,
    references=[test_results_ref],
    created_by="perf_agent_001"
)
```

**Urgent Question**

```python
# Use convenience function for questions
mcp__knowledge__ask_question(
    question="How to implement sharding for the knowledge graph?",
    domain="distributed-systems",
    urgency=0.7,
    entity_id=knowledge_id,
    created_by="architect_agent_002"
)
```

**Pattern Insight**

```python
# Use convenience function for insights
mcp__knowledge__record_insight(
    insight="Batch processing entities in groups of 100 optimizes memory usage",
    connections=[perf_note_1, perf_note_2, config_note],
    strength=0.9,
    created_by="optimization_agent_003"
)
```

## Query Patterns

### Working Queries

```python
# Get all entities (limited to 100)
results = mcp__knowledge__query(
    query="MATCH (n) RETURN n"
)

# Get all entities with generic label
results = mcp__knowledge__query(
    query="MATCH (n:Entity) RETURN n"
)
```

### Currently Limited Queries

```python
# Type-specific queries are not yet working
# Use the working queries above or get_entity() instead
results = mcp__knowledge__query(
    query="MATCH (n:Technology) RETURN n"  # Returns empty
)
```

### Relationship Traversal

```python
results = mcp__knowledge__query(
    query="MATCH (p:Project)-[r:Uses]->(t:Technology) WHERE t.name = 'Rust' RETURN p, r, t"
)
```

### Recent Insights

```python
results = mcp__knowledge__query(
    query="MATCH (n:Note) WHERE n.note_type = 'insight' RETURN n ORDER BY n.created_at DESC LIMIT 10"
)
```

### Entity with Notes

```python
entity_data = mcp__knowledge__get_entity(entity_id)
# Returns entity with all attached notes
```

## Convenience Functions

### Record Observation

```python
mcp__knowledge__observe(
    content="The system shows linear scalability up to 64 cores",
    context="performance-testing",
    confidence=0.9,
    entity_id=system_entity_id,
    references=[test_report_ref_id],
    created_by="perf_agent_001"
)
```

### Ask Question

```python
mcp__knowledge__ask_question(
    question="How does temporal decay affect long-term memory consolidation?",
    domain="temporal-reasoning",
    urgency=0.7,
    entity_id=memory_system_entity_id,
    created_by="research_agent_002"
)
```

### Record Insight

```python
mcp__knowledge__record_insight(
    insight="Event sourcing prevents temporal inconsistencies in distributed systems",
    connections=[note_id1, note_id2],  # Notes that led to this insight
    strength=0.9,
    created_by="analyst_agent_003"
)
```

## Best Practices

### 1. Check for Duplicates Before Creating

```python
# Bad: Creating entities without checking
entity_id = create_entity("Important Concept", "Concept", "agent_001")

# Good: Handle potential duplicates
try:
    entity_id = create_entity("Important Concept", "Concept", "agent_001")
except Exception as e:
    if "Duplicate entity" in str(e):
        # Use existing entity
        existing_id = extract_id_from_error(e)
        entity_id = existing_id
```

### 2. Always Create References First

```python
# Bad: Creating entities without references
entity_id = create_entity("Important Concept", "Concept", "agent_001")

# Good: Create reference first for traceability
ref_id = create_reference("Research Paper", "Article", "agent_001", url="https://...")
entity_id = create_entity("Important Concept", "Concept", "agent_001", references=[ref_id])
```

### 3. Use Specific Entity Types

```python
# Bad: Generic entity type
create_entity("Tokio", "Thing", "agent_001")

# Good: Specific entity type
create_entity("Tokio", "Technology", "agent_001")
```

### 4. Attach Notes to Entities

```python
# Bad: Isolated note
create_note("System is fast", {"type": "observation"}, "agent_001")

# Good: Note attached to entity
create_note(
    "System achieves 100k req/s",
    {"type": "observation", "context": "benchmark", "confidence": 0.95},
    "agent_001",
    attachments=[{"target_type": "entity", "entity_id": system_id, "relevance": 0.9}]
)
```

### 5. Use Session IDs for Related Work

```python
session_id = f"research_{research_topic}_{timestamp}"

# All related notes use same session
create_note(content1, note_type1, agent_id, session_id=session_id)
create_note(content2, note_type2, agent_id, session_id=session_id)
```

### 6. Reinforce Important Notes

```python
# Prevent decay of critical observations
if note_importance > 0.8:
    mcp__knowledge__reinforce_note(note_id)
```

### 7. Connect Insights to Source Notes

```python
# Track reasoning chain
observation_notes = [obs_note_1, obs_note_2, obs_note_3]
insight_note = record_insight(
    "Pattern discovered across observations",
    connections=observation_notes,
    strength=0.85
)
```

## Advanced Patterns & Workflows

### Complete Task Agent Research Workflow

```python
class ResearchAgent:
    def __init__(self, agent_id: str, research_topic: str):
        self.agent_id = agent_id
        self.session_id = f"research_{research_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.entities = {}
        self.references = {}
        self.notes = []
        
    async def research_technology_stack(self, project_name: str):
        """Complete workflow for researching a project's technology stack"""
        
        # Phase 1: Gather References
        print(f"[{self.agent_id}] Phase 1: Gathering references...")
        
        # Create references for different sources
        self.references['github'] = mcp__knowledge__create_reference(
            source_type="CodeRepository",
            title=f"{project_name} GitHub Repository",
            created_by=self.agent_id,
            url=f"https://github.com/example/{project_name}",
            metadata={"stars": 1500, "language": "rust"},
            content_hash="sha256:abc123..."
        )["reference_id"]
        
        self.references['docs'] = mcp__knowledge__create_reference(
            source_type="Documentation",
            title=f"{project_name} Architecture Guide",
            created_by=self.agent_id,
            metadata={"version": "2.0", "sections": ["overview", "design", "api"]}
        )["reference_id"]
        
        # Phase 2: Create Entities with Deduplication
        print(f"[{self.agent_id}] Phase 2: Creating entities...")
        
        # Create project entity
        project_id = await self.create_or_get_entity(
            name=project_name,
            entity_type="Project",
            properties={
                "description": "Multi-agent orchestration framework",
                "status": "active",
                "tech_stack": [],  # Will be populated
                "architecture_pattern": "event-driven"
            },
            references=[self.references['github'], self.references['docs']]
        )
        self.entities['project'] = project_id
        
        # Discover and create technology entities
        technologies = [
            ("Rust", "systems-programming", "1.75.0"),
            ("PostgreSQL", "database", "14.0"),
            ("Tokio", "async-runtime", "1.36.0")
        ]
        
        for tech_name, category, version in technologies:
            tech_id = await self.create_or_get_entity(
                name=tech_name,
                entity_type="Technology",
                properties={
                    "category": category,
                    "version": version,
                    "license": "MIT"  # Simplified
                },
                references=[self.references['github']]
            )
            self.entities[tech_name.lower()] = tech_id
            
            # Phase 3: Map Relationships
            print(f"[{self.agent_id}] Creating relationship: {project_name} uses {tech_name}")
            
            rel_id = mcp__knowledge__create_relationship(
                rel_type="Uses",
                source_id=project_id,
                target_id=tech_id,
                created_by=self.agent_id,
                properties={
                    "version": version,
                    "purpose": category,
                    "critical": True
                },
                confidence=0.95,
                references=[self.references['github']]
            )["relationship_id"]
            
            # Phase 4: Record Observations
            obs_content = f"{project_name} uses {tech_name} {version} for {category}"
            mcp__knowledge__observe(
                content=obs_content,
                context="tech-stack-analysis",
                confidence=0.9,
                entity_id=project_id,
                references=[self.references['github']],
                created_by=self.agent_id
            )
            
        # Phase 5: Analyze and Generate Insights
        print(f"[{self.agent_id}] Phase 5: Generating insights...")
        
        # Record architectural insight
        insight = mcp__knowledge__record_insight(
            insight="Project combines Rust's memory safety with PostgreSQL's reliability and Tokio's async performance for a robust event-driven architecture",
            connections=self.notes[-3:],  # Last 3 observations
            strength=0.85,
            created_by=self.agent_id
        )
        
        # Phase 6: Ask Follow-up Questions
        question = mcp__knowledge__ask_question(
            question="How does the event-driven architecture handle backpressure in high-load scenarios?",
            domain="distributed-systems",
            urgency=0.6,
            entity_id=project_id,
            created_by=self.agent_id
        )
        
        return {
            "entities_created": len(self.entities),
            "relationships_mapped": len(technologies),
            "insights_generated": 1,
            "questions_raised": 1
        }
    
    async def create_or_get_entity(self, name, entity_type, properties, references):
        """Helper with deduplication handling"""
        try:
            result = mcp__knowledge__create_entity(
                name=name,
                entity_type=entity_type,
                created_by=self.agent_id,
                properties=properties,
                confidence=0.9,
                references=references
            )
            entity_id = result["entity_id"]
            print(f"Created new entity: {name} ({entity_id})")
            return entity_id
        except Exception as e:
            if "Duplicate entity" in str(e):
                entity_id = str(e).split("ID: ")[1].strip()
                print(f"Found existing entity: {name} ({entity_id})")
                # Enrich with new references
                mcp__knowledge__observe(
                    f"Additional references found for {name}",
                    entity_id=entity_id,
                    confidence=0.9,
                    created_by=self.agent_id
                )
                return entity_id
            raise

# Usage
agent = ResearchAgent("research_agent_001", "kb_tech_stack")
results = await agent.research_technology_stack("KB System")
print(f"Research complete: {results}")
```

### Problem Investigation Workflow

```python
class DebugAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.session_id = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    async def investigate_issue(self, issue_description: str, affected_component: str):
        """Complete workflow for investigating and documenting an issue"""
        
        # Create issue entity
        issue_id = mcp__knowledge__create_entity(
            name=f"Issue: {issue_description[:50]}...",
            entity_type="Issue",
            created_by=self.agent_id,
            properties={
                "description": issue_description,
                "status": "investigating",
                "component": affected_component,
                "reported_date": datetime.now().isoformat()
            },
            confidence=1.0
        )["entity_id"]
        
        # Create component entity (with deduplication)
        try:
            component_id = mcp__knowledge__create_entity(
                name=affected_component,
                entity_type="Technology",
                created_by=self.agent_id,
                properties={"type": "component"},
                confidence=0.9
            )["entity_id"]
        except Exception as e:
            if "Duplicate entity" in str(e):
                component_id = str(e).split("ID: ")[1].strip()
        
        # Link issue to component
        mcp__knowledge__create_relationship(
            rel_type="Affects",
            source_id=issue_id,
            target_id=component_id,
            created_by=self.agent_id,
            properties={"impact": "functionality", "severity": "medium"},
            confidence=0.9
        )
        
        # Record investigation steps
        steps = [
            "Reproduced issue in test environment",
            "Identified SQL query generation as root cause",
            "Found entity_type case sensitivity mismatch",
            "Tested fix with get_entities_by_type method"
        ]
        
        observation_notes = []
        for i, step in enumerate(steps):
            note_id = mcp__knowledge__create_note(
                content=step,
                note_type={
                    "type": "observation",
                    "context": "debugging",
                    "confidence": 0.8 + (i * 0.05)  # Increasing confidence
                },
                created_by=self.agent_id,
                attachments=[{
                    "target_type": "entity",
                    "entity_id": issue_id,
                    "relevance": 0.9
                }],
                session_id=self.session_id,
                decay_rate=0.2
            )["note_id"]
            observation_notes.append(note_id)
        
        # Generate insight from investigation
        insight_id = mcp__knowledge__create_note(
            content="Case-insensitive entity matching requires consistent uppercase conversion in both storage and query layers",
            note_type={
                "type": "insight",
                "connections": observation_notes,
                "strength": 0.9
            },
            created_by=self.agent_id,
            attachments=[{
                "target_type": "entity",
                "entity_id": component_id,
                "relevance": 0.95
            }],
            session_id=self.session_id,
            decay_rate=0.1  # Important insight
        )["note_id"]
        
        # Reinforce the insight
        mcp__knowledge__reinforce_note(insight_id)
        
        # Update issue status
        mcp__knowledge__observe(
            "Issue resolved: Applied fix to storage.rs line 325",
            entity_id=issue_id,
            confidence=0.95,
            created_by=self.agent_id
        )
        
        return {
            "issue_id": issue_id,
            "root_cause": "Case sensitivity mismatch",
            "solution": "Uppercase conversion in query layer",
            "insights_generated": 1
        }
```

### Knowledge Mining Pattern

```python
class KnowledgeMiner:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.entity_cache = {}  # Local cache to prevent duplicates
        
    async def mine_codebase(self, repo_path: str, focus_areas: List[str]):
        """Extract structured knowledge from a codebase"""
        
        # Create repository reference
        repo_ref = mcp__knowledge__create_reference(
            source_type="CodeRepository",
            title=f"Repository: {repo_path}",
            created_by=self.agent_id,
            metadata={"path": repo_path, "focus_areas": focus_areas}
        )["reference_id"]
        
        discoveries = {
            "entities": [],
            "relationships": [],
            "patterns": []
        }
        
        # Mine each focus area
        for area in focus_areas:
            if area == "dependencies":
                deps = await self.extract_dependencies(repo_path)
                for dep_name, dep_info in deps.items():
                    # Check cache first
                    if dep_name in self.entity_cache:
                        entity_id = self.entity_cache[dep_name]
                    else:
                        entity_id = await self.create_or_get_entity(
                            name=dep_name,
                            entity_type="Technology",
                            properties=dep_info,
                            references=[repo_ref]
                        )
                        self.entity_cache[dep_name] = entity_id
                    discoveries["entities"].append(entity_id)
                    
            elif area == "architecture":
                patterns = await self.extract_patterns(repo_path)
                for pattern in patterns:
                    pattern_id = await self.create_or_get_entity(
                        name=pattern["name"],
                        entity_type="Concept",
                        properties=pattern["properties"],
                        references=[repo_ref]
                    )
                    discoveries["patterns"].append(pattern_id)
                    
                    # Record observation about pattern usage
                    mcp__knowledge__observe(
                        f"Pattern '{pattern['name']}' used in {pattern['occurrences']} places",
                        entity_id=pattern_id,
                        context="architecture-mining",
                        confidence=0.85,
                        created_by=self.agent_id
                    )
        
        # Generate mining summary insight
        if len(discoveries["entities"]) > 5:
            insight = mcp__knowledge__record_insight(
                f"Codebase analysis reveals {len(discoveries['entities'])} key dependencies and {len(discoveries['patterns'])} architectural patterns",
                strength=0.8,
                created_by=self.agent_id
            )
            
        return discoveries
```

# 3. Relationship Mapping

implements_rel = create_relationship( "Implements", lion_entity, memory_entity,
"research_agent", confidence=0.8 )

# 4. Observation Recording

observe( "Lion uses event sourcing for temporal consistency",
context="architecture-analysis", entity_id=lion_entity, references=[code_ref] )

# 5. Question Generation

ask_question( "How can we optimize the consolidation algorithm?",
domain="memory-systems", urgency=0.7, entity_id=memory_entity )

# 6. Insight Discovery

record_insight( "Event sourcing + temporal decay creates natural memory
consolidation", connections=[obs_note_id, question_note_id], strength=0.9 )

````
## Common Anti-Patterns to Avoid

### 1. Creating Entities Without References
```python
# ❌ BAD: No traceability
entity = mcp__knowledge__create_entity(
    name="Important Concept",
    entity_type="Concept",
    created_by="agent_001"
)

# ✅ GOOD: Always include references
ref = mcp__knowledge__create_reference(...)["reference_id"]
entity = mcp__knowledge__create_entity(
    name="Important Concept",
    entity_type="Concept",
    created_by="agent_001",
    references=[ref]
)
````

### 2. Ignoring Duplicate Errors

```python
# ❌ BAD: Silent failure
entity_id = None
try:
    entity_id = create_entity(...)["entity_id"]
except:
    pass  # Entity might not be created!

# ✅ GOOD: Handle duplicates properly
try:
    entity_id = create_entity(...)["entity_id"]
except Exception as e:
    if "Duplicate entity" in str(e):
        entity_id = str(e).split("ID: ")[1].strip()
        # Use existing entity
    else:
        raise  # Don't swallow other errors
```

### 3. Creating Orphaned Notes

```python
# ❌ BAD: Note not attached to anything
create_note(
    "System is fast",
    {"type": "observation"},
    "agent_001"
)

# ✅ GOOD: Always attach to entities
create_note(
    "System achieves 100k req/s",
    {"type": "observation", "context": "performance"},
    "agent_001",
    attachments=[{"target_type": "entity", "entity_id": system_id}]
)
```

### 4. Poor Session Management

```python
# ❌ BAD: No session coherence
create_note(content1, type1, agent_id)  # No session
create_note(content2, type2, agent_id)  # No session

# ✅ GOOD: Group related work
session_id = f"research_task_{timestamp}"
create_note(content1, type1, agent_id, session_id=session_id)
create_note(content2, type2, agent_id, session_id=session_id)
```

### 5. Forgetting to Reinforce Important Notes

```python
# ❌ BAD: Critical insight will decay
insight_id = record_insight(
    "Fundamental architectural pattern discovered",
    strength=0.95
)["note_id"]

# ✅ GOOD: Reinforce important discoveries
insight_id = record_insight(
    "Fundamental architectural pattern discovered",
    strength=0.95
)["note_id"]
mcp__knowledge__reinforce_note(insight_id)
```

### Problem Analysis Pattern

```python
# 1. Create issue entity
issue = create_entity("Performance Bottleneck", "Issue", agent_id)

# 2. Ask clarifying questions
questions = [
    ask_question("What causes the bottleneck?", urgency=0.9),
    ask_question("Which components are affected?", urgency=0.8)
]

# 3. Record observations
observations = gather_observations(issue)

# 4. Generate insights
if pattern_detected(observations):
    record_insight("Root cause identified", connections=observations)
```

### Knowledge Consolidation Pattern

```python
# 1. Query related notes
recent_notes = query("MATCH (n:Note) WHERE n.session_id = $session RETURN n")

# 2. Identify important notes
for note in recent_notes:
    if note.relevance > threshold:
        reinforce_note(note.id)

# 3. Create summary insight
record_insight(
    "Session discovered X new patterns",
    connections=[n.id for n in important_notes]
)
```

## Error Handling

```python
# Always check results
result = create_entity(...)
if result["success"]:
    entity_id = result["entity_id"]
else:
    # Handle error
    create_note(
        f"Failed to create entity: {result['error']}",
        {"type": "agent_error", "error_type": "data_error", "severity": "medium"},
        agent_id
    )
```

## Performance Considerations

1. **Batch Related Operations**: Create all references first, then entities,
   then relationships
2. **Use Specific Queries**: Avoid "MATCH (n) RETURN n" for large graphs
3. **Leverage Decay**: Let unimportant notes naturally decay instead of querying
   everything
4. **Session Grouping**: Use session IDs to efficiently query related work

## Integration with Task Agents

### Entity Extraction Pattern (Round 1)

```python
class EntityExtractionAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.session_id = f"{agent_id}_{timestamp}"
    
    def extract_entities_from_kb(self, domain_focus):
        # 1. Create references for sources
        refs = self.cite_actual_sources()
        
        # 2. READ actual files and extract entities
        entities = self.extract_entities_from_content(refs)
        
        # 3. Record observations about entities
        self.record_entity_findings(entities)
        
        # 4. Generate insights
        self.synthesize_knowledge()
```

### Relationship Extraction Pattern (Round 2) - EVIDENCE-BASED

```python
class RelationshipExtractionAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.session_id = f"{agent_id}_{timestamp}"
    
    def extract_relationships_from_kb(self, relationship_domain):
        # 1. READ actual kb_ files to find relationship evidence
        evidence_files = self.scan_kb_references_for_relationships()
        
        # 2. Create references for each file read
        refs = []
        for file_path in evidence_files:
            content = Read(file_path)
            ref_id = mcp__knowledge__create_reference(
                source_type="Documentation",
                title=f"KB Reference: {file_path.split('/')[-1]}",
                created_by=self.agent_id,
                url=f"file://{file_path}",
                metadata={"domain": relationship_domain, "content_length": len(content)}
            )["reference_id"]
            refs.append(ref_id)
        
        # 3. Query existing entities AND discover new ones
        existing_entities = mcp__knowledge__query("MATCH (n) RETURN n LIMIT 100")["entities"]
        
        # 4. DISCOVER NEW ENTITIES from content evidence
        discovered_entities = self.discover_entities_from_content(file_contents, refs)
        
        # 5. Combine existing and newly discovered entities
        all_entities = existing_entities + discovered_entities
        
        # 6. Extract relationships FROM actual content evidence
        relationships = []
        for entity1 in all_entities:
            for entity2 in all_entities:
                if entity1["id"] != entity2["id"]:
                    # Find evidence of relationship in read content
                    evidence = self.find_relationship_evidence(entity1, entity2, content)
                    if evidence:
                        rel_id = mcp__knowledge__create_relationship(
                            rel_type=evidence["relationship_type"],
                            source_id=entity1["id"],
                            target_id=entity2["id"],
                            created_by=self.agent_id,
                            properties={
                                "evidence": evidence["description"],
                                "source_file": evidence["source_file"],
                                "confidence_basis": evidence["strength"]
                            },
                            confidence=evidence["confidence"],
                            references=refs  # Critical: attach evidence
                        )["relationship_id"]
                        relationships.append(rel_id)
    
    def discover_entities_from_content(self, file_contents, refs):
        """Discover and create new entities from content analysis"""
        discovered_entities = []
        
        for file_path, content in file_contents.items():
            # Look for new entity patterns in content
            new_entity_candidates = self.extract_entity_candidates(content)
            
            for candidate in new_entity_candidates:
                # Check if entity already exists
                existing = mcp__knowledge__find_entity_by_name(candidate["name"])
                if not existing["entities"]:
                    # Create new entity with evidence
                    entity_id = mcp__knowledge__create_entity(
                        name=candidate["name"],
                        entity_type=candidate["type"],
                        created_by=self.agent_id,
                        properties=candidate["properties"],
                        confidence=candidate["confidence"],
                        references=refs
                    )["entity_id"]
                    
                    discovered_entities.append({
                        "id": entity_id,
                        "name": candidate["name"],
                        "entity_type": candidate["type"]
                    })
        
        return discovered_entities
    
    def extract_entity_candidates(self, content):
        """Extract potential new entities from content"""
        candidates = []
        lines = content.split('\n')
        
        for line in lines:
            # Look for patterns that suggest new entities
            if any(pattern in line.lower() for pattern in [
                "architecture", "pattern", "system", "service", "framework",
                "algorithm", "protocol", "methodology", "process", "workflow"
            ]):
                # Extract potential entity name and classify
                entity_candidate = self.parse_entity_from_line(line)
                if entity_candidate:
                    candidates.append(entity_candidate)
        
        return candidates
        
        # 5. Record observations about relationship patterns
        mcp__knowledge__observe(
            f"Found {len(relationships)} evidence-based relationships in {relationship_domain}",
            context="relationship_extraction",
            confidence=0.9,
            references=refs,
            created_by=self.agent_id
        )
        
        return relationships
    
    def find_relationship_evidence(self, entity1, entity2, content):
        """Find actual evidence of relationships in content"""
        name1 = entity1["name"].lower()
        name2 = entity2["name"].lower()
        
        # Look for explicit mentions of both entities together
        if name1 in content.lower() and name2 in content.lower():
            # Extract surrounding context
            lines = content.split('\n')
            evidence_lines = []
            for line in lines:
                if name1 in line.lower() and name2 in line.lower():
                    evidence_lines.append(line.strip())
            
            if evidence_lines:
                # Determine relationship type from context
                context = ' '.join(evidence_lines).lower()
                rel_type = self.infer_relationship_type(context, name1, name2)
                
                return {
                    "relationship_type": rel_type,
                    "description": evidence_lines[0][:200],
                    "source_file": "content_analysis",
                    "strength": "high",
                    "confidence": 0.85
                }
        return None
    
    def infer_relationship_type(self, context, name1, name2):
        """Infer relationship type from contextual evidence"""
        if any(word in context for word in ["uses", "utilizes", "leverages"]):
            return "Uses"
        elif any(word in context for word in ["depends", "requires", "needs"]):
            return "DependsOn"
        elif any(word in context for word in ["implements", "realizes", "provides"]):
            return "Implements"
        elif any(word in context for word in ["contains", "includes", "encompasses"]):
            return "Contains"
        elif any(word in context for word in ["coordinates", "manages", "orchestrates"]):
            return "Coordinates"
        else:
            return "RelatedTo"
```

### Document Processing Pattern (Round 3)

```python
class DocumentProcessingAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.session_id = f"{agent_id}_{timestamp}"
    
    def process_kb_documents(self, document_category):
        # 1. Systematically read kb_ documents
        # 2. Extract structured information
        # 3. Create comprehensive references
        # 4. Link to existing entities/relationships
        # 5. Generate structured summaries
        pass
```

## Query Patterns & Workarounds

### Working Query Patterns

```python
# 1. Get all entities (with filtering)
all_entities = mcp__knowledge__query("MATCH (n) RETURN n")["entities"]

# Filter by type locally
technologies = [e for e in all_entities if e["entity_type"] == "TECHNOLOGY"]
projects = [e for e in all_entities if e["entity_type"] == "PROJECT"]

# 2. Get entity with all its notes
entity_data = mcp__knowledge__get_entity(entity_id)
entity = entity_data["entity"]
notes = entity_data["notes"]

print(f"Entity: {entity['name']} has {len(notes)} notes attached")
for note in notes:
    print(f"  - {note['note_type']['type']}: {note['content'][:50]}...")

# 3. Search entities by properties
all_entities = mcp__knowledge__query("MATCH (n) RETURN n")["entities"]

# Find entities with specific property values
rust_projects = [
    e for e in all_entities 
    if e.get("properties", {}).get("language") == "rust"
]

# Find entities created by specific agent
agent_entities = [
    e for e in all_entities
    if e["created_by"] == "research_agent_001"
]

# Find recent entities
recent_entities = sorted(
    all_entities,
    key=lambda e: e["created_at"],
    reverse=True
)[:10]
```

### Query Workaround Strategies

```python
class KnowledgeQueryHelper:
    """Helper class for working around current query limitations"""
    
    @staticmethod
    def get_entities_by_type(entity_type: str) -> List[Dict]:
        """Get all entities of a specific type"""
        all_entities = mcp__knowledge__query("MATCH (n) RETURN n")["entities"]
        return [
            e for e in all_entities 
            if e["entity_type"] == entity_type.upper()
        ]
    
    @staticmethod
    def find_entities_by_property(key: str, value: Any) -> List[Dict]:
        """Find entities with specific property value"""
        all_entities = mcp__knowledge__query("MATCH (n) RETURN n")["entities"]
        return [
            e for e in all_entities
            if e.get("properties", {}).get(key) == value
        ]
    
    @staticmethod
    def get_entities_with_notes(min_notes: int = 1) -> List[Dict]:
        """Get entities that have attached notes"""
        all_entities = mcp__knowledge__query("MATCH (n) RETURN n")["entities"]
        entities_with_notes = []
        
        for entity in all_entities:
            entity_data = mcp__knowledge__get_entity(entity["id"])
            if len(entity_data["notes"]) >= min_notes:
                entities_with_notes.append({
                    "entity": entity,
                    "note_count": len(entity_data["notes"]),
                    "notes": entity_data["notes"]
                })
        
        return entities_with_notes
    
    @staticmethod
    def find_related_entities(entity_id: str) -> Dict[str, List]:
        """Find entities that might be related (via shared references or notes)"""
        entity_data = mcp__knowledge__get_entity(entity_id)
        entity = entity_data["entity"]
        
        # Get all entities
        all_entities = mcp__knowledge__query("MATCH (n) RETURN n")["entities"]
        
        related = {
            "by_references": [],
            "by_creator": [],
            "by_session": [],
            "by_type": []
        }
        
        # Find entities with shared references
        entity_refs = set(entity.get("references", []))
        for other in all_entities:
            if other["id"] == entity_id:
                continue
                
            other_refs = set(other.get("references", []))
            if entity_refs & other_refs:  # Intersection
                related["by_references"].append(other)
            
            # Same creator
            if other["created_by"] == entity["created_by"]:
                related["by_creator"].append(other)
            
            # Same type
            if other["entity_type"] == entity["entity_type"]:
                related["by_type"].append(other)
        
        return related

# Usage examples
helper = KnowledgeQueryHelper()

# Get all technologies
tech_entities = helper.get_entities_by_type("Technology")
print(f"Found {len(tech_entities)} technology entities")

# Find Rust projects
rust_projects = helper.find_entities_by_property("language", "rust")

# Get entities with many notes
active_entities = helper.get_entities_with_notes(min_notes=5)

# Find related entities
related = helper.find_related_entities(some_entity_id)
print(f"Found {len(related['by_references'])} entities with shared references")
```

## Performance Optimization Strategies

### 1. Batch Entity Creation

```python
async def batch_create_entities(entities_data: List[Dict], agent_id: str):
    """Create multiple entities efficiently with deduplication"""
    created = []
    existing = []
    
    for data in entities_data:
        try:
            entity_id = mcp__knowledge__create_entity(
                name=data["name"],
                entity_type=data["type"],
                created_by=agent_id,
                properties=data.get("properties", {}),
                confidence=data.get("confidence", 0.8),
                references=data.get("references", [])
            )["entity_id"]
            created.append((entity_id, data["name"]))
        except Exception as e:
            if "Duplicate entity" in str(e):
                entity_id = str(e).split("ID: ")[1].strip()
                existing.append((entity_id, data["name"]))
    
    print(f"Created {len(created)} new entities, found {len(existing)} existing")
    return created, existing
```

### 2. Efficient Note Accumulation

```python
def accumulate_observations(observations: List[str], entity_id: str, agent_id: str, context: str):
    """Efficiently accumulate multiple observations"""
    session_id = f"obs_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    note_ids = []
    
    for i, obs in enumerate(observations):
        # Use convenience function for speed
        mcp__knowledge__observe(
            content=obs,
            context=context,
            confidence=0.8,
            entity_id=entity_id,
            created_by=agent_id
        )
        
        # Small delay to prevent overwhelming the system
        if i % 10 == 0 and i > 0:
            time.sleep(0.1)
    
    return note_ids
```

### 3. Caching Strategy

```python
class EntityCache:
    """Local cache to reduce duplicate entity creation attempts"""
    
    def __init__(self):
        self.cache = {}  # name -> entity_id
        self.type_cache = defaultdict(list)  # type -> [entity_ids]
    
    def add(self, name: str, entity_type: str, entity_id: str):
        self.cache[name.lower()] = entity_id
        self.type_cache[entity_type].append(entity_id)
    
    def get(self, name: str) -> Optional[str]:
        return self.cache.get(name.lower())
    
    def get_by_type(self, entity_type: str) -> List[str]:
        return self.type_cache.get(entity_type, [])
    
    async def get_or_create(self, name: str, entity_type: str, properties: Dict, 
                           references: List[str], agent_id: str) -> str:
        """Get from cache or create new entity"""
        # Check cache first
        cached_id = self.get(name)
        if cached_id:
            return cached_id
        
        # Try to create
        try:
            entity_id = mcp__knowledge__create_entity(
                name=name,
                entity_type=entity_type,
                created_by=agent_id,
                properties=properties,
                references=references
            )["entity_id"]
            self.add(name, entity_type, entity_id)
            return entity_id
        except Exception as e:
            if "Duplicate entity" in str(e):
                entity_id = str(e).split("ID: ")[1].strip()
                self.add(name, entity_type, entity_id)
                return entity_id
            raise

# Usage
cache = EntityCache()
entity_id = await cache.get_or_create(
    "PostgreSQL", 
    "Technology", 
    {"category": "database"}, 
    [ref_id], 
    "agent_001"
)
```

## Known Limitations

1. **Type-specific Cypher queries** (e.g., `MATCH (n:Technology)`) are not yet
   working
2. **WHERE clauses** are parsed but not yet implemented
3. **Relationship queries** are not yet implemented
4. Use `get_entity()` for specific entities or filter results from
   `MATCH (n) RETURN n`

Remember: The Knowledge MCP is append-only at the agent interface. All
modifications come from later-stage processes. This ensures data integrity and
enables temporal reasoning by preserving the full history of knowledge
accumulation.
