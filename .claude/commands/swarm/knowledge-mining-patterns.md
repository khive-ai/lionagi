# Knowledge Mining Swarm Patterns

## Overview

Knowledge mining swarms are specialized patterns for systematic knowledge
extraction and graph construction using the Knowledge MCP. These patterns
emphasize deduplication, entity normalization, and structured knowledge
accumulation.

## Core Principles

1. **Entity Integrity**: Every entity must be deduplicated and normalized
2. **Reference Traceability**: All claims must link to sources
3. **Temporal Coherence**: Use session IDs to group related work
4. **Progressive Enhancement**: Build on existing knowledge, don't duplicate

## Knowledge Mining Decision Matrix

```yaml
decision_matrix:
  dimensions:
    - name: "Scope"
      options: ["Single Domain", "Cross-Domain", "Full Graph"]
      weight: 0.3

    - name: "Depth"
      options: ["Surface Scan", "Deep Dive", "Exhaustive"]
      weight: 0.3

    - name: "Novelty"
      options: ["Known Territory", "Adjacent Possible", "Frontier"]
      weight: 0.2

    - name: "Urgency"
      options: ["Exploratory", "Time-Sensitive", "Critical"]
      weight: 0.2

  strategy_selection:
    - score: [0.8, 1.0]
      pattern: "Deep Mining Swarm"
      agents: 15-20
      rounds: 3-5

    - score: [0.5, 0.8]
      pattern: "Focused Mining Swarm"
      agents: 7-12
      rounds: 2-3

    - score: [0.0, 0.5]
      pattern: "Quick Survey Swarm"
      agents: 3-5
      rounds: 1-2
```

## Entity Normalization Protocol

```python
class EntityNormalizer:
    """Comprehensive entity normalization to prevent duplicates"""
    
    # Common abbreviations and variations
    ABBREVIATIONS = {
        "Incorporated": "Inc",
        "Corporation": "Corp",
        "Limited": "Ltd",
        "Company": "Co",
        "International": "Intl",
        "Association": "Assoc",
        "Department": "Dept",
        "University": "Univ"
    }
    
    # Technology name variations
    TECH_ALIASES = {
        "JS": "JavaScript",
        "TS": "TypeScript", 
        "K8s": "Kubernetes",
        "k8s": "Kubernetes",
        "ML": "Machine Learning",
        "AI": "Artificial Intelligence",
        "LLM": "Large Language Model",
        "DB": "Database"
    }
    
    @staticmethod
    def normalize(name: str, entity_type: str) -> str:
        """Normalize entity name based on type"""
        
        # Step 1: Clean whitespace
        name = " ".join(name.strip().split())
        
        # Step 2: Handle technology aliases
        if entity_type == "Technology":
            for alias, full in EntityNormalizer.TECH_ALIASES.items():
                if name.upper() == alias.upper():
                    return full
        
        # Step 3: Standardize case
        if entity_type in ["Person", "Organization"]:
            # Preserve proper nouns
            name = name.title()
        elif entity_type == "Technology":
            # Preserve tech naming (camelCase, etc)
            if not (name.isupper() and len(name) > 4):
                pass  # Keep original
            else:
                name = name.title()
        
        # Step 4: Normalize abbreviations
        for long, short in EntityNormalizer.ABBREVIATIONS.items():
            name = name.replace(f" {long}", f" {short}")
            name = name.replace(f" {long.lower()}", f" {short}")
        
        # Step 5: Remove special characters
        name = name.replace("'s", "")
        name = name.replace("&", "and")
        
        return name

    @staticmethod
    def generate_search_variations(name: str) -> List[str]:
        """Generate variations for duplicate checking"""
        variations = [name]
        
        # Add without spaces
        variations.append(name.replace(" ", ""))
        
        # Add with dashes
        variations.append(name.replace(" ", "-"))
        
        # Add acronym if multi-word
        words = name.split()
        if len(words) > 1:
            acronym = "".join(w[0].upper() for w in words)
            variations.append(acronym)
        
        return variations
```

## Knowledge Mining Tree Structure

```yaml
knowledge_tree:
  root:
    query: "MATCH (n:Entity) WHERE n.entity_type = 'Project' RETURN n"
    expand_strategy: "breadth_first"

  levels:
    - depth: 1
      name: "Direct Dependencies"
      relationship_types: ["Uses", "DependsOn", "Implements"]
      agent_count: 5
      parallel: true

    - depth: 2
      name: "Transitive Dependencies"
      relationship_types: ["*"] # All relationships
      agent_count: 10
      parallel: true
      batch_size: 50

    - depth: 3
      name: "Ecosystem Connections"
      relationship_types: ["RelatedTo", "Influences", "Competes"]
      agent_count: 7
      parallel: true
      stop_condition: "no_new_entities"

  pruning_rules:
    - rule: "Low relevance"
      condition: "confidence < 0.5"
      action: "skip_subtree"

    - rule: "Circular reference"
      condition: "entity in ancestors"
      action: "mark_cycle"

    - rule: "Max depth reached"
      condition: "depth > 5"
      action: "stop_expansion"
```

## Swarm Pattern: Deep Knowledge Mining

```yaml
pattern: deep_knowledge_mining
type: hybrid
rounds: 3-5
agents_per_round: [5, 10, 7, 5, 3]

phases:
  - name: "Entity Discovery"
    round: 1
    type: parallel
    agents: 5
    tasks:
      - "Scan sources for entity mentions"
      - "Normalize and deduplicate entities"
      - "Create entity records with references"
    deduplication:
      strategy: "check_before_create"
      normalization: true

  - name: "Relationship Mapping"
    round: 2
    type: parallel
    agents: 10
    tasks:
      - "Identify relationships between entities"
      - "Classify relationship types"
      - "Record relationship properties"
    batch_strategy:
      entities_per_agent: 20
      relationship_limit: 100

  - name: "Deep Analysis"
    round: 3
    type: parallel
    agents: 7
    tasks:
      - "Record detailed observations"
      - "Ask clarifying questions"
      - "Identify patterns"
    knowledge_tools:
      - mcp__knowledge__observe
      - mcp__knowledge__ask_question

  - name: "Pattern Recognition"
    round: 4
    type: parallel
    agents: 5
    tasks:
      - "Analyze observation clusters"
      - "Generate insights"
      - "Connect related notes"
    min_pattern_size: 3
    confidence_threshold: 0.7

  - name: "Knowledge Synthesis"
    round: 5
    type: sequential
    agents: 3
    tasks:
      - "Review all discoveries"
      - "Create meta-insights"
      - "Reinforce key findings"
    final_actions:
      - reinforce_top_insights
      - create_session_summary
```

## Entity Log Tracking Protocol

```python
class EntityLogger:
    """Track entity operations to prevent duplicates and monitor coverage"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.entity_log = {}
        self.operation_count = {
            "queries": 0,
            "creates": 0,
            "duplicates_prevented": 0,
            "relationships": 0,
            "observations": 0
        }
    
    def log_entity_check(self, name: str, normalized: str, 
                        entity_type: str, found_id: Optional[str]):
        """Log entity lookup attempt"""
        key = f"{entity_type}:{normalized}"
        
        if key not in self.entity_log:
            self.entity_log[key] = {
                "original_names": [],
                "normalized": normalized,
                "entity_id": found_id,
                "check_count": 0,
                "created_at": None
            }
        
        self.entity_log[key]["original_names"].append(name)
        self.entity_log[key]["check_count"] += 1
        self.operation_count["queries"] += 1
        
        if found_id:
            self.entity_log[key]["entity_id"] = found_id
            self.operation_count["duplicates_prevented"] += 1
    
    def log_entity_creation(self, name: str, normalized: str, 
                           entity_type: str, entity_id: str):
        """Log successful entity creation"""
        key = f"{entity_type}:{normalized}"
        
        if key in self.entity_log:
            self.entity_log[key]["entity_id"] = entity_id
            self.entity_log[key]["created_at"] = datetime.now()
        else:
            self.entity_log[key] = {
                "original_names": [name],
                "normalized": normalized,
                "entity_id": entity_id,
                "check_count": 1,
                "created_at": datetime.now()
            }
        
        self.operation_count["creates"] += 1
    
    def get_entity_id(self, name: str, entity_type: str) -> Optional[str]:
        """Quick lookup from log before querying MCP"""
        normalized = EntityNormalizer.normalize(name, entity_type)
        key = f"{entity_type}:{normalized}"
        
        if key in self.entity_log:
            return self.entity_log[key].get("entity_id")
        return None
    
    def generate_report(self) -> Dict:
        """Generate deduplication report"""
        return {
            "session_id": self.session_id,
            "statistics": self.operation_count,
            "unique_entities": len([e for e in self.entity_log.values() 
                                   if e["entity_id"]]),
            "duplicate_variations": sum(len(e["original_names"]) - 1 
                                      for e in self.entity_log.values()),
            "most_referenced": sorted(
                [(k, v["check_count"]) for k, v in self.entity_log.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
```

## Swarm Pattern: Focused Knowledge Survey

```yaml
pattern: focused_knowledge_survey
type: parallel
rounds: 2
agents_per_round: [3, 2]

configuration:
  target_entities: 50
  max_depth: 2
  confidence_threshold: 0.6

phases:
  - name: "Quick Scan"
    round: 1
    agents:
      - entity_scanner_1:
          focus: "Primary domain"
          entity_quota: 20
      - entity_scanner_2:
          focus: "Adjacent domains"
          entity_quota: 20
      - entity_scanner_3:
          focus: "Dependencies"
          entity_quota: 10

  - name: "Relationship Discovery"
    round: 2
    agents:
      - relationship_mapper_1:
          focus: "Direct connections"
          max_relationships: 50
      - relationship_mapper_2:
          focus: "Transitive connections"
          max_relationships: 30

knowledge_quality_gates:
  - gate: "Entity completeness"
    check: "All entities have descriptions"
    action: "Request descriptions"

  - gate: "Reference coverage"
    check: "70% entities have references"
    action: "Find supporting sources"

  - gate: "Relationship balance"
    check: "No orphan entities"
    action: "Find connections"
```

## Deduplication Strategies

### Strategy 1: Preventive Deduplication

```python
async def create_entity_safe(name: str, entity_type: str, 
                           agent_id: str, **kwargs) -> str:
    """Create entity only if it doesn't exist"""
    
    # Step 1: Normalize
    normalized = EntityNormalizer.normalize(name, entity_type)
    
    # Step 2: Check log first (fast)
    entity_id = entity_logger.get_entity_id(name, entity_type)
    if entity_id:
        return entity_id
    
    # Step 3: Query Knowledge MCP
    existing = await check_entity_exists(normalized, entity_type)
    if existing:
        entity_logger.log_entity_check(name, normalized, entity_type, existing)
        return existing
    
    # Step 4: Create new entity
    result = await mcp__knowledge__create_entity(
        name=normalized,
        entity_type=entity_type,
        created_by=agent_id,
        **kwargs
    )
    
    entity_id = result["entity_id"]
    entity_logger.log_entity_creation(name, normalized, entity_type, entity_id)
    
    return entity_id
```

### Strategy 2: Batch Deduplication

```python
async def deduplicate_batch(entities: List[Dict]) -> List[Dict]:
    """Deduplicate a batch of entities before creation"""
    
    # Group by normalized name
    normalized_groups = {}
    for entity in entities:
        normalized = EntityNormalizer.normalize(
            entity["name"], 
            entity["entity_type"]
        )
        key = f"{entity['entity_type']}:{normalized}"
        
        if key not in normalized_groups:
            normalized_groups[key] = []
        normalized_groups[key].append(entity)
    
    # Check existence for unique names
    deduplicated = []
    for key, group in normalized_groups.items():
        # Use the entity with highest confidence
        best = max(group, key=lambda e: e.get("confidence", 0.5))
        
        # Check if exists
        entity_type, normalized = key.split(":", 1)
        existing_id = await check_entity_exists(normalized, entity_type)
        
        if existing_id:
            best["entity_id"] = existing_id
            best["action"] = "reused"
        else:
            best["name"] = normalized  # Use normalized name
            best["action"] = "create"
            
        deduplicated.append(best)
    
    return deduplicated
```

## Knowledge Mining Metrics

```yaml
mining_metrics:
  coverage:
    - total_entities_discovered
    - unique_entities_created
    - duplicate_entities_prevented
    - entity_types_distribution

  quality:
    - entities_with_descriptions: ">90%"
    - entities_with_references: ">80%"
    - average_confidence_score: ">0.75"
    - orphan_entities: "<5%"

  efficiency:
    - deduplication_rate
    - queries_per_entity
    - time_per_entity
    - agent_utilization

  knowledge_density:
    - relationships_per_entity
    - observations_per_entity
    - insights_per_session
    - cross_references_count
```

## Best Practices for Knowledge Mining

1. **Always Normalize First**: Use EntityNormalizer before any entity operation
2. **Check Before Create**: Query existing entities to prevent duplicates
3. **Use Entity Logger**: Track all operations for reporting and optimization
4. **Batch Similar Entities**: Process related entities together for efficiency
5. **Set Clear Targets**: Define entity quotas and relationship limits
6. **Progressive Enhancement**: Build on existing knowledge graph
7. **Session Coherence**: Use consistent session IDs for related work
8. **Quality Gates**: Implement checks between mining phases
9. **Cite Sources**: Every entity should have at least one reference
10. **Review and Reinforce**: Strengthen high-value discoveries

## Common Pitfalls and Solutions

| Pitfall                   | Impact                        | Solution                      |
| ------------------------- | ----------------------------- | ----------------------------- |
| Case-sensitive duplicates | "Redis" vs "redis"            | Normalize with context        |
| Abbreviation variants     | "ML" vs "Machine Learning"    | Maintain alias map            |
| Partial name matches      | "React" vs "React.js"         | Fuzzy matching with threshold |
| Temporal variations       | "GPT-3" vs "GPT-3.5"          | Version as property           |
| Compound entities         | "React Native" treated as two | Preserve phrases              |
| Unicode variations        | "café" vs "cafe"              | ASCII normalization           |

## Integration Example

```python
async def run_knowledge_mining_swarm(topic: str, depth: int = 2):
    """Execute a complete knowledge mining operation"""
    
    # Initialize tracking
    session_id = f"mining_{topic}_{datetime.now().isoformat()}"
    entity_logger = EntityLogger(session_id)
    
    # Phase 1: Entity Discovery (5 agents)
    print(f"Starting entity discovery for {topic}...")
    entity_agents = []
    for i in range(5):
        agent = asyncio.create_task(
            discover_entities(topic, entity_logger, f"scanner_{i}")
        )
        entity_agents.append(agent)
    
    entities = await asyncio.gather(*entity_agents)
    discovered_count = sum(len(e) for e in entities)
    print(f"Discovered {discovered_count} potential entities")
    
    # Phase 2: Deduplication
    all_entities = [e for sublist in entities for e in sublist]
    unique_entities = await deduplicate_batch(all_entities)
    print(f"Reduced to {len(unique_entities)} unique entities")
    
    # Phase 3: Relationship Mapping (10 agents)
    print("Mapping relationships...")
    rel_agents = []
    batch_size = len(unique_entities) // 10
    for i in range(10):
        start = i * batch_size
        end = start + batch_size if i < 9 else len(unique_entities)
        agent = asyncio.create_task(
            map_relationships(unique_entities[start:end], f"mapper_{i}")
        )
        rel_agents.append(agent)
    
    await asyncio.gather(*rel_agents)
    
    # Phase 4: Deep Analysis (7 agents)
    print("Performing deep analysis...")
    analysis_agents = []
    for i in range(7):
        agent = asyncio.create_task(
            analyze_entities(unique_entities, f"analyst_{i}")
        )
        analysis_agents.append(agent)
    
    await asyncio.gather(*analysis_agents)
    
    # Phase 5: Generate Report
    report = entity_logger.generate_report()
    print(f"\nKnowledge Mining Complete:")
    print(f"- Unique entities: {report['unique_entities']}")
    print(f"- Duplicates prevented: {report['duplicate_variations']}")
    print(f"- Total operations: {sum(report['statistics'].values())}")
    
    return report
```

Remember: Knowledge mining is about building a high-quality, deduplicated
knowledge graph that serves as the foundation for AGI development. Quality over
quantity!
