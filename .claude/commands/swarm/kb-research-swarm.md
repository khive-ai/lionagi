# KB Research Swarm

## Overview

Comprehensive parallel research execution following an approved research plan.
Deploys specialized agents to gather evidence, run experiments, and synthesize
findings for decision-making.

## Agent Composition

### Phase 1: Discovery Agents (Parallel)

#### 1. Codebase Explorer

```yaml
role: research.codebase_explorer
batch_size: 100 files
tools: [Glob, Grep, Read, Task, mcp__knowledge__*]
focus: "Deep analysis of current implementation and patterns"
knowledge:
  session_id: "{research_id}_codebase_exploration"
  entity_types: [Technology, Concept, Pattern]

exploration_strategy:
  - Pattern identification in existing code
  - Architecture impact analysis
  - Integration point mapping
  - Performance bottleneck discovery
  - Test coverage assessment

search_patterns:
  - Current implementation details
  - Similar solved problems
  - Integration patterns
  - Configuration approaches
  - Error handling strategies

knowledge_accumulation:
  - Create Technology entities for discovered tools
  - Map Uses/DependsOn relationships
  - Record observations about patterns
  - Ask questions about unclear implementations

output:
  - Knowledge graph entities and relationships
  - codebase_analysis.md (for GitHub issue)
  - pattern_inventory.yaml
  - integration_map.json
  - technical_constraints.md
```

#### 2. Memory Archaeologist

```yaml
role: research.memory_archaeologist
batch_size: 50 memories
tools: [mcp__memory__search, mcp__knowledge__*, Task]
focus: "Extract historical knowledge and lessons learned"
knowledge:
  session_id: "{research_id}_memory_archaeology"
  consolidate_from: memory_mcp

memory_mining:
  - Past similar implementations
  - Failed approaches to avoid
  - Performance discoveries
  - Architecture decisions
  - Team knowledge and expertise

knowledge_graph_queries:
  - Related concepts and patterns
  - Technology relationships
  - Decision rationale history
  - Success/failure patterns

knowledge_migration:
  - Convert memory facts to Knowledge entities
  - Create Insight notes from patterns
  - Link historical decisions to current entities
  - Record lessons as observations

output:
  - Migrated knowledge in Knowledge MCP
  - historical_insights.md (summary)
  - lessons_learned.yaml
  - decision_precedents.json
  - expertise_map.md
```

#### 3. Documentation Synthesizer

```yaml
role: research.documentation_synthesizer
batch_size: 30 documents
tools: [Read, WebSearch, WebFetch, Task]
focus: "Aggregate knowledge from all documentation sources"

source_hierarchy:
  internal:
    - Architecture documents
    - API specifications
    - README files
    - Code comments
    - Team wikis

  external:
    - Official documentation
    - Research papers
    - Industry benchmarks
    - Community best practices
    - Vendor comparisons

synthesis_approach:
  - Extract key concepts
  - Compare approaches
  - Identify best practices
  - Note limitations
  - Gather metrics

output:
  - documentation_synthesis.md
  - vendor_comparison.yaml
  - best_practices.json
  - external_research.md
```

### Phase 2: Experiment Agents (Parallel per Option)

#### 4. Option Evaluator (Multiple Instances)

```yaml
role: research.option_evaluator_{option_name}
batch_size: 5-10 tests per option
tools: [Bash, Write, Read, Task]
focus: "Evaluate specific solution option thoroughly"

evaluation_framework:
  technical:
    - Performance benchmarks
    - Scalability tests
    - Integration complexity
    - Maintenance burden
    - Security assessment

  business:
    - Cost analysis
    - Learning curve
    - Vendor stability
    - Community support
    - Future roadmap

experiments:
  - Proof of concept implementation
  - Load testing scenarios
  - Integration testing
  - Cost modeling
  - Team skill assessment

output:
  - option_{name}_evaluation.md
  - benchmark_results.yaml
  - cost_model.json
  - poc_code/
```

#### 5. Benchmark Runner

```yaml
role: research.benchmark_runner
batch_size: 10-20 scenarios
tools: [Bash, Write, mcp__memory__add]
focus: "Execute standardized performance comparisons"

benchmark_suite:
  - Baseline measurements
  - Load scenarios
  - Edge cases
  - Stress tests
  - Real-world simulations

metrics_collected:
  - Latency (p50, p95, p99)
  - Throughput (ops/sec)
  - Resource usage (CPU, memory)
  - Cost per operation
  - Failure rates

output:
  - benchmark_report.md
  - metrics_comparison.yaml
  - performance_graphs.json
  - raw_data/
```

### Phase 3: Synthesis Agent (Sequential)

#### 6. Findings Compiler

```yaml
role: research.findings_compiler
batch_size: ALL outputs
tools: [Read, Write, mcp__memory__add]
focus: "Synthesize all research outputs into coherent findings"

compilation_process:
  1. Aggregate all agent outputs
  2. Identify key insights
  3. Resolve contradictions
  4. Score confidence levels
  5. Extract decision factors
  6. Generate recommendations

synthesis_structure:
  - Executive summary
  - Detailed findings by option
  - Comparative analysis
  - Risk assessment
  - Recommendation rationale
  - Implementation considerations

output:
  - research_findings.md
  - decision_matrix.yaml
  - insight_catalog.json
  - recommendation.md
```

## Swarm Execution Pattern

```python
research_swarm = {
    "phase_1_discovery": [
        "Codebase Explorer",
        "Memory Archaeologist",
        "Documentation Synthesizer"
    ],
    
    "phase_2_evaluation": [
        "Option Evaluator[PostgreSQL]",
        "Option Evaluator[Qdrant]", 
        "Option Evaluator[Pinecone]",
        "Benchmark Runner"
    ],
    
    "phase_3_synthesis": [
        "Findings Compiler"
    ]
}

# Execution time: ~30 minutes (vs 3+ hours sequential)
```

## Input Format

```yaml
research_plan:
  research_id: "MEM_004"
  title: "Vector Database Evaluation"
  options_to_evaluate:
    - name: "PostgreSQL+pgvector"
      hypothesis: "Most cost-effective with good performance"
    - name: "Qdrant"
      hypothesis: "Best performance for our scale"
    - name: "Pinecone"
      hypothesis: "Lowest operational overhead"
  success_criteria:
    - metric: "query_latency_p95"
      target: "<50ms"
    - metric: "monthly_cost"
      target: "<$1000"
  methodology: "Comparative benchmarking with production-like data"
```

## Output Format

```yaml
research_summary:
  research_id: "MEM_004"
  completion_date: "2024-02-01"
  confidence_level: "high"

  findings:
    - finding: "PostgreSQL+pgvector achieves 45ms p95 latency"
      evidence: "benchmark_results/pgvector_load_test.json"
      confidence: 0.95

    - finding: "Qdrant provides 23ms p95 but costs $1200/month"
      evidence: "benchmark_results/qdrant_pricing.yaml"
      confidence: 0.90

    - finding: "Team has strong PostgreSQL expertise"
      evidence: "memory_insights/team_skills.md"
      confidence: 0.85

  recommendations:
    primary: "Adopt PostgreSQL+pgvector"
    rationale: "Meets performance requirements within budget with lowest risk"
    confidence: 0.88

  risks:
    - risk: "pgvector performance may degrade beyond 20M embeddings"
      likelihood: "medium"
      impact: "high"
      mitigation: "Plan for sharding strategy"
```

## Quality Gates

```yaml
validation_requirements:
  phase_1:
    - Min 3 historical references found
    - All current implementations documented
    - External research covers top 5 options

  phase_2:
    - Each option has POC code
    - Benchmarks run min 3 times
    - Cost models validated

  phase_3:
    - All findings have evidence
    - Confidence scores provided
    - Contradictions resolved
```

## Knowledge MCP Integration

```python
# Pre-research: Check existing knowledge
existing_entities = await mcp__knowledge__query(
    "MATCH (t:Technology) WHERE t.name IN ['PostgreSQL', 'Qdrant', 'Pinecone'] RETURN t"
)

# During research: Accumulate structured knowledge
for option in options_to_evaluate:
    # Create or reuse entity
    entity_id = await create_entity_safe(
        name=option.name,
        entity_type="Technology",
        agent_id=agent_id,
        properties={"category": "vector_database"},
        confidence=0.9
    )
    
    # Record observations
    await mcp__knowledge__observe(
        content=f"{option.name} achieves {metrics.p95_latency}ms p95 latency",
        context="performance_analysis",
        entity_id=entity_id,
        confidence=0.95
    )
    
    # Ask follow-up questions
    await mcp__knowledge__ask_question(
        question=f"How does {option.name} handle sharding at scale?",
        domain="distributed-systems",
        entity_id=entity_id,
        urgency=0.7
    )

# Post-research: Create synthesis insight
await mcp__knowledge__record_insight(
    insight="PostgreSQL+pgvector provides best ROI for our scale",
    connections=[obs1_id, obs2_id, obs3_id],
    strength=0.88
)
```

## Error Recovery

```yaml
failure_modes:
  agent_timeout:
    - Save partial results
    - Flag incomplete sections
    - Adjust scope if needed

  benchmark_failure:
    - Retry with smaller dataset
    - Use synthetic data
    - Document limitations

  insufficient_data:
    - Note gaps in findings
    - Adjust confidence scores
    - Recommend follow-up research
```
