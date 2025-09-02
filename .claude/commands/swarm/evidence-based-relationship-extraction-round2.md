# Evidence-Based Relationship Extraction - Round 2

## Mission: Extract Grounded Relationships from kb_ References

Deploy 50 individual Task agents to extract evidence-based relationships by
reading actual kb_ files and finding documented connections between entities.

## Critical Requirements

### 1. MANDATORY FILE READING

- Each agent MUST read 3-5 actual kb_ files
- Extract relationships only with textual evidence
- Create references for every file read
- No generic or assumed relationships

### 2. EVIDENCE-BASED EXTRACTION

- Only create relationships with supporting evidence from content
- Include evidence excerpts in relationship properties
- Attach source file references to every relationship
- Confidence based on strength of textual evidence

### 3. KNOWLEDGE MCP COMPLIANCE

- Follow updated usage guide patterns
- Create references before relationships
- Use evidence-based relationship properties
- Link all relationships to source files

## Agent Deployment Strategy

### Batch 1-10: Technology Stack Evidence Extraction (10 agents)

Each agent reads different sets of files and extracts technology relationships
with evidence:

**Agent 11: Database Integration Evidence Extractor**

- Files: `technical-architecture.md`, `database-integration.md`, architecture
  diaries
- Extract: Database ↔ Framework relationships with implementation evidence
- Evidence: Code snippets, configuration details, integration patterns

**Agent 12: API Framework Evidence Extractor**

- Files: `api-system-*.md`, service integration diaries
- Extract: API ↔ Service relationships with usage evidence
- Evidence: Endpoint definitions, request/response patterns

**Agent 13-20: [Continue with specific file assignments for each agent]**

### Batch 11-20: System Architecture Evidence Extraction (10 agents)

**Agent 21: Event-Driven Architecture Evidence Extractor**

- Files: `EVENT_DRIVEN_DEVELOPMENT_GUIDE.md`, event-related diaries
- Extract: Event producer ↔ consumer relationships with flow evidence
- Evidence: Event schemas, handler implementations, flow diagrams

### Batch 21-30: Implementation Evidence Extraction (10 agents)

### Batch 31-40: Knowledge Management Evidence Extraction (10 agents)

### Batch 41-50: Cross-Domain Evidence Extraction (10 agents)

## Evidence-Based Extraction Protocol

### Phase 1: File Reading (MANDATORY)

```python
# Read assigned files
file_paths = [
    ".khive/docs/technical-architecture.md",
    ".khive/notes/diaries/2025-06-01-memory-system-implementation.md",
    ".khive/notes/architecture/KHIVE_ARCHITECTURE.md"
]

references = []
file_contents = {}

for file_path in file_paths:
    content = Read(file_path)
    ref_id = mcp__knowledge__create_reference(
        source_type="Documentation",
        title=f"KB Reference: {file_path.split('/')[-1]}",
        created_by=agent_id,
        url=f"file://{file_path}",
        metadata={
            "domain": extraction_domain,
            "content_length": len(content),
            "extraction_session": session_id
        }
    )["reference_id"]
    references.append(ref_id)
    file_contents[file_path] = content
```

### Phase 2: Entity Discovery

```python
# Query existing entities to find relationship candidates
entities = mcp__knowledge__query("MATCH (n) RETURN n LIMIT 100")["entities"]

# Filter entities that appear in the read content
relevant_entities = []
for entity in entities:
    entity_name = entity["name"].lower()
    for file_path, content in file_contents.items():
        if entity_name in content.lower():
            relevant_entities.append({
                "entity": entity,
                "mentioned_in": file_path,
                "context": extract_context(content, entity_name)
            })
```

### Phase 3: Evidence-Based Relationship Extraction

```python
relationships_created = 0

for entity1_info in relevant_entities:
    for entity2_info in relevant_entities:
        if entity1_info["entity"]["id"] != entity2_info["entity"]["id"]:
            # Look for co-occurrence and relationship evidence
            evidence = find_relationship_evidence(
                entity1_info, 
                entity2_info, 
                file_contents
            )
            
            if evidence:
                rel_id = mcp__knowledge__create_relationship(
                    rel_type=evidence["relationship_type"],
                    source_id=entity1_info["entity"]["id"],
                    target_id=entity2_info["entity"]["id"],
                    created_by=agent_id,
                    properties={
                        "evidence_text": evidence["text_excerpt"][:500],
                        "source_file": evidence["source_file"],
                        "context_line": evidence["line_number"],
                        "extraction_method": "content_analysis",
                        "confidence_basis": evidence["strength_reason"]
                    },
                    confidence=evidence["confidence"],
                    references=references
                )["relationship_id"]
                relationships_created += 1

# Document extraction results
mcp__knowledge__observe(
    f"Extracted {relationships_created} evidence-based relationships from {len(file_paths)} files",
    context="evidence_extraction",
    confidence=0.9,
    references=references,
    created_by=agent_id
)
```

### Phase 4: Evidence Analysis Functions

```python
def find_relationship_evidence(entity1_info, entity2_info, file_contents):
    """Find textual evidence of relationships between entities"""
    name1 = entity1_info["entity"]["name"].lower()
    name2 = entity2_info["entity"]["name"].lower()
    
    for file_path, content in file_contents.items():
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Look for both entities in same line or nearby lines
            if name1 in line_lower and name2 in line_lower:
                context_lines = lines[max(0, i-2):min(len(lines), i+3)]
                context = '\n'.join(context_lines)
                
                relationship_type = infer_relationship_type(context, name1, name2)
                confidence = calculate_evidence_confidence(context, name1, name2)
                
                return {
                    "relationship_type": relationship_type,
                    "text_excerpt": line.strip(),
                    "source_file": file_path.split('/')[-1],
                    "line_number": i + 1,
                    "strength_reason": f"Co-occurrence with {relationship_type.lower()} context",
                    "confidence": confidence
                }
    
    # Check for cross-file relationships
    if entity1_info["mentioned_in"] != entity2_info["mentioned_in"]:
        # Look for indirect relationships through content analysis
        return check_cross_file_relationships(entity1_info, entity2_info, file_contents)
    
    return None

def infer_relationship_type(context, name1, name2):
    """Infer relationship type from textual context"""
    context_lower = context.lower()
    
    relationship_patterns = {
        "Uses": ["uses", "utilizes", "leverages", "employs", "relies on"],
        "DependsOn": ["depends on", "requires", "needs", "based on", "built on"],
        "Implements": ["implements", "realizes", "provides", "fulfills"],
        "Contains": ["contains", "includes", "encompasses", "has", "comprises"],
        "Coordinates": ["coordinates with", "manages", "orchestrates", "controls"],
        "Integrates": ["integrates with", "connects to", "interfaces with"],
        "Supports": ["supports", "enables", "facilitates", "assists"],
        "Creates": ["creates", "generates", "produces", "builds"]
    }
    
    for rel_type, patterns in relationship_patterns.items():
        if any(pattern in context_lower for pattern in patterns):
            return rel_type
    
    return "RelatedTo"

def calculate_evidence_confidence(context, name1, name2):
    """Calculate confidence based on evidence strength"""
    context_lower = context.lower()
    
    # High confidence: explicit action verbs
    if any(verb in context_lower for verb in ["implements", "uses", "depends", "creates"]):
        return 0.9
    
    # Medium confidence: structural relationships  
    if any(word in context_lower for word in ["architecture", "system", "framework"]):
        return 0.8
    
    # Lower confidence: general mentions
    return 0.7
```

## Quality Standards

### Evidence Requirements

- **Textual Evidence**: Every relationship must have supporting text excerpt
- **Source Attribution**: File name and line number for traceability
- **Context Preservation**: Surrounding context for relationship inference
- **Confidence Justification**: Reason for confidence score

### Relationship Quality

- **15+ relationships per agent** with textual evidence
- **Confidence scores 0.7+** based on evidence strength
- **Complete references** linking to source files
- **Rich properties** with evidence details

### File Coverage

- **3-5 files per agent** systematically read
- **No file overlap** between agents in same batch
- **Diverse content types**: docs, diaries, architecture notes
- **Cross-domain connections** when supported by evidence

## Success Metrics

- **750+ evidence-based relationships** with supporting text
- **150+ source files** systematically processed
- **100% traceability** - every relationship linked to evidence
- **Zero generic relationships** - all grounded in content

## Agent Signature Format

`[REL-EVIDENCE-{AGENT_NUM}-2025-06-29T{TIME}] EVIDENCE-BASED-ARCHITECT`

This approach ensures every relationship is grounded in actual textual evidence
from kb_ files, with complete traceability and supporting references.
