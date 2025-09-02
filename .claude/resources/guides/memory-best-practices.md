# Memory Best Practices for KB System

## Overview

This document defines best practices for using the memory MCP system within the
KB lifecycle, ensuring memories are properly tied to PRs, issues, and specific
commits across multiple projects.

## Memory Creation Standards

### 1. Repository Context Pattern

Always include full repository context when creating memories:

```python
# PATTERN: {repo_owner}/{repo_name}#{issue_number}@{commit_sha}
# Example: ohdearquant/kb#123@42746ca

await mcp__memory__save(
    content="Discovered pgvector outperforms dedicated vector DBs for our scale",
    type="fact",
    topics=["vector-db", "performance", "pgvector"],
    metadata={
        "repo": "https://github.com/ohdearquant/kb",
        "issue": "https://github.com/ohdearquant/kb/issues/123",
        "pr": "https://github.com/ohdearquant/kb/pull/124",
        "commit": "42746ca",
        "research_id": "MEM_004"
    }
)
```

### 2. Memory Type Guidelines

| Type         | Use For                           | Priority | Example                                                     |
| ------------ | --------------------------------- | -------- | ----------------------------------------------------------- |
| `insight`    | New understanding or breakthrough | High     | "Parallel agent execution reduces research time by 80%"     |
| `solution`   | Concrete problem resolution       | High     | "Fixed race condition with mutex at Work.status"            |
| `decision`   | Architecture/design choices       | Critical | "Selected PostgreSQL+pgvector over Pinecone"                |
| `pattern`    | Recurring approaches              | High     | "Swarm pattern: parallel discovery → sequential validation" |
| `error`      | Failure modes to avoid            | Medium   | "Agent timeout at 30min due to memory leak"                 |
| `preference` | Team/project preferences          | Low      | "Team prefers YAML over JSON for configs"                   |

### 3. Cross-Project Memory Namespacing

When working across multiple projects, use clear namespacing:

```yaml
# Project-specific memory
content: "[claude-code] Fixed WebSocket reconnection issue"
topics: ["claude-code", "websocket", "bug-fix"]
metadata:
  project: "claude-code"
  repo: "https://github.com/ohdearquant/kb"

# Shared insight across projects
content: "PATTERN: Event-driven orchestration scales better than polling"
topics: ["architecture", "orchestration", "cross-project"]
metadata:
  applies_to: ["claude-code", "kb-system", "mcp-servers"]
  discovered_in: "https://github.com/ohdearquant/kb#456"
```

## Integration with Core Commands

### Session Start Memory Loading

```python
# session-start automatically loads relevant memories
async def load_session_context():
    # Load project-specific memories
    project_memories = await mcp__memory__search(
        query=f"project:{current_project}",
        topics=[current_project],
        time_range={"start": "7 days ago"}
    )
    
    # Load cross-project patterns
    patterns = await mcp__memory__search_by_type(
        type="pattern",
        limit=10
    )
    
    # Load unresolved issues
    open_issues = await mcp__memory__search(
        query="status:open OR unresolved",
        topics=["issue", current_project]
    )
```

### Session Complete Memory Capture

```python
# session-complete captures session insights
async def capture_session_insights():
    # Capture work done
    for commit in session_commits:
        await mcp__memory__save(
            content=f"Implemented: {commit.message}",
            type="fact",
            topics=[project, "implementation"],
            metadata={
                "commit": commit.sha,
                "files": commit.changed_files,
                "pr": extract_pr_from_branch(commit.branch)
            }
        )
    
    # Capture decisions made
    for decision in session_decisions:
        await mcp__memory__save(
            content=decision.summary,
            type="decision",
            topics=[project, decision.category],
            metadata={
                "rationale": decision.rationale,
                "alternatives": decision.alternatives,
                "commit": decision.implementing_commit
            }
        )
```

### Compact Conversation Memory Extraction

```python
# compact-convo extracts insights automatically
async def extract_conversation_insights(summary):
    # Extract breakthroughs
    for breakthrough in summary.breakthroughs:
        await mcp__memory__save(
            content=f"BREAKTHROUGH: {breakthrough.description}",
            type="insight",
            topics=["breakthrough", project],
            metadata={
                "conversation_id": summary.id,
                "timestamp": summary.timestamp,
                "context": breakthrough.context
            }
        )
    
    # Extract patterns
    for pattern in summary.patterns:
        await mcp__memory__save(
            content=f"PATTERN: {pattern.description}",
            type="pattern",
            topics=["pattern", pattern.category],
            metadata={
                "examples": pattern.examples,
                "applicability": pattern.scope
            }
        )
```

## Memory Lifecycle with KB Events

### 1. Research Phase Memory Capture

```yaml
# During research.proposed
memory:
  content: "Research proposed: {title}"
  type: "event"
  topics: ["research", "proposed", "{category}"]
  metadata:
    research_id: "{research_id}"
    issue: "https://github.com/{repo}/issues/{number}"
    hypothesis: "{hypothesis}"

# During research.active
memory:
  content: "Discovery: {finding}"
  type: "fact"
  topics: ["research", "finding", "{category}"]
  metadata:
    research_id: "{research_id}"
    confidence: "{confidence_level}"
    source: "{data_source}"
    commit: "{analysis_commit_sha}"
```

### 2. Decision Phase Memory Capture

```yaml
# During decision.ready
memory:
  content: "Decision: {selected_option} for {problem}"
  type: "decision"
  topics: ["decision", "{category}"]
  metadata:
    research_id: "{research_id}"
    pr: "https://github.com/{repo}/pull/{number}"
    rationale: "{decision_rationale}"
    alternatives_considered: ["{option_a}", "{option_b}"]

# During decision.approved
memory:
  content: "Approved: {decision_title}"
  type: "event"
  topics: ["decision", "approved", "{category}"]
  metadata:
    decision_id: "{decision_id}"
    approvers: ["{approver_list}"]
    implementation_pr: "https://github.com/{repo}/pull/{number}"
```

### 3. Implementation Phase Memory Capture

```yaml
# During implementation.active
memory:
  content: "Implementation insight: {learning}"
  type: "insight"
  topics: ["implementation", "{feature}"]
  metadata:
    pr: "https://github.com/{repo}/pull/{number}"
    commits: ["{sha1}", "{sha2}"]
    files_changed: ["{file1}", "{file2}"]

# Error patterns discovered
memory:
  content: "Error pattern: {error_description}"
  type: "error"
  topics: ["error", "{error_category}"]
  metadata:
    stack_trace: "{relevant_trace}"
    fix_commit: "{fix_sha}"
    pr: "https://github.com/{repo}/pull/{number}"
```

## Memory Search Patterns

### 1. Project-Specific Searches

```python
# Find all decisions for current project
decisions = await mcp__memory__search(
    query=f"project:{project_name} type:decision",
    topics=["decision", project_name],
    limit=20
)

# Find related issues
related = await mcp__memory__search(
    query=f"issue:{issue_number} OR references:{issue_number}",
    topics=[project_name]
)

# Find implementation patterns
patterns = await mcp__memory__search(
    query="implementation pattern success",
    topics=["pattern", "implementation"],
    time_range={"start": "30 days ago"}
)
```

### 2. Cross-Repository Searches

```python
# Find similar problems across projects
similar_issues = await mcp__memory__search(
    query=problem_description,
    limit=10
)

# Filter to high-confidence solutions
solutions = [m for m in similar_issues 
             if m.metadata.get('confidence') == 'high'
             and m.type == 'solution']
```

### 3. Commit-Aware Searches

```python
# Find all memories related to a commit
commit_memories = await mcp__memory__search(
    query=f"commit:{commit_sha} OR {commit_sha}",
    limit=20
)

# Find memories in a commit range
range_memories = await mcp__memory__search(
    query=f"commits BETWEEN {start_sha} AND {end_sha}",
    time_range={
        "start": commit_start_date,
        "end": commit_end_date
    }
)
```

## Best Practices Checklist

### ✅ When Creating Memories

- [ ] Include full repository URL in metadata
- [ ] Add issue/PR links with full URLs
- [ ] Reference specific commit SHAs for code changes
- [ ] Use consistent topic naming: `[project, category, subcategory]`
- [ ] Set appropriate memory type (insight, solution, decision, etc.)
- [ ] Add confidence levels for findings
- [ ] Include research_id for KB lifecycle items

### ✅ When Searching Memories

- [ ] Use semantic search for natural language queries
- [ ] Filter by type for specific categories
- [ ] Include time ranges for recent work
- [ ] Check for duplicates before creating new memories
- [ ] Search across projects for similar problems

### ✅ For Multi-Project Work

- [ ] Namespace project-specific memories clearly
- [ ] Mark cross-project patterns explicitly
- [ ] Include "applies_to" field for shared insights
- [ ] Use full GitHub URLs (not relative paths)
- [ ] Track which project discovered shared patterns

### ✅ For Long-Term Knowledge

- [ ] Regularly run memory-optimization-swarm
- [ ] Update outdated memories rather than creating duplicates
- [ ] Use "forget" for invalidated knowledge (maintains audit trail)
- [ ] Create knowledge graph relationships
- [ ] Document evolution of solutions across commits

## Automated Memory Triggers

### Git Hook Integration

```bash
# .git/hooks/post-commit
#!/bin/bash
# Auto-capture commit insights

COMMIT_SHA=$(git rev-parse HEAD)
COMMIT_MSG=$(git log -1 --pretty=%B)
ISSUE_NUM=$(echo $COMMIT_MSG | grep -oP '#\K\d+')

if [ ! -z "$ISSUE_NUM" ]; then
    claude code memory-add "Commit: $COMMIT_MSG" \
        --metadata "{\"commit\": \"$COMMIT_SHA\", \"issue\": \"#$ISSUE_NUM\"}"
fi
```

### PR Template Integration

````markdown
<!-- .github/pull_request_template.md -->

## Insights Captured

Please run the following to capture key insights:

```bash
claude code memory-add "YOUR_KEY_INSIGHT_HERE" --auto
```
````

- [ ] Implementation patterns documented
- [ ] Error patterns captured
- [ ] Performance insights recorded
- [ ] Architecture decisions saved

````
## Memory Maintenance

### Weekly Memory Review

```python
# Run weekly to maintain memory quality
async def weekly_memory_maintenance():
    # Find potential duplicates
    all_memories = await mcp__memory__list_memories(limit=100)
    
    # Group by similarity
    similar_groups = find_similar_memories(all_memories, threshold=0.85)
    
    # Consolidate duplicates
    for group in similar_groups:
        consolidated = consolidate_memories(group)
        await mcp__memory__update(
            id=group[0].id,
            updates={"content": consolidated.content}
        )
        
        # Forget duplicates
        for memory in group[1:]:
            await mcp__memory__forget(
                id=memory.id,
                reason="Consolidated into primary memory"
            )
````

---

_Memory is the foundation of learning. Every insight captured accelerates future
development._
