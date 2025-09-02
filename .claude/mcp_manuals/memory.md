# Memory Server User Guide for Claude Code Sessions

## 🚀 Quick Start

The Memory Server provides persistent memory storage across AI sessions using
MCP tools. Here's everything you need to use it effectively.

### Available Tools

| Tool             | Purpose                  | Example                                                                 |
| ---------------- | ------------------------ | ----------------------------------------------------------------------- |
| `save`           | Store new memories       | `save("User prefers TypeScript", type="preference", topics=["coding"])` |
| `search`         | Find memories            | `search("typescript preferences")`                                      |
| `update`         | Modify existing memories | `update("abc12345", {"content": "User strongly prefers TypeScript"})`   |
| `forget`         | Invalidate memories      | `forget("abc12345", reason="Outdated")`                                 |
| `search_by_type` | Find by category         | `search_by_type("preference", limit=10)`                                |
| `list_memories`  | Browse all memories      | `list_memories(limit=20, offset=0)`                                     |

## 📝 Using the Save Tool

### Basic Usage

```python
# Simple save
save("The user's main project is an e-commerce platform")

# With type and topics
save("User prefers vim keybindings", type="preference", topics=["editor", "vim"])

# With metadata
save("API endpoint changed to /v3/data", 
     type="fact", 
     topics=["api", "backend"],
     metadata={"version": "3.0", "breaking_change": True})
```

### Memory Types

- `note` - General observations (default)
- `preference` - User preferences
- `fact` - Factual information
- `event` - Time-based events

### Best Practices

- Use specific, descriptive content
- Add relevant topics for better search
- Include type for categorization
- Keep metadata minimal but useful

## 🔍 Using the Search Tool

### Basic Search

```python
# Natural language search
search("user preferences about editors")

# Limit results
search("recent meetings", limit=5)

# Filter by topics
search("database decisions", topics=["architecture", "backend"])

# Time-based search
search("what happened yesterday", time_range={"start": "2025-06-26"})
```

### Search Tips

- Semantic search understands meaning, not just keywords
- Use natural language queries
- Combine with topics for precision
- Results include relevance scores (0-1)

## 📝 Updating Memories

### Update Examples

```python
# Update content
update("abc12345", {"content": "User now prefers VS Code with vim mode"})

# Add/modify topics
update("def67890", {"topics": ["python", "ai", "machine-learning"]})

# Update metadata
update("ghi13579", {"metadata": {"priority": "high", "project": "ml-pipeline"}})
```

### Update Strategy

- Search first to find the memory ID
- Update instead of creating duplicates
- Preserve important context
- Version history is maintained

## 🗑️ Forgetting Memories

### Forget Examples

```python
# Simple forget
forget("abc12345")

# With reason
forget("def67890", reason="User changed preference")
```

### Important Notes

- Memories are invalidated, not deleted
- Maintains audit trail
- Cannot be undone
- Use sparingly

## 🏷️ Search by Type

### Type Search Examples

```python
# Get all preferences
search_by_type("preference", limit=20)

# Get recent events
search_by_type("event", limit=10)

# Get all facts
search_by_type("fact", limit=50)
```

## 📋 List All Memories

### Listing Examples

```python
# Get first 20 memories
list_memories()

# Get next page
list_memories(limit=20, offset=20)

# Get more at once
list_memories(limit=50)
```

## 💡 Common Patterns

### 1. Project Context Management

```python
# Save project context
save("Working on React e-commerce app with Stripe integration", 
     type="note", 
     topics=["project", "current"])

# Track decisions
save("Chose PostgreSQL over MySQL for JSON support", 
     type="fact", 
     topics=["project", "database", "decisions"])

# Remember TODOs
save("TODO: Implement cart abandonment emails", 
     type="event", 
     topics=["project", "tasks"])
```

### 2. User Preference Tracking

```python
# Communication style
save("User prefers concise explanations without code comments", 
     type="preference", 
     topics=["communication"])

# Technical preferences
save("User works primarily in TypeScript and Python", 
     type="preference", 
     topics=["languages"])

# Environment preferences
save("User uses macOS with VS Code", 
     type="preference", 
     topics=["environment"])
```

### 3. Learning & Knowledge Base

```python
# Save learned information
save("React.memo prevents unnecessary re-renders", 
     type="fact", 
     topics=["react", "performance"])

# Track solutions
save("Fixed performance issue using useMemo in ProductList component", 
     type="note", 
     topics=["react", "debugging", "performance"])
```

### 4. Session Continuity

```python
# Before ending session
save("Currently debugging authentication flow in auth.service.ts line 45", 
     type="note", 
     topics=["session", "debugging"])

# Starting new session
results = search("current debugging session", topics=["session"])
```

## 🔧 Practical Examples

### Example 1: Remembering User Preferences

```
User: "I always prefer to see the full error stack trace"
Assistant: I'll remember that preference for you.
[Uses: save("User prefers to see full error stack traces", type="preference", topics=["debugging", "errors"])]
```

### Example 2: Retrieving Project Context

```
User: "What were we working on last time?"
Assistant: Let me check our previous session notes.
[Uses: search("session current project", topics=["session", "project"])]
```

### Example 3: Updating Information

```
User: "Actually, I switched from Stripe to Square for payments"
Assistant: I'll update that information.
[First uses: search("stripe payments", topics=["project", "payments"])]
[Then uses: update("abc12345", {"content": "Using Square for payment processing (switched from Stripe)"})]
```

## 🚨 Important Considerations

### What NOT to Store

- Passwords, API keys, or secrets
- Personal identifying information
- Large code blocks (store references instead)
- Temporary debugging output

### Memory Management

- Periodically search and update outdated information
- Use forget for truly obsolete memories
- Keep content concise but descriptive
- Use consistent topic naming

### Performance Tips

- Limit search results appropriately (5-10 for most queries)
- Use topic filters to narrow searches
- Batch related saves when possible
- Search before creating potential duplicates

## 🛠️ Troubleshooting

### No Results Found

- Try broader search terms
- Remove topic filters
- Check if memories exist with list_memories()

### Wrong Memory Updated

- Use more characters from the ID
- Search again to confirm the right memory
- IDs are unique, so partial matches work

### Duplicate Memories

- Search before saving
- Update existing memories instead
- Use consistent phrasing

## 📚 Quick Reference Card

```python
# Save
save("content", type="preference", topics=["tag1", "tag2"])

# Search
search("query", limit=10, topics=["tag1"])

# Update
update("id12345", {"content": "new content"})

# Forget
forget("id12345", reason="outdated")

# List by type
search_by_type("preference", limit=20)

# Browse all
list_memories(limit=20, offset=0)
```

## 🎯 Memory Strategy

1. **Save frequently**: Don't wait for "important" things
2. **Be descriptive**: Write for your future self
3. **Use topics**: Consistent categorization helps
4. **Update don't duplicate**: Keep information current
5. **Review periodically**: Search and update old memories

---

Remember: The Memory Server is your persistent knowledge base across sessions.
The more you use it, the more valuable it becomes!
