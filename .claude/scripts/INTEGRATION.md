# Integration Guide: Task Master with Orchestrator

## For Claude/Orchestrator

When you (the orchestrator) are processing KB tasks, here's how to use the task
master:

### 1. Before Starting Work

```python
# Check what's pending
Task: "Run python /Users/lion/liongate/.claude/scripts/task_master.py --list"
```

### 2. Get Next Task

```python
# See what to work on next
Task: "Run python /Users/lion/liongate/.claude/scripts/task_master.py --next"
```

This will tell you:

- Which issue to process
- What type of swarm to run
- The priority level

### 3. Process the Task

Based on what the task master says, run the appropriate swarm:

```python
# If it says "Run kb-intake-swarm"
Task: "Execute kb-intake-swarm for issue #123
      Reference: .claude/commands/swarm/kb-intake-swarm.md"

# If it says "Run kb-research-swarm"  
Task: "Execute kb-research-swarm for issue #456
      Reference: .claude/commands/swarm/kb-research-swarm.md"
```

### 4. CRITICAL: Before Declaring Completion

**You MUST run this check:**

```python
# Run the gatekeeper check
Task: "Run python /Users/lion/liongate/.claude/scripts/simple_gatekeeper.py"
```

If it returns:

- ✅ **"Task completion allowed"** - You can complete
- ❌ **"Task completion NOT allowed"** - Keep processing tasks

### 5. Continuous Loop Pattern

```python
# Pseudo-code for your main loop
while True:
    # Check for tasks
    result = Task("Run python /Users/lion/liongate/.claude/scripts/task_master.py --check")
    
    if no_tasks_pending:
        # Run gatekeeper final check
        can_complete = Task("Run python /Users/lion/liongate/.claude/scripts/simple_gatekeeper.py")
        if can_complete:
            break  # Done!
    
    # Get and process next task
    Task("Run python /Users/lion/liongate/.claude/scripts/task_master.py --next")
    # ... process the task ...
```

## Example Scenarios

### Scenario 1: Multiple Research Requests

```
Issues in GitHub:
- #123: "Research vector databases" (stage:research.requested)
- #124: "Research LLM memory" (stage:research.requested)
- #125: "Implement caching" (stage:research.active)

Task Master says:
1. Process #123 with kb-intake-swarm
2. Process #124 with kb-intake-swarm (can run in parallel!)
3. Process #125 with kb-research-swarm

Gatekeeper says: ❌ Cannot complete - 3 tasks pending
```

### Scenario 2: All Tasks Complete

```
Issues in GitHub:
- #123: "Research vector databases" (stage:knowledge.captured) ✅
- #124: "Old closed issue" (closed) ✅

Task Master says:
- No pending tasks!

Gatekeeper says: ✅ Task completion allowed
```

## Remember

The gatekeeper is your safety net. It prevents you from stopping work when there
are still KB lifecycle events to process. Always run it before declaring "I'm
done!"
