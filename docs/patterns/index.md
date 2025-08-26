# Patterns

Proven multi-agent workflows that solve common problems.

## Available Patterns

### [Fan-Out/In](fan-out-in.md)

Distribute work to parallel agents, then synthesize results. **Use for**,
Research, analysis, brainstorming

### [Sequential Analysis](sequential-analysis.md)

Build understanding step-by-step through dependent operations. **Use for**,
Document processing, complex reasoning

### [Conditional Flows](conditional-flows.md)

Execute different paths based on runtime conditions. **Use for**, Dynamic
workflows, decision trees

### [ReAct with RAG](react-with-rag.md)

Combine reasoning with retrieval-augmented generation. **Use for**,
Knowledge-intensive tasks

### [Tournament Validation](tournament-validation.md)

Multiple approaches compete, best solution wins. **Use for**, Quality-critical
outputs

## Pattern Selection Guide

```python
def select_pattern(task):
    if task.needs_multiple_perspectives:
        return "fan-out-in"
    elif task.requires_step_by_step_building:
        return "sequential-analysis"
    elif task.has_conditional_logic:
        return "conditional-flows"
    elif task.needs_external_knowledge:
        return "react-with-rag"
    elif task.requires_best_quality:
        return "tournament-validation"
    else:
        return "simple-branch"
```

## Common Combinations

### Research Pipeline

```
Fan-Out (gather) → Sequential (analyze) → Fan-Out (verify)
```

### Document Processing

```
Sequential (extract) → Conditional (classify) → Fan-Out (process by type)
```

### Decision Making

```
Fan-Out (options) → Tournament (evaluate) → Sequential (implement)
```
