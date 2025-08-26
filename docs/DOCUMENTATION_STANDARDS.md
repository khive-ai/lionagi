# LionAGI Documentation Standards

_Practical guidelines for writing docs that actually help._

## Core Principle

**Show, don't tell.** Every page should have working code that solves a real
problem.

## Page Types & Templates

### 1. Pattern Pages (`/patterns/`)

````markdown
# [Pattern Name]

When you need to [problem this solves].

## The Pattern

```python
# Complete working example
from lionagi import Session, Branch, Builder

async def pattern_name():
    # Implementation
    pass

# Usage
result = await pattern_name()
```
````

## When It Works

- Scenario 1: [specific use case]
- Scenario 2: [another use case]

## Success Rate

~95% based on [context]

````
### 2. Cookbook Pages (`/cookbook/`)

```markdown
# [Solution Name]

[One sentence: what this builds]

## Problem

[2-3 sentences on the specific challenge]

## Solution

```python
# Full implementation
# Can be 50-200 lines
# Must be copy-paste ready
````

## Results

```
[Actual output from running the code]
```

## Customization

- To adapt for X: [change this]
- To scale up: [modify that]

````
### 3. Concept Pages (`/core-concepts/`)

```markdown
# [Concept Name]

[One sentence definition]

## Quick Example

```python
# Minimal example showing the concept
````

## Key Points

- Point 1: [essential info]
- Point 2: [essential info]
- Point 3: [essential info]

## Common Usage

```python
# Realistic example
```

````
### 4. Quickstart Pages

```markdown
# [Getting Started with X]

## Install

```bash
uv add lionagi
````

## First Example

```python
# Simplest possible working example
```

## Next Steps

- Try [pattern]
- Read about [concept]
- See [cookbook example]

````
## Code Standards

### Every Code Block Must:

1. **Run without modification** - Include all imports
2. **Show realistic usage** - Not just toy examples
3. **Handle errors gracefully** - At least try/except where it matters

```python
# GOOD: Complete and runnable
from lionagi import Branch, iModel
import asyncio

async def example():
    branch = Branch(chat_model=iModel(provider="openai", model="gpt-4"))
    try:
        result = await branch.chat("Analyze this")
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

# Run it
# result = asyncio.run(example())
````

```python
# BAD: Fragment without context
branch.chat("Analyze this")  # What's branch? How to run?
```

## Writing Style

### Keep It Simple

- **Short sentences** (max 20 words)
- **Active voice** ("Use X to..." not "X can be used to...")
- **Direct instructions** ("Do this" not "You might want to consider")
- **Skip the fluff** (No "In this section we will explore...")

### Show Success Metrics

When claiming something works, show evidence:

- "95% success rate" not "usually works"
- "2.3 second average" not "fast"
- "Handles 1000 req/sec" not "scalable"

## For AI Agents

### Pattern Recognition Format

Help AI agents understand when to use patterns:

```markdown
## When to Use

IF task requires parallel analysis: USE fan-out-in pattern ELIF task needs
step-by-step building: USE sequential-analysis pattern ELSE: USE single-branch
ReAct
```

### Executable Templates

Provide parameterized code AI can modify:

```python
async def orchestrate(roles: list[str], task: str):
    """Template AI agents can adapt."""
    branches = [Branch(system=f"You are a {role}") for role in roles]
    results = await asyncio.gather(*[b.chat(task) for b in branches])
    return synthesize(results)
```

## Documentation Workflow

### Adding New Docs

1. **Check if needed** - Does this solve a new problem?
2. **Pick the right type** - Pattern, cookbook, concept, or quickstart?
3. **Use the template** - Don't reinvent the structure
4. **Test the code** - Every example must run
5. **Get it merged** - Perfect is the enemy of done

### Updating Docs

- **Fix errors immediately** - Don't wait
- **Update metrics quarterly** - Keep data fresh
- **Add examples from issues** - Real problems, real solutions

## Quality Checklist

Before merging any doc:

- [ ] Code runs without errors
- [ ] Solves a real problem
- [ ] Uses appropriate template
- [ ] Includes actual output/metrics
- [ ] Links to related content

## What NOT to Document

- **Obvious things** - We have good docstrings
- **Every parameter** - API reference handles that
- **Theory without practice** - This isn't an academic paper
- **Features not in main** - Document what's shipped

## Examples of Good Docs

### Good Pattern Doc

- Clear problem statement
- Complete working code
- Success metrics
- When to use/not use

### Good Cookbook Entry

- Specific real-world scenario
- Full implementation
- Actual results
- How to customize

### Good Concept Page

- Simple definition
- Minimal example
- Key points only
- Practical usage

## Maintenance

### Quarterly Review

- Update success metrics
- Fix broken examples
- Remove outdated patterns
- Add new proven patterns

### Continuous

- Fix errors when found
- Add clarifications from support questions
- Update for API changes

---

**Remember**: If you wouldn't copy-paste it into your own project, don't put it
in the docs.
