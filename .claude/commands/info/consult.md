# External Consultation Command

## Purpose

Systematically prepare external consultation requests when internal
problem-solving would be inefficient or when external expertise provides
significant time/token savings.

## When to Use

**Consult externally when:**

- Problem complexity exceeds internal knowledge depth
- Multiple failed solution attempts internally
- Time-sensitive decisions requiring specialized expertise
- Cost of consultation < cost of internal trial-and-error
- Need validation of complex architectural decisions
- Seeking industry best practices and proven patterns

**Don't consult when:**

- Simple debugging or straightforward implementation
- Internal resources can solve within reasonable time
- Problem requires deep context that's hard to transfer
- Security/privacy concerns with external sharing

## Consultation Framework

### 1. Problem Definition

```markdown
## Executive Summary

[2-3 sentences: What we need and why]

## Core Questions

1. **[Category]** - [Specific question]
2. **[Category]** - [Specific question]
3. **[Category]** - [Specific question]
```

### 2. Context Package

```markdown
## System Overview

[Architecture, key components, constraints]

## Current Implementation

[Code samples, current approach, what works/doesn't]

## Attempted Solutions

[What we tried, why it failed, lessons learned]
```

### 3. Clarification Questions

```markdown
## Scope Clarifications

1. **Evaluation Focus**: [What specifically to analyze]
2. **Metrics/Constraints**: [What we can measure/track]
3. **Output Format**: [How we want recommendations structured]
4. **Implementation Timeline**: [When we need to act]
5. **Change Tolerance**: [How much we can modify]
```

### 4. Expected Deliverables

```markdown
## Deliverables Request

1. **[Analysis Type]** - [What we need]
2. **[Recommendations]** - [Format and specificity]
3. **[Implementation Guide]** - [Step-by-step if needed]
4. **[Anti-Patterns]** - [What to avoid]
```

## Example: KHIVE System Consultation

### Problem Context

**Complex multi-agent orchestration system needs optimization**

**Core Questions:**

1. **Logic Flow** - How to optimize agent coordination and handoffs?
2. **Usage Patterns** - What are proven orchestration patterns for different
   task types?
3. **Prompt Engineering** - How to improve our role+domain composition prompts?
4. **Consolidation** - Best practices for synthesizing multi-agent outputs?

### System Package

````markdown
## KHIVE Overview

- Role + Domain composition model (12 roles × 61 domains)
- Orchestrator + Task Agents + Consolidation Specialist
- Artifact-based coordination, no direct agent communication
- Planning via 10 concurrent evaluation agents

## Key Architecture

```python
# Orchestration Planning
class OrchestrationPlanner:
    def evaluate_request(self, request: str) -> list[dict]:
        """10 agents evaluate in parallel"""
    def build_consensus(self, evaluations: list[dict]) -> str:
        """Synthesize recommendations"""

# Agent Composition  
class AgentComposer:
    def compose_agent(self, role: str, domains: str, context: str) -> dict:
        """Create specialized agent from role + domain(s)"""
```
````

## Workflow Example

```bash
khive plan "implement OAuth2" → agent recommendations → spawn agents → 
parallel execution → artifacts → consolidation → unified deliverable
```

### Current Metrics

- Cost per plan: ~$0.004
- Response time: ~2 seconds
- 12 roles, 61 domains available
- Parallel execution up to 8 agents

### Clarifications Needed

1. **Scope**: Both YAML role specs AND composed prompts
2. **Metrics**: Can add structured logging, quality scoring
3. **Format**: Separate analysis files by topic area
4. **Timeline**: Progressive implementation over 2-4 weeks

````
## Template Structure

### Minimal Consultation Request
```markdown
# [System/Problem] Consultation Request

## Executive Summary
[Problem statement + what we need]

## System Context  
[Key architecture/constraints in 3-5 bullets]

## Specific Questions
1. [Question 1]
2. [Question 2] 
3. [Question 3]

## Expected Output
[Format and deliverables needed]

## Clarifications
[Scope, metrics, timeline, constraints]
````

## Implementation Steps

1. **Identify consultation opportunity**
   - Clear problem definition
   - Cost/benefit analysis vs internal work

2. **Prepare context package**
   - Essential system overview
   - Code samples and current state
   - Previous attempts and failures

3. **Generate clarification questions**
   - Scope boundaries
   - Output format preferences
   - Implementation constraints

4. **Submit consultation request**
   - Include all context
   - Ask for clarifications upfront
   - Specify deliverable format

5. **Process recommendations**
   - Organize by implementation priority
   - Create action items
   - Track implementation progress

## Success Criteria

**Good consultation request:**

- Clear problem scope and objectives
- Sufficient context without overload
- Specific, actionable questions
- Realistic timeline and constraints
- Measurable success criteria

**Effective consultation outcome:**

- Actionable recommendations
- Clear implementation path
- Identifies anti-patterns to avoid
- Provides reusable patterns
- Demonstrates clear ROI vs internal effort

---

_Use this command when external expertise provides faster, higher-quality
solutions than internal trial-and-error approaches._

Arguments: $ARGUMENTS
