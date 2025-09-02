# Research Request Template

## Purpose

Standard template for creating research requests that trigger the KB lifecycle.

## Usage

Copy this template when creating new GitHub issues for research requests.

## Template

```markdown
---
title: "[RESEARCH] {Brief descriptive title}"
labels: ["research-request", "stage:research.requested", "priority:{high|medium|low}"]
assignees: []
projects: ["Research Requests"]
---

## Research Request

**Problem Statement**: {Clear, specific description of the problem to be solved}

**Business Impact**: {Why this research matters - revenue, efficiency, risk
reduction, etc.}

**Desired Outcome**: {What success looks like - specific, measurable outcomes}

## Context

**Background**: {Additional context about the problem}

**Current State**: {How things work today}

**Stakeholders**: {Who is affected by this problem}

**Timeline**: {When do you need results}

## Constraints

- [ ] **Budget**: ${amount} or {X hours}
- [ ] **Timeline**: {deadline}
- [ ] **Technical**: {any technical limitations}
- [ ] **Regulatory**: {compliance requirements}
- [ ] **Dependencies**: {what else needs to happen first}

## Success Criteria

What would make this research successful:

1. {Criterion 1 - should be measurable}
2. {Criterion 2 - should be specific}
3. {Criterion 3 - should have clear validation}

## Research Category

- [ ] AIO (AI/Automation/Optimization)
- [ ] MEM (Memory/Knowledge/Learning)
- [ ] TLI (Tools/Languages/Infrastructure)
- [ ] ARC (Architecture/Design/Patterns)
- [ ] DEV (Development/Process/Quality)
- [ ] UXP (User Experience/Product/Interface)

## Additional Information

{Any other relevant information, links, or resources}

---

<!-- Auto-populated by intake agent -->

Research ID: {Will be assigned} Created: {Timestamp} Priority: {Will be
assessed}
```

## Field Guidelines

### Problem Statement

- Be specific and concrete
- Avoid generic terms like "improve performance"
- Include measurable aspects where possible
- Example: "API response times exceed 500ms for 15% of requests during peak
  hours"

### Business Impact

- Quantify when possible ($X revenue, Y% efficiency gain)
- Connect to business objectives
- Include opportunity cost
- Example: "Slow API responses cause 3% cart abandonment, costing ~$50k/month"

### Desired Outcome

- Make it testable/verifiable
- Include metrics and targets
- Specify timeframe
- Example: "Reduce P95 API response time to <200ms within 6 weeks"

### Constraints

- Be realistic about limitations
- Include hidden constraints (team availability, etc.)
- Consider regulatory/compliance needs
- Example: "Must maintain GDPR compliance, limited to current budget"

## Examples

### Example 1: Performance Research

```markdown
---
title: "[RESEARCH] Vector Database Performance at 10M+ Scale"
labels: ["research-request", "stage:research.requested", "priority:high"]
---

## Research Request

**Problem Statement**: Current PostgreSQL setup struggles with vector similarity
searches above 8M embeddings, with query times reaching 2-3 seconds.

**Business Impact**: Search latency directly affects user experience and
conversion rates. Each 100ms delay reduces conversions by 1%, costing
approximately $25k/month.

**Desired Outcome**: Identify optimal vector database solution that maintains
<50ms P95 query latency at 15M+ embedding scale.

## Context

**Background**: User search feature is core to product experience. Growth
projections show 15M embeddings by Q3.

**Current State**: PostgreSQL with pgvector extension, 8M embeddings, degrading
performance.

**Stakeholders**: Product team (user experience), Engineering (maintenance),
Finance (costs)

**Timeline**: Decision needed by end of month for Q2 implementation.

## Constraints

- [x] **Budget**: <$2000/month additional infrastructure costs
- [x] **Timeline**: Must implement by end of Q2
- [x] **Technical**: Must integrate with existing Python/FastAPI stack
- [x] **Dependencies**: Requires approval for new infrastructure

## Success Criteria

1. Query latency P95 <50ms at 15M embedding scale
2. Infrastructure costs <$2000/month additional
3. Implementation complexity allows 4-week deployment
```

### Example 2: Architecture Research

```markdown
---
title: "[RESEARCH] Microservices Migration Strategy for Monolith"
labels: ["research-request", "stage:research.requested", "priority:medium"]
---

## Research Request

**Problem Statement**: Current monolithic architecture limits team velocity and
deployment flexibility, with 15-minute builds and deployment coupling.

**Business Impact**: Slow deployment cycle reduces feature velocity by ~30%,
affecting competitive positioning and time-to-market.

**Desired Outcome**: Clear migration strategy that improves deployment frequency
while maintaining system reliability.

## Context

**Background**: 150k LOC Python monolith serving 100k+ daily users. Team has
grown to 12 engineers across 3 squads.

**Current State**: Single deployment pipeline, shared database, 15-min builds,
weekly deployments.

**Stakeholders**: Engineering teams, DevOps, Product management

**Timeline**: Strategy needed for Q3 planning cycle

## Constraints

- [x] **Budget**: Migration must fit within current engineering capacity
- [x] **Timeline**: 6-month maximum migration timeline
- [x] **Technical**: Zero-downtime migration required
- [x] **Regulatory**: Maintain SOC2 compliance throughout
```

## Validation Checklist

Before submitting research request:

- [ ] Problem statement is specific and measurable
- [ ] Business impact is quantified where possible
- [ ] Desired outcome is testable/verifiable
- [ ] At least 3 constraints specified
- [ ] Success criteria are measurable
- [ ] Research category selected
- [ ] Timeline is realistic
- [ ] Stakeholders identified

## Automation Notes

This template triggers:

1. **Event 001**: research.requested
2. **Agent Assignment**: Research Intake Agent
3. **Label Updates**: Automatic stage tracking
4. **Process Flow**: Moves to research.proposed upon completion

The intake agent will:

- Assign research ID (format: CATEGORY_###)
- Validate request completeness
- Generate formal research proposal
- Assess priority and complexity
- Estimate resource requirements
