# Event 001: Research Requested

## Event Definition

```yaml
event:
  id: "001"
  name: "research.requested"
  description: "New research request submitted via GitHub issue"
  triggers:
    - "Issue created with 'research-request' label"
    - "Issue created in 'Research Requests' project"
  stage: "intake"
  parallelizable: true
```

## Trigger Conditions

- GitHub issue opened
- Has label: `research-request` OR `stage:research.requested`
- No existing `research_id` assigned
- Issue body contains required sections

## Required Issue Structure

```markdown
### Research Request

**Problem Statement**: [Clear description of the problem] **Business Impact**:
[Why this matters] **Desired Outcome**: [What success looks like]

### Context

[Additional background information]

### Constraints

- [ ] Budget: $X
- [ ] Timeline: Y weeks
- [ ] Technical: [any limitations]
```

## Agent Assignment

- **Primary**: Research Intake Agent
- **Support**: Context Discovery Agent (if needed)

## Expected Outputs

1. Research proposal (YAML format)
2. Research ID assignment
3. Initial categorization (AIO/MEM/TLI/ARC/DEV/UXP)
4. Priority assessment

## State Transitions

- **Success**: → Event 002 (research.proposed)
- **Failure**: → Remains in research.requested with error comment
- **Blocked**: → If missing required information

## Validation Rules

- Problem statement must be specific and measurable
- Business impact must be clearly articulated
- At least one constraint must be defined
- Issue author must have research request permissions

## GitHub Label Updates

```yaml
on_start:
  add: ["status:in-progress", "agent:intake"]
  remove: ["status:new"]

on_success:
  add: ["stage:research.proposed", "has:proposal"]
  remove: ["stage:research.requested", "status:in-progress"]

on_failure:
  add: ["status:needs-info", "status:blocked"]
  remove: ["status:in-progress"]
```

## Comment Templates

### Agent Start Comment

```markdown
[INTAKE-{timestamp}] Processing Research Request

**Status**: 🔄 Analyzing request **Research ID**: Pending assignment

I'm reviewing your research request and will:

1. Validate the problem statement
2. Assess feasibility and scope
3. Generate a formal research proposal
4. Assign appropriate category and priority

This typically takes 5-10 minutes.
```

### Success Comment

```markdown
[INTAKE-{timestamp}] Research Proposal Ready

**Status**: ✅ Proposal created **Research ID**: `{research_id}` **Category**:
{category} **Priority**: {priority}

### Proposal Summary

{proposal_summary}

### Next Steps

Your research proposal has been created and is ready for planning. The Research
Planning Agent will be automatically notified.

View full proposal: [proposals/{research_id}.yaml]
```

### Failure Comment

```markdown
[INTAKE-{timestamp}] Additional Information Needed

**Status**: ⚠️ Cannot proceed

### Issues Found

{validation_errors}

### Required Information

Please update your issue with: {missing_requirements}

Once updated, I'll automatically retry processing.
```

## Monitoring Metrics

- Average processing time: <10 minutes
- Success rate target: >95%
- Auto-retry on transient failures: 3 attempts
- Escalation after: 30 minutes
