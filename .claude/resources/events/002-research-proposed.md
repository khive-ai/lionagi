# Event 002: Research Proposed

## Event Definition

```yaml
event:
  id: "002"
  name: "research.proposed"
  description: "Research proposal created and awaiting planning"
  triggers:
    - "Label changed to 'stage:research.proposed'"
    - "Proposal file created in proposals/"
    - "Previous event 001 completed successfully"
  stage: "planning"
  parallelizable: true
```

## Trigger Conditions

- Has label: `stage:research.proposed`
- Has valid `research_id` in issue
- Proposal YAML exists at `proposals/{research_id}.yaml`
- No research plan exists yet

## Required Context

```yaml
context_required:
  - research_id: "From event 001"
  - proposal_file: "proposals/{research_id}.yaml"
  - issue_number: "GitHub issue reference"
  - category: "Research category"
  - priority: "Assigned priority"
```

## Agent Assignment

- **Primary**: Research Planning Agent
- **Support**: Context Discovery Agent (parallel)
- **Optional**: Domain Expert Agents (based on category)

## Expected Outputs

1. Detailed research plan (plans/{research_id}_plan.yaml)
2. Resource requirements identified
3. Timeline with milestones
4. Risk assessment
5. Data collection strategy

## State Transitions

- **Success**: → Event 003 (research.active)
- **Needs Approval**: → Hold in current state for human review
- **Blocked**: → If resources unavailable

## Validation Rules

- Proposal must be approved (if high priority)
- Resources must be available
- Timeline must be realistic
- No conflicting research in progress

## GitHub Label Updates

```yaml
on_start:
  add: ["status:planning", "agent:planner"]
  remove: ["status:ready"]

on_success:
  add: ["stage:research.active", "has:plan"]
  remove: ["stage:research.proposed", "status:planning"]

on_blocked:
  add: ["status:blocked", "needs:resources"]
  remove: ["status:planning"]
```

## Comment Templates

### Planning Start

```markdown
[PLANNER-{timestamp}] Creating Research Plan

**Status**: 🔄 Analyzing proposal **Research ID**: `{research_id}` **Estimated
Duration**: 10-15 minutes

### Planning Activities

- [ ] Analyzing research scope
- [ ] Identifying resource requirements
- [ ] Creating phased approach
- [ ] Assessing risks and dependencies
- [ ] Defining success metrics

@{requester} I'll notify you when the plan is ready.
```

### Plan Ready

```markdown
[PLANNER-{timestamp}] Research Plan Completed

**Status**: ✅ Plan ready **Total Duration**: {estimated_weeks} weeks
**Phases**: {phase_count}

### Plan Summary

{plan_summary}

### Resource Requirements

{resource_list}

### Key Milestones

{milestone_table}

### Next Steps

- Review the detailed plan: [plans/{research_id}_plan.yaml]
- Resources will be allocated
- Research execution begins automatically

{approval_required_note}
```

### Approval Required

```markdown
[PLANNER-{timestamp}] Plan Requires Approval

**Status**: 🔔 Awaiting approval **Reason**: {approval_reason}

### Plan Overview

{plan_overview}

### Decision Required By

**Approvers**: @{approver_list} **Deadline**: {deadline}

Please review and comment with:

- `APPROVED` to proceed
- `REVISE` with specific changes needed
- `REJECT` with rationale
```

## Parallel Execution

When multiple research proposals exist, planning can occur in parallel:

- Max parallel planning: 5 agents
- Resource conflict detection across plans
- Automatic serialization if conflicts detected

## Monitoring Metrics

- Planning time: <20 minutes average
- Plan approval rate: >90%
- Resource conflict rate: <10%
- Milestone accuracy: ±15%
