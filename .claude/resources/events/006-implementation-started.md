# Event 006: Implementation Started

## Event Definition

```yaml
event:
  id: "006"
  name: "implementation.started"
  description: "Approved decision moves to active implementation"
  triggers:
    - "Decision approved in review"
    - "Label changed to 'stage:implementation.started'"
    - "Implementation team assigned"
  stage: "implementation"
  parallelizable: false # Sequential phases within implementation
```

## Trigger Conditions

- Has label: `stage:implementation.started`
- Decision document approved
- Implementation plan exists
- Resources allocated
- Team assigned

## Implementation Context

```yaml
implementation_context:
  decision_id: "Approved decision reference"
  implementation_plan: "From decision document"
  resource_allocation: "Confirmed resources"
  team_assignments: "Who does what"
  success_metrics: "What we're measuring"
  timeline: "Key milestones and dates"
```

## Agent Assignment

- **Primary**: Implementation Tracker Agent
- **Monitoring**: ROI Analysis Agent (continuous)
- **Validation**: Critic Agent (phase gates)
- **Communication**: Automated updates

## Phased Execution

```yaml
implementation_phases:
  preparation:
    duration: "1-3 days"
    activities:
      - "Environment setup"
      - "Access provisioning"
      - "Baseline metrics capture"
    owner: "DevOps team"

  phase_1:
    name: "Core implementation"
    duration: "From plan"
    activities: "Phase 1 activities"
    success_gate: "Acceptance criteria"

  phase_2:
    name: "Integration"
    duration: "From plan"
    activities: "Phase 2 activities"
    success_gate: "Integration tests pass"

  stabilization:
    duration: "3-5 days"
    activities:
      - "Monitor metrics"
      - "Address issues"
      - "Document changes"
```

## State Transitions

- **Phase Complete**: → Next phase
- **Milestone Reached**: → Update progress
- **Blocked**: → Hold with blocker
- **Failed**: → Rollback initiated
- **Success**: → Event 007 (metrics.collection)

## GitHub Label Updates

```yaml
on_start:
  add: ["status:implementing", "phase:prep"]
  remove: ["status:approved"]

on_phase_change:
  add: ["phase:{new_phase}"]
  remove: ["phase:{old_phase}"]

on_complete:
  add: ["stage:metrics.collection", "status:complete"]
  remove: ["stage:implementation.started", "status:implementing"]
```

## Implementation Kickoff Comment

````markdown
[IMPLEMENTATION-TRACKER-{timestamp}] Implementation Started 🚀

**Decision**: {decision_title} **Implementation ID**: `{impl_id}` **Target
Completion**: {target_date}

### Implementation Plan Overview

{phase_table}

### Success Metrics

<!-- Machine-readable format for automation -->

```yaml
metrics:
  - name: "P95 latency"
    baseline: 45ms
    target: "<50ms"
    measurement: "prometheus"
  - name: "Error rate"
    baseline: 0.5%
    target: "<0.5%"
    measurement: "grafana"
```
````

### Team Assignments

| Phase                   | Owner | Team Members |
| ----------------------- | ----- | ------------ |
| {team_assignment_table} |       |              |

### Tracking

- **Daily Updates**: High-priority items
- **Weekly Updates**: Standard items
- **Metrics Dashboard**: [Link to dashboard]

First update: {next_update_time}

````
## Progress Update Format
```markdown
[IMPLEMENTATION-TRACKER-{timestamp}] Implementation Progress Update

**Phase**: {current_phase} ({phase_n}/{total_phases})
**Progress**: ▓▓▓▓▓▓▓░░░ 70%
**Status**: 🟢 On Track | 🟡 At Risk | 🔴 Blocked

### Completed This Period
✅ {completed_item_1}
✅ {completed_item_2}

### In Progress
🔄 {active_item_1} (85% complete)
🔄 {active_item_2} (40% complete)

### Current Metrics
```yaml
metrics_update:
  timestamp: "{ISO_8601}"
  values:
    - metric: "P95 latency"
      current: 42ms
      trend: "improving"
      vs_target: "✅ Meeting"
    - metric: "Error rate"  
      current: 0.3%
      trend: "stable"
      vs_target: "✅ Meeting"
````

### Risks & Issues

{risk_status_table}

**Next Update**: {next_update_schedule}

````
## Phase Completion Notification
```markdown
[IMPLEMENTATION-TRACKER-{timestamp}] Phase Completed ✅

**Phase**: {phase_name}
**Duration**: {actual_duration} (planned: {planned_duration})
**Status**: ✅ Successfully Completed

### Phase Outcomes
{outcomes_achieved}

### Metrics at Phase End
{metrics_snapshot}

### Lessons Learned
- {lesson_1}
- {lesson_2}

### Next Phase
**Starting**: {next_phase_start}
**Owner**: @{phase_owner}
**Key Activities**: {activity_preview}
````

## Automated Monitoring

```yaml
monitoring_config:
  metrics_collection:
    frequency: "Every 15 minutes"
    storage: "Time series DB"

  threshold_alerts:
    - metric: "latency"
      warning: ">48ms"
      critical: ">55ms"
      action: "Notify team + emit event"

  automated_rollback:
    trigger: "Critical threshold for >30min"
    approval: "Auto for non-prod, manual for prod"
```

## Risk Management

```markdown
### Active Risk Monitoring

| Risk     | Probability | Impact | Status        | Mitigation     |
| -------- | ----------- | ------ | ------------- | -------------- |
| {risk_1} | Medium      | High   | 🟡 Watching   | {mitigation_1} |
| {risk_2} | Low         | Medium | 🟢 Controlled | {mitigation_2} |

**Escalation Triggers**:

- Any risk moves to High/High
- Multiple risks at Medium/High
- Mitigation fails
```

## Success Criteria Tracking

```yaml
success_tracking:
  phase_gates:
    - phase: "Core implementation"
      criteria:
        - "All unit tests pass"
        - "Performance within 10% of target"
        - "No P1 bugs"
      validation: "Automated + manual review"

  final_criteria:
    - "All phases complete"
    - "Metrics meeting targets for 48h"
    - "User acceptance signed off"
    - "Documentation complete"
```

## Rollback Procedures

```yaml
rollback_plan:
  triggers:
    - "Critical metric breach"
    - "Data corruption detected"
    - "Multiple P1 issues"

  procedure:
    1: "Pause implementation"
    2: "Assess impact"
    3: "Execute rollback if needed"
    4: "Document lessons learned"

  communication:
    immediate: "Incident channel"
    stakeholders: "Within 15 minutes"
    post_mortem: "Within 48 hours"
```

## Stakeholder Updates

Based on priority and stakeholder preferences:

### Executive Summary (Weekly)

```markdown
**Implementation**: {name} **Overall Status**: {status_emoji} {status_text}
**Completion**: {percentage}% **Key Metrics**: {all_green|some_yellow|some_red}
**Budget**: {on_track|over|under} **Timeline**: {on_schedule|ahead|behind}
```

### Technical Details (Daily for High Priority)

```markdown
**Today's Progress**: {detailed_items} **Metrics Trend**: {charts_link}
**Blockers**: {any_blockers} **Tomorrow's Plan**: {next_items}
```

## Monitoring Metrics

- Phase completion accuracy: ±10% of planned duration
- Metric achievement rate: >95%
- Risk mitigation effectiveness: >80%
- Stakeholder satisfaction: >4/5
- Rollback frequency: <5%
