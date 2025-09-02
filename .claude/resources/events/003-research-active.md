# Event 003: Research Active

## Event Definition

```yaml
event:
  id: "003"
  name: "research.active"
  description: "Research plan approved and execution in progress"
  triggers:
    - "Label changed to 'stage:research.active'"
    - "Research plan exists and approved"
    - "Resources allocated"
  stage: "execution"
  parallelizable: false # Sequential phases within research
```

## Trigger Conditions

- Has label: `stage:research.active`
- Valid research plan at `plans/{research_id}_plan.yaml`
- Resources confirmed available
- No blocking dependencies

## Execution Context

```yaml
execution_context:
  plan_phases: "From research plan"
  active_phase: "Current execution phase"
  phase_outputs: "Accumulated results"
  time_elapsed: "Since execution start"
  resource_usage: "Tracking consumption"
```

## Agent Swarm Assignment

```yaml
phase_1_discovery: # Parallel execution
  agents:
    - codebase_analyst_agent
    - context_discovery_agent
    - experiment_runner_agent
  pattern: "parallel"
  max_duration: "2h"

phase_2_analysis: # Sequential processing
  agents:
    - decision_synthesis_agent
    - critic_agent
  pattern: "sequential"

phase_3_validation:
  agents:
    - peer_review_agent
    - critic_agent
  pattern: "iterative"
```

## Expected Outputs

1. Phase completion reports
2. Research findings (findings/{research_id}/)
3. Data artifacts (data/{research_id}/)
4. Experiment results
5. Preliminary recommendations

## State Transitions

- **Phase Complete**: → Next phase or Event 004
- **Blocked**: → Pause with blocking issue
- **Failed**: → Event 006 (research.failed)
- **Success**: → Event 004 (decision.ready)

## Long-Running Execution

For research lasting >1 hour:

- Create dedicated PR for communication
- Post hourly progress updates
- Checkpoint intermediate results
- Support pause/resume

## GitHub Label Updates

```yaml
on_start:
  add: ["status:executing", "phase:discovery"]
  remove: ["status:ready"]

on_phase_complete:
  add: ["phase:{next_phase}"]
  remove: ["phase:{current_phase}"]

on_complete:
  add: ["stage:decision.ready", "has:findings"]
  remove: ["stage:research.active", "status:executing"]
```

## Progress Tracking Comments

### Execution Start

```markdown
[ORCHESTRATOR-{timestamp}] Research Execution Started

**Status**: 🚀 Phase 1 - Discovery **Research ID**: `{research_id}` **Estimated
Completion**: {estimated_time}

### Execution Plan

{phase_summary_table}

### Active Agents

- 🔍 Codebase Analyst: Examining repository structure
- 🌐 Context Discovery: Gathering related information
- 🧪 Experiment Runner: Setting up test scenarios

Progress updates every 30 minutes.
```

### Phase Progress Update

```markdown
[ORCHESTRATOR-{timestamp}] Phase Progress Update

**Current Phase**: {phase_name} ({phase_number}/{total_phases}) **Progress**:
{progress_bar} {percentage}% **Time Elapsed**: {elapsed_time}

### Completed Activities

✅ {completed_activity_list}

### In Progress

🔄 {current_activity}

### Preliminary Findings

{finding_preview}

**Next Update**: {next_update_time}
```

### Checkpoint Report

```markdown
[{AGENT_ID}-{timestamp}] Research Checkpoint

**Phase**: {phase_name} **Status**: 📊 Intermediate results available

### Key Discoveries

{discovery_list}

### Data Collected

- Samples analyzed: {sample_count}
- Patterns identified: {pattern_count}
- Anomalies found: {anomaly_count}

### Confidence Levels

{confidence_matrix}

Full details: [findings/{research_id}/checkpoint_{number}.yaml]
```

### Phase Completion

```markdown
[CRITIC-{timestamp}] Phase Validation Complete

**Phase**: {phase_name} **Status**: ✅ Validated and approved

### Quality Metrics

- Completeness: {completeness}%
- Accuracy confidence: {confidence}
- Coverage: {coverage_areas}

### Issues Found

{minor_issues}

### Recommendations for Next Phase

{recommendations}

Proceeding to: **{next_phase_name}**
```

## Resource Management

```yaml
resource_tracking:
  compute:
    allocated: "4 agents parallel"
    consumed: "Track per agent"
    remaining: "Monitor availability"

  api_calls:
    github: "Rate limit aware"
    external: "Quota tracking"

  time:
    phase_timeout: "Per plan"
    total_timeout: "8 hours max"
    checkpoint_interval: "30 minutes"
```

## Error Handling

```yaml
error_strategies:
  agent_failure:
    action: "Retry with backoff"
    max_retries: 3
    fallback: "Reduce scope"

  resource_exhaustion:
    action: "Pause and wait"
    notification: "Alert stakeholders"

  timeout:
    action: "Save partial results"
    decision: "Continue or abort"
```

## Monitoring Metrics

- Phase completion rate: >95%
- Average phase duration: Within ±20% of estimate
- Agent failure rate: <5%
- Resource utilization: 70-90% optimal
- Finding quality score: >85%
