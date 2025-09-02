# Event 004: Decision Ready

## Event Definition

```yaml
event:
  id: "004"
  name: "decision.ready"
  description: "Research complete, decision synthesis required"
  triggers:
    - "Research phase completed successfully"
    - "Label changed to 'stage:decision.ready'"
    - "Research findings validated"
  stage: "decision"
  parallelizable: false # Decisions require consolidated view
```

## Trigger Conditions

- Has label: `stage:decision.ready`
- Research findings exist at `findings/{research_id}/`
- All research phases completed
- No critical validation failures

## Decision Context

```yaml
decision_context:
  research_id: "Source research"
  findings_summary: "Key discoveries"
  constraints: "From original request"
  stakeholders: "Decision makers"
  deadline: "Decision required by"
  confidence_threshold: "Minimum confidence required"
```

## Agent Assignment

- **Primary**: Decision Synthesis Agent
- **Support**: ROI Analysis Agent (preliminary)
- **Validation**: Peer Review Agent
- **Quality**: Critic Agent

## Expected Outputs

1. Decision document (decisions/{research_id}_decision.yaml)
2. Options analysis matrix
3. Risk assessment
4. Implementation roadmap
5. Stakeholder impact analysis

## State Transitions

- **Success**: → Event 005 (decision.review)
- **Insufficient Data**: → Back to Event 003 (more research)
- **Multiple Viable Options**: → Human decision required
- **No Viable Options**: → Event 008 (research.no-solution)

## Sequential Process

```yaml
decision_process:
  step_1:
    agent: decision_synthesis_agent
    task: "Synthesize findings into options"
    duration: "30-45 minutes"

  step_2:
    agent: roi_analysis_agent
    task: "Preliminary cost-benefit analysis"
    duration: "20-30 minutes"

  step_3:
    agent: critic_agent
    task: "Validate logic and completeness"
    duration: "15-20 minutes"

  step_4:
    agent: decision_synthesis_agent
    task: "Finalize recommendation"
    duration: "10-15 minutes"
```

## GitHub Label Updates

```yaml
on_start:
  add: ["status:synthesizing", "agent:decision"]
  remove: ["status:ready"]

on_success:
  add: ["stage:decision.review", "has:recommendation"]
  remove: ["stage:decision.ready", "status:synthesizing"]

on_needs_human:
  add: ["status:needs-decision", "human-required"]
```

## Comment Templates

### Synthesis Start

```markdown
[DECISION-SYNTHESIS-{timestamp}] Beginning Decision Analysis

**Status**: 🤔 Analyzing research findings **Research ID**: `{research_id}`
**Findings Reviewed**: {finding_count} documents

### Decision Process

1. ⏳ Synthesizing research findings
2. ⏳ Generating solution options
3. ⏳ Analyzing trade-offs
4. ⏳ Formulating recommendation

Expected completion: 45-60 minutes
```

### Options Generated

```markdown
[DECISION-SYNTHESIS-{timestamp}] Options Analysis Complete

**Status**: 📊 {option_count} viable options identified

### Option Summary

{options_table}

### Evaluation Criteria

- ✅ Meets requirements: {requirements_matrix}
- 💰 Cost analysis: {cost_comparison}
- ⏱️ Timeline impact: {timeline_comparison}
- 🎯 Success probability: {success_scores}

Proceeding to recommendation synthesis...
```

### Preliminary Decision

```markdown
[DECISION-SYNTHESIS-{timestamp}] Preliminary Recommendation

**Status**: 🎯 Recommendation ready for review **Recommended Option**:
{selected_option} **Confidence Level**: {confidence}%

### Rationale

{decision_rationale}

### Key Trade-offs

{tradeoff_analysis}

### Implementation Overview

{implementation_summary}

**Next**: Peer review and validation
```

### ROI Preview

```markdown
[ROI-ANALYSIS-{timestamp}] Preliminary ROI Assessment

**Option**: {option_name} **ROI Estimate**: {roi_percentage}% over {time_period}

### Cost Breakdown

{cost_table}

### Benefit Analysis

{benefit_list}

### Break-even Point

{breakeven_analysis}

_Note: Full ROI analysis available after implementation_
```

### Human Decision Required

```markdown
[DECISION-SYNTHESIS-{timestamp}] Human Decision Required

**Status**: 🔔 Multiple viable options - need human input **Reason**:
{reason_for_escalation}

### Options Requiring Decision

{detailed_options_comparison}

### Decision Framework

I've prepared a decision matrix to help: {decision_matrix}

### Required Input

Please comment with your decision:

- `SELECT OPTION {A|B|C}` - Choose an option
- `NEED MORE INFO` - Request specific analysis
- `DEFER` - Postpone decision with reason

**Decision needed by**: {deadline} cc: @{decision_makers}
```

## Decision Quality Criteria

```yaml
quality_requirements:
  completeness:
    - All findings incorporated
    - All constraints considered
    - All stakeholders identified

  logic:
    - Clear cause-effect reasoning
    - No circular dependencies
    - Evidence-based conclusions

  practicality:
    - Implementation feasible
    - Resources available
    - Timeline realistic

  risk_assessment:
    - Risks identified
    - Mitigations proposed
    - Contingencies planned
```

## Special Cases

### Insufficient Confidence

```markdown
[CRITIC-{timestamp}] Low Confidence Alert

**Issue**: Decision confidence below threshold **Current**:
{current_confidence}% **Required**: {required_confidence}%

### Gaps Identified

{confidence_gaps}

### Recommended Actions

1. {specific_research_needed}
2. {additional_validation}

Recommend returning to research phase for targeted investigation.
```

### No Viable Options

```markdown
[DECISION-SYNTHESIS-{timestamp}] No Viable Solution Found

**Status**: ❌ Cannot recommend any option

### Blocking Issues

{blocker_list}

### Potential Paths Forward

1. Relax constraints: {constraint_options}
2. Increase budget/timeline
3. Consider alternative approaches
4. Pivot problem statement

Escalating to stakeholders for guidance.
```

## Monitoring Metrics

- Decision synthesis time: <90 minutes
- Recommendation confidence: >80% target
- Stakeholder alignment: Track feedback
- Decision revision rate: <20%
- Implementation success correlation: >85%
