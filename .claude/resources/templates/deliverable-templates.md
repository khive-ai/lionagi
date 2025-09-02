# KB Event Deliverable Templates

Each event MUST post its deliverable as a GitHub issue comment following these
templates.

## Event 001: Research Requested → Research Proposal

````markdown
## 📋 Research Proposal Deliverable

**Event**: 001 - Research Requested → Research Proposed\
**Agent**: [RESEARCH_INTAKE_AGENT-{timestamp}]\
**Research ID**: {research_id}

### Proposal Summary

```yaml
research_id: { research_id }
title: { title }
category: { AIO|MEM|TLI|ARC|DEV|UXP }
priority: { critical|high|medium|low }
complexity: { low|medium|high }
estimated_duration: { duration }
```
````

### Problem Analysis

{problem_statement}

### Proposed Approach

{approach}

### Success Criteria

- [ ] {criterion_1}
- [ ] {criterion_2}
- [ ] {criterion_3}

### Resources Required

{resources}

**Status**: ✅ Ready for planning\
**Next Stage**: stage:research.proposed

````
## Event 002: Research Proposed → Research Plan

```markdown
## 📅 Research Plan Deliverable

**Event**: 002 - Research Proposed → Research Active  
**Agent**: [RESEARCH_PLANNING_AGENT-{timestamp}]  
**Research ID**: {research_id}

### Research Plan

```yaml
research_id: {research_id}
phases:
  - phase_1:
      name: {name}
      duration: {duration}
      deliverables: [{list}]
  - phase_2:
      name: {name}
      duration: {duration}
      deliverables: [{list}]
````

### Methodology

{methodology}

### Risk Assessment

{risks}

### Milestones

- [ ] {milestone_1} - {date}
- [ ] {milestone_2} - {date}
- [ ] {milestone_3} - {date}

**Status**: ✅ Ready for execution\
**Next Stage**: stage:research.active

````
## Event 003: Research Active → Research Findings

```markdown
## 🔬 Research Findings Deliverable

**Event**: 003 - Research Active → Decision Ready  
**Agent**: [CODEBASE_ANALYST_AGENT-{timestamp}]  
**Research ID**: {research_id}

### Executive Summary
{summary}

### Key Findings

```yaml
findings:
  - finding_1:
      description: {description}
      confidence: {high|medium|low}
      evidence: {evidence}
  - finding_2:
      description: {description}
      confidence: {high|medium|low}
      evidence: {evidence}
````

### Recommendations

1. {recommendation_1}
2. {recommendation_2}
3. {recommendation_3}

### Supporting Data

- Analysis results: {link_or_summary}
- Test results: {link_or_summary}
- Benchmarks: {link_or_summary}

**Status**: ✅ Research complete\
**Next Stage**: stage:decision.ready

````
## Event 004: Decision Ready → Decision Document

```markdown
## 🤔 Decision Document Deliverable

**Event**: 004 - Decision Ready → Decision Review  
**Agent**: [DECISION_SYNTHESIS_AGENT-{timestamp}]  
**Research ID**: {research_id}

### Decision Summary

```yaml
research_id: {research_id}
recommendation: {recommended_action}
confidence_level: {high|medium|low}
decision_type: {technical|architectural|process|tool}
````

### Rationale

{detailed_rationale}

### Options Considered

1. **Option A**: {description}
   - Pros: {pros}
   - Cons: {cons}
2. **Option B**: {description}
   - Pros: {pros}
   - Cons: {cons}

### Implementation Impact

- Technical: {impact}
- Timeline: {impact}
- Resources: {impact}

### Risks & Mitigations

{risks_and_mitigations}

**Status**: ✅ Ready for review\
**Next Stage**: stage:decision.review

````
## Event 005: Decision Review → Approval Records

```markdown
## ✅ Approval Records Deliverable

**Event**: 005 - Decision Review → Implementation Approved  
**Agent**: [PEER_REVIEW_AGENT-{timestamp}]  
**Research ID**: {research_id}

### Review Summary

```yaml
research_id: {research_id}
decision_status: {approved|rejected|needs_revision}
reviewers:
  - name: {reviewer_1}
    verdict: {approve|reject|abstain}
    comments: {comments}
  - name: {reviewer_2}
    verdict: {approve|reject|abstain}
    comments: {comments}
````

### Conditions of Approval

- [ ] {condition_1}
- [ ] {condition_2}

### Implementation Guidelines

{guidelines}

**Status**: ✅ Approved for implementation\
**Next Stage**: stage:implementation.approved

````
## Event 006: Implementation Started → Implementation Progress

```markdown
## 🚀 Implementation Progress Deliverable

**Event**: 006 - Implementation Approved → Implementation Started  
**Agent**: [IMPLEMENTATION_TRACKER_AGENT-{timestamp}]  
**Research ID**: {research_id}

### Implementation Status

```yaml
research_id: {research_id}
implementation_phase: {planning|development|testing|deployment}
progress_percentage: {0-100}
blockers: [{list_of_blockers}]
````

### Completed Tasks

- [x] {completed_task_1}
- [x] {completed_task_2}
- [ ] {pending_task_1}
- [ ] {pending_task_2}

### Key Milestones

{milestones_and_dates}

### Resource Usage

- Time: {actual} vs {estimated}
- Budget: {actual} vs {allocated}
- Team: {resources_deployed}

**Status**: ✅ Implementation in progress\
**Next Stage**: stage:metrics.collection

````
## Event 007: Metrics Collection → Metrics Data

```markdown
## 📊 Metrics Collection Deliverable

**Event**: 007 - Implementation Started → Metrics Collection  
**Agent**: [METRICS_COLLECTION_AGENT-{timestamp}]  
**Research ID**: {research_id}

### Metrics Summary

```yaml
research_id: {research_id}
collection_period: {start_date} to {end_date}
metrics_collected:
  - performance:
      baseline: {value}
      current: {value}
      improvement: {percentage}
  - quality:
      baseline: {value}
      current: {value}
      improvement: {percentage}
````

### Key Performance Indicators

1. **{KPI_1}**: {value} ({trend})
2. **{KPI_2}**: {value} ({trend})
3. **{KPI_3}**: {value} ({trend})

### Data Sources

- {source_1}: {description}
- {source_2}: {description}

### Anomalies Detected

{anomalies_if_any}

**Status**: ✅ Metrics collected\
**Next Stage**: stage:metrics.review

````
## Event 008: Metrics Review → ROI Analysis

```markdown
## 💰 ROI Analysis Deliverable

**Event**: 008 - Metrics Collection → Metrics Review  
**Agent**: [ROI_ANALYSIS_AGENT-{timestamp}]  
**Research ID**: {research_id}

### ROI Summary

```yaml
research_id: {research_id}
roi_calculation:
  investment: {total_cost}
  returns: {measured_value}
  roi_percentage: {percentage}
  payback_period: {duration}
````

### Value Delivered

1. **Quantitative**:
   - {metric_1}: {improvement}
   - {metric_2}: {improvement}
2. **Qualitative**:
   - {benefit_1}
   - {benefit_2}

### Lessons Learned

{key_learnings}

### Recommendations

- [ ] {future_optimization_1}
- [ ] {future_optimization_2}

### Follow-up Research

{suggested_follow_up_research_ids}

**Status**: ✅ ROI analysis complete\
**Next Stage**: stage:knowledge.captured

````
## Event 009: Knowledge Captured → Knowledge Repository Update

```markdown
## 🧠 Knowledge Capture Deliverable

**Event**: 009 - Metrics Review → Knowledge Captured  
**Agent**: [KNOWLEDGE_CURATOR_AGENT-{timestamp}]  
**Research ID**: {research_id}

### Knowledge Summary

```yaml
research_id: {research_id}
knowledge_type: {pattern|solution|lesson|insight}
reusability_score: {high|medium|low}
tags: [{tag1}, {tag2}, {tag3}]
````

### Captured Knowledge

{detailed_knowledge_description}

### Integration Points

- Knowledge Graph Nodes: [{node_ids}]
- Related Research: [{research_ids}]
- Templates Created: [{template_names}]

### Future Applications

{how_this_knowledge_can_be_reused}

### Archive Location

- Primary: {location}
- Backups: {locations}

**Status**: ✅ Knowledge captured and indexed\
**Next Stage**: TERMINAL STATE - Cycle Complete! 🎉

```
## Validation Rules

1. **Mandatory Fields**: Every template field marked with {} must be filled
2. **YAML Blocks**: Must be valid YAML syntax
3. **Agent Signature**: Must include proper agent signature with timestamp
4. **Status Line**: Must include status and next stage
5. **Research ID**: Must match the pattern [A-Z]{3}_\d{3}
```
