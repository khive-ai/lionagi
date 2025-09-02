# Research Plan Template

## Purpose

Detailed execution plan for approved research proposals, outlining methodology,
resources, and deliverables.

## Template

```yaml
# Research Plan
research_plan:
  # Identification
  research_id: "[CATEGORY]_[###]"
  proposal_ref: "[Link to approved proposal]"
  plan_version: "1.0"
  created_by: "[PLANNER-timestamp]"

  # Executive Summary
  objective: "[One-line statement of what will be achieved]"
  key_questions:
    - "[Primary question to answer]"
    - "[Secondary question 1]"
    - "[Secondary question 2]"
  expected_outcome: "[What success looks like]"

  # Methodology
  approach:
    type: "[comparative|experimental|analytical|exploratory]"
    description: "[Detailed methodology description]"
    justification: "[Why this approach is best]"

  phases:
    - phase: "Discovery"
      duration: "[X days]"
      activities:
        - "[Activity 1]"
        - "[Activity 2]"
      deliverables:
        - "[Deliverable 1]"
      success_criteria: "[What marks completion]"

    - phase: "Analysis"
      duration: "[X days]"
      activities:
        - "[Activity 1]"
        - "[Activity 2]"
      deliverables:
        - "[Analysis report]"
      success_criteria: "[Completion criteria]"

    - phase: "Synthesis"
      duration: "[X days]"
      activities:
        - "[Activity 1]"
      deliverables:
        - "[Final recommendations]"
      success_criteria: "[Completion criteria]"

  # Resources
  resources:
    tools:
      - name: "[Tool name]"
        purpose: "[Why needed]"
        access: "[How to access]"

    data_sources:
      - source: "[Data source]"
        type: "[logs|metrics|docs|code]"
        access_method: "[API|query|file]"

    external_dependencies:
      - dependency: "[External system/team]"
        needed_for: "[Why required]"
        contact: "[Who to contact]"

    team_resources:
      - role: "[Expert type]"
        allocation: "[% time or hours]"
        responsibilities: "[What they'll do]"

  # Data Collection Plan
  data_requirements:
    - data_type: "[Type of data]"
      source: "[Where from]"
      collection_method: "[How to gather]"
      volume: "[Expected amount]"
      quality_criteria: "[Acceptance criteria]"

  # Analysis Framework
  analysis_methods:
    - method: "[Analysis technique]"
      tool: "[Tool/framework to use]"
      inputs: "[Required data]"
      outputs: "[Expected results]"
      validation: "[How to verify]"

  # Risk Management
  risks:
    - risk: "[Risk description]"
      probability: "[high|medium|low]"
      impact: "[high|medium|low]"
      mitigation: "[How to prevent/handle]"
      contingency: "[Backup plan]"

  # Timeline
  schedule:
    start_date: "[YYYY-MM-DD]"
    key_milestones:
      - date: "[YYYY-MM-DD]"
        milestone: "[What will be achieved]"
        checkpoint: "[Review/decision point]"
    end_date: "[YYYY-MM-DD]"
    buffer_time: "[X days contingency]"

  # Success Metrics
  success_metrics:
    quantitative:
      - metric: "[Measurable outcome]"
        target: "[Specific value]"
        measurement: "[How to measure]"
    qualitative:
      - metric: "[Quality outcome]"
        assessment: "[How to evaluate]"

  # Communication Plan
  communication:
    stakeholder_updates:
      - audience: "[Stakeholder group]"
        frequency: "[Daily|Weekly|On milestone]"
        format: "[Email|Meeting|Report]"
    progress_tracking:
      - method: "[GitHub issues|Dashboard|Reports]"
      - location: "[Where to find updates]"

  # Deliverables
  deliverables:
    - name: "[Deliverable name]"
      type: "[report|code|data|recommendation]"
      format: "[markdown|yaml|json|notebook]"
      due_date: "[YYYY-MM-DD]"
      acceptance_criteria: "[What makes it complete]"

  # Approval
  approval_required:
    - checkpoint: "[Phase completion]"
      approver: "[Role/person]"
      criteria: "[What they're approving]"
```

## Field Descriptions

### Methodology Fields

- **approach**: Overall research strategy
- **phases**: Sequential stages with clear boundaries
- **activities**: Specific tasks within each phase
- **deliverables**: Tangible outputs from each phase

### Resource Planning

- **tools**: Software, frameworks, APIs needed
- **data_sources**: Where information comes from
- **team_resources**: Human expertise required
- **external_dependencies**: Outside requirements

### Risk Management

- **probability**: Likelihood of occurrence
- **impact**: Severity if it happens
- **mitigation**: Preventive measures
- **contingency**: Backup approach

### Success Metrics

- **quantitative**: Measurable numbers
- **qualitative**: Experience/quality measures
- **targets**: Specific goals to hit

## Examples

### Example 1: Database Benchmarking Plan

```yaml
research_plan:
  research_id: "MEM_004"
  objective: "Benchmark 5 vector databases for 10M embedding scale"

  approach:
    type: "comparative"
    description: "Head-to-head performance testing with production workload"

  phases:
    - phase: "Environment Setup"
      duration: "3 days"
      activities:
        - "Deploy test instances of each database"
        - "Load 10M embeddings dataset"
        - "Configure monitoring"
      deliverables:
        - "Test environment documentation"
        - "Baseline performance metrics"

    - phase: "Performance Testing"
      duration: "5 days"
      activities:
        - "Run query latency tests"
        - "Test concurrent user loads"
        - "Measure resource utilization"
      deliverables:
        - "Performance test results"
        - "Resource usage reports"

  success_metrics:
    quantitative:
      - metric: "P95 query latency"
        target: "<50ms"
        measurement: "Prometheus metrics"
```

### Example 2: Architecture Migration Plan

```yaml
research_plan:
  research_id: "ARC_007"
  objective: "Design microservices migration strategy for monolith"

  approach:
    type: "analytical"
    description: "Domain-driven design analysis with incremental migration path"

  phases:
    - phase: "Domain Analysis"
      duration: "7 days"
      activities:
        - "Map existing functionality"
        - "Identify bounded contexts"
        - "Analyze data flows"
      deliverables:
        - "Domain model diagram"
        - "Service boundary recommendations"
```

## Validation Rules

1. **Phase Completeness**: Each phase must have activities and deliverables
2. **Resource Availability**: All resources must be confirmed available
3. **Timeline Realism**: Duration must align with complexity
4. **Metric Measurability**: All metrics must have measurement methods
5. **Dependency Management**: External dependencies need contacts
6. **Risk Coverage**: High-impact items need mitigation plans

## Usage Notes

- Generated by Research Planning Agent after proposal approval
- Reviewed and approved before execution begins
- Updated during execution with actual progress
- Forms basis for decision document structure
- Stored in `plans/` directory with version control
