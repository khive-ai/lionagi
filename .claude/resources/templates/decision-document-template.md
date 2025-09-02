# Decision Document Template

## Purpose

Structured format for presenting research findings and recommendations for
decision-making.

## Template

```yaml
# Decision Document
decision_document:
  # Identification
  decision_id: "[DEC]_[research_id]_[version]"
  research_id: "[CATEGORY]_[###]"
  title: "[Decision Title - Action-Oriented]"
  decision_type: "[technical|process|tool|architecture|policy]"
  created_by: "[DECISION-SYNTHESIS-timestamp]"

  # Executive Summary
  summary:
    situation: "[Current state and problem]"
    recommendation: "[Proposed solution in 1-2 sentences]"
    impact: "[Expected outcome]"
    urgency: "[immediate|short-term|long-term]"
    confidence: "[high|medium|low]"

  # Context
  background:
    problem_statement: "[Detailed problem description]"
    business_context: "[Why this matters to the organization]"
    constraints:
      - "[Budget constraint]"
      - "[Time constraint]"
      - "[Technical constraint]"
    assumptions:
      - "[Key assumption 1]"
      - "[Key assumption 2]"

  # Research Findings
  findings:
    - finding: "[Key discovery 1]"
      evidence: "[Supporting data/source]"
      confidence: "[high|medium|low]"
      implications: "[What this means]"

    - finding: "[Key discovery 2]"
      evidence: "[Supporting data/source]"
      confidence: "[high|medium|low]"
      implications: "[What this means]"

  # Options Analysis
  options:
    - option_id: "A"
      name: "[Option name]"
      description: "[Detailed description]"
      pros:
        - "[Advantage 1]"
        - "[Advantage 2]"
      cons:
        - "[Disadvantage 1]"
        - "[Disadvantage 2]"
      cost_estimate: "[$ or effort estimate]"
      time_estimate: "[Implementation duration]"
      risk_level: "[high|medium|low]"

    - option_id: "B"
      name: "[Option name]"
      description: "[Detailed description]"
      pros:
        - "[Advantage 1]"
      cons:
        - "[Disadvantage 1]"
      cost_estimate: "[$ or effort estimate]"
      time_estimate: "[Implementation duration]"
      risk_level: "[high|medium|low]"

  # Recommendation
  recommendation:
    selected_option: "[Option ID]"
    rationale: "[Why this option is best]"

    decision_criteria:
      - criterion: "[Cost effectiveness]"
        weight: "[high|medium|low]"
        winner: "[Option ID]"
      - criterion: "[Implementation speed]"
        weight: "[high|medium|low]"
        winner: "[Option ID]"
      - criterion: "[Long-term maintainability]"
        weight: "[high|medium|low]"
        winner: "[Option ID]"

    implementation_approach: "[High-level implementation strategy]"

  # Implementation Plan
  implementation:
    phases:
      - phase: "[Phase 1 name]"
        duration: "[X weeks]"
        objectives: "[What will be accomplished]"
        deliverables:
          - "[Deliverable 1]"
          - "[Deliverable 2]"

    prerequisites:
      - "[Required approval]"
      - "[Resource allocation]"
      - "[Technical setup]"

    success_criteria:
      - "[Measurable outcome 1]"
      - "[Measurable outcome 2]"

    rollback_plan: "[How to revert if needed]"

  # Risk Analysis
  risks:
    - risk: "[Risk description]"
      probability: "[high|medium|low]"
      impact: "[high|medium|low]"
      mitigation: "[How to prevent/reduce]"
      owner: "[Who manages this risk]"

  # Stakeholder Impact
  stakeholder_analysis:
    - stakeholder: "[Group/role]"
      impact: "[How they're affected]"
      engagement: "[How to involve them]"
      concerns: "[Potential objections]"

  # Decision Request
  decision_request:
    requested_action: "[approve|reject|defer]"
    decision_by: "[YYYY-MM-DD]"
    approvers:
      - role: "[Approver role]"
        name: "[Optional: specific person]"
    escalation_path: "[If not approved by date]"

  # Supporting Data
  supporting_data:
    research_artifacts:
      - "[Link to analysis]"
      - "[Link to benchmarks]"
    references:
      - "[External source 1]"
      - "[Industry benchmark]"
    appendices:
      - title: "[Appendix A: Detailed Cost Analysis]"
        location: "[Link or embed]"
```

## Field Descriptions

### Summary Fields

- **situation**: Brief problem statement
- **recommendation**: Clear action to take
- **impact**: Business value delivered
- **confidence**: How sure we are

### Options Analysis

- **pros/cons**: Balanced evaluation
- **cost_estimate**: TCO including hidden costs
- **risk_level**: Overall risk assessment

### Decision Criteria

- **criterion**: What matters for the decision
- **weight**: Relative importance
- **winner**: Which option best satisfies

### Implementation

- **phases**: Logical groupings of work
- **prerequisites**: What must happen first
- **rollback_plan**: How to undo if needed

## Examples

### Example 1: Database Selection Decision

```yaml
decision_document:
  decision_id: "DEC_MEM_004_v1"
  title: "Select PostgreSQL + pgvector for Vector Database"

  summary:
    situation: "Need vector database for 10M embeddings with <50ms latency"
    recommendation: "Deploy PostgreSQL with pgvector extension"
    impact: "Meet performance needs at 70% lower cost than alternatives"
    confidence: "high"

  options:
    - option_id: "A"
      name: "PostgreSQL + pgvector"
      pros:
        - "30ms p95 latency in tests"
        - "$300/month total cost"
        - "Team has PostgreSQL expertise"
      cons:
        - "Manual index tuning required"
        - "Limited to 16K dimensions"

    - option_id: "B"
      name: "Dedicated Vector DB (Pinecone)"
      pros:
        - "25ms p95 latency"
        - "Fully managed"
      cons:
        - "$1000/month cost"
        - "Vendor lock-in"
        - "New technology for team"
```

### Example 2: Architecture Decision

```yaml
decision_document:
  decision_id: "DEC_ARC_007_v1"
  title: "Adopt Strangler Fig Pattern for Microservices Migration"

  summary:
    situation: "Monolith becoming unmaintainable, blocking feature velocity"
    recommendation: "Incrementally migrate using Strangler Fig pattern"
    impact: "Reduce deployment risk while improving system modularity"
    confidence: "high"

  implementation:
    phases:
      - phase: "Edge Services"
        duration: "6 weeks"
        objectives: "Extract authentication and API gateway"
      - phase: "Core Domain Split"
        duration: "12 weeks"
        objectives: "Separate order and inventory management"
```

## Decision Types

### Technical Decisions

- Tool/technology selection
- Architecture patterns
- Performance optimizations
- Security implementations

### Process Decisions

- Workflow changes
- Team structures
- Development practices
- Operational procedures

### Policy Decisions

- Standards adoption
- Compliance approaches
- Data governance
- Access controls

## Validation Rules

1. **Option Balance**: At least 2 viable options analyzed
2. **Evidence-Based**: All findings backed by data
3. **Risk Coverage**: All high risks have mitigation
4. **Stakeholder Completeness**: All affected parties identified
5. **Implementation Reality**: Plan is achievable
6. **Success Measurability**: Clear metrics defined

## Review Checklist

- [ ] Problem clearly defined
- [ ] Options fairly evaluated
- [ ] Recommendation justified
- [ ] Risks identified and mitigated
- [ ] Implementation plan realistic
- [ ] Stakeholders considered
- [ ] Success criteria measurable
- [ ] Decision reversible if needed

## Usage Notes

- Created by Decision Synthesis Agent
- Reviewed by Peer Review Agent
- Approved by designated decision makers
- Stored in `decisions/` directory
- Versioned for tracking changes
- Becomes input for implementation tracking
