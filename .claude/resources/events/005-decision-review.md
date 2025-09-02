# Event 005: Decision Review

## Event Definition

```yaml
event:
  id: "005"
  name: "decision.review"
  description: "Decision document under stakeholder review"
  triggers:
    - "Decision document created"
    - "Label changed to 'stage:decision.review'"
    - "Review requested from stakeholders"
  stage: "review"
  parallelizable: false # Reviews must be coordinated
```

## Trigger Conditions

- Has label: `stage:decision.review`
- Decision document exists at `decisions/{research_id}_decision.yaml`
- Stakeholders identified and notified
- Review deadline set

## Review Context

```yaml
review_context:
  decision_id: "Reference to decision"
  reviewers: "List of required reviewers"
  review_criteria: "What to evaluate"
  deadline: "Review completion date"
  approval_threshold: "Required approvals"
  revision_count: "Current revision number"
```

## Review Process

```yaml
review_stages:
  initial_review:
    participants: "All stakeholders"
    duration: "24-48 hours"
    async: true

  consolidation:
    agent: peer_review_agent
    task: "Consolidate feedback"
    duration: "2-4 hours"

  revision:
    agent: decision_synthesis_agent
    task: "Address feedback"
    condition: "If changes requested"

  final_review:
    participants: "Key approvers"
    duration: "4-8 hours"
    requirement: "Explicit approval"
```

## Agent Assignments

- **Coordinator**: Peer Review Agent
- **Revisions**: Decision Synthesis Agent
- **Validation**: Critic Agent
- **Communication**: Implementation Tracker Agent

## Expected Outcomes

1. Approved decision with sign-offs
2. Revision requests with specific changes
3. Rejected decision with rationale
4. Deferred decision with conditions

## State Transitions

- **Approved**: → Event 006 (implementation.approved)
- **Needs Revision**: → Back to Event 004 (revision loop)
- **Rejected**: → Event 009 (decision.rejected)
- **Deferred**: → Hold with conditions

## GitHub Label Updates

```yaml
on_start:
  add: ["status:under-review", "awaiting:feedback"]
  remove: ["status:ready"]

on_revision_needed:
  add: ["status:needs-revision", "revision:{count}"]
  remove: ["status:under-review"]

on_approved:
  add: ["stage:implementation.approved", "status:approved"]
  remove: ["stage:decision.review", "status:under-review"]
```

## Review Request Comment

```markdown
[PEER-REVIEW-{timestamp}] Decision Review Requested

**Status**: 📋 Review period started **Decision**: {decision_title} **Review
Deadline**: {deadline}

### Reviewers Requested

{reviewer_table_with_status}

### Review Checklist

Please evaluate:

- [ ] Solution addresses the original problem
- [ ] Constraints are satisfied
- [ ] Risks are acceptable
- [ ] Implementation is feasible
- [ ] Resources are available

### How to Review

1. Read the full decision document: [decisions/{research_id}_decision.yaml]
2. Focus on your area of expertise
3. Comment with feedback using the template below

### Review Response Template
```

DECISION: [APPROVE | REQUEST_CHANGES | REJECT]

FEEDBACK:

- [Your specific feedback here]

CONCERNS:

- [Any risks or issues]

CONDITIONS: (if any)

- [Specific conditions for approval]

```
cc: @{all_reviewers}
```

## Feedback Collection

```markdown
[PEER-REVIEW-{timestamp}] Review Status Update

**Reviews Completed**: {completed}/{total} **Current Status**: {overall_status}

### Review Summary

| Reviewer               | Decision | Key Feedback |
| ---------------------- | -------- | ------------ |
| {review_summary_table} |          |              |

### Consensus Analysis

- **Approvals**: {approval_count}
- **Changes Requested**: {change_count}
- **Rejections**: {rejection_count}

{consensus_statement}
```

## Revision Handling

```markdown
[DECISION-SYNTHESIS-{timestamp}] Addressing Review Feedback

**Status**: 📝 Revising decision based on feedback **Revision**:
v{revision_number}

### Changes Requested

{consolidated_change_requests}

### Revisions Made

1. ✅ {revision_1}
2. ✅ {revision_2}
3. ⏳ {in_progress_revision}

### Updated Sections

- {section_1}: {change_summary}
- {section_2}: {change_summary}

Estimated completion: {completion_time}
```

## Approval Notification

```markdown
[PEER-REVIEW-{timestamp}] Decision Approved ✅

**Status**: ✅ APPROVED **Approval Count**: {approver_count} **Conditions**:
{any_conditions}

### Approvers

{approver_list_with_timestamps}

### Implementation Authorization

This decision is now approved for implementation.

### Next Steps

1. Implementation team will be notified
2. Resources will be allocated
3. Implementation tracking begins
4. Success metrics will be monitored

### Conditions of Approval

{conditions_if_any}

Transitioning to: **Implementation Planning**
```

## Rejection Handling

```markdown
[PEER-REVIEW-{timestamp}] Decision Not Approved

**Status**: ❌ REJECTED **Reason**: {primary_rejection_reason}

### Rejection Details

{detailed_rejection_feedback}

### Options Forward

1. **Major Revision**: Address fundamental concerns
2. **Alternative Approach**: Consider different solution
3. **Scope Change**: Modify problem statement
4. **Abandon**: Close without implementation

### Stakeholder Guidance Needed

@{decision_makers} Please provide direction on next steps.
```

## Review Analytics

```yaml
review_metrics:
  response_time:
    target: "24 hours"
    track: "Per reviewer"

  revision_cycles:
    average: 1.5
    maximum: 3

  approval_correlation:
    track: "Approval vs implementation success"
    target: ">90% correlation"

  feedback_quality:
    specificity: "Actionable feedback %"
    coverage: "Areas reviewed"
```

## Escalation Process

```yaml
escalation_triggers:
  - condition: "No response after 48 hours"
    action: "Notify reviewer manager"

  - condition: "Deadlock in reviews"
    action: "Escalate to senior stakeholder"

  - condition: "3+ revision cycles"
    action: "Architecture review board"

  - condition: "Budget impact >$100k"
    action: "Executive approval required"
```

## Special Review Types

### Fast-Track Review

For low-risk, low-impact decisions:

- Single approver required
- 4-hour review window
- Simplified checklist

### Architecture Board Review

For high-impact technical decisions:

- Full board review required
- Presentation prepared
- Deep technical evaluation
- May take 1-2 weeks

### Emergency Review

For critical time-sensitive decisions:

- Expedited 2-hour window
- Verbal approvals accepted
- Written follow-up required

## Monitoring Metrics

- Review completion rate: >95% within deadline
- First-pass approval rate: >70%
- Average revision cycles: <2
- Reviewer response time: <24 hours
- Decision quality score: Post-implementation validation
