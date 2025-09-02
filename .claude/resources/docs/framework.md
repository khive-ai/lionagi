# Multi‑Agent Knowledge Graph Lifecycle Framework

## Lifecycle Stages and Events Overview

The knowledge creation process is broken into discrete **lifecycle events** from
initial request through implementation and ROI analysis. Each event corresponds
to a stage in the research/decision lifecycle and triggers specific agent
actions. Below is the end-to-end sequence of stages:

1. **`research.requested` – Intake:** A raw research request is submitted (e.g.
   via a GitHub Issue). The **Research Intake Agent** scoping the problem
   transforms it into a structured proposal for approval.
2. **`research.proposed` – Planning:** An approved proposal is expanded by the
   **Research Planning Agent** into a detailed research plan (methodology,
   milestones). This plan guides the execution of the research.
3. **(Research Execution – In Progress):** _Not a single event, but the phase
   where the research plan is carried out._ The orchestrator may spawn multiple
   agents or tasks to gather data, run experiments, and compile findings. This
   culminates in the research being completed.
4. **`decision.ready` – Decision Drafting:** Once research is done, the
   **Decision Agent** synthesizes the findings into a decision document with
   recommendations and an implementation plan. This includes a machine-readable
   summary and updates the research status.
5. **`decision.review_requested` – Peer Review:** The **Peer Review Agent** (or
   a human reviewer) evaluates the decision document against a quality
   checklist. The review yields an output indicating **APPROVED**,
   **NEEDS\_REVISION**, or **REQUIRES\_REWORK** with feedback. This is a quality
   gate before implementation.
6. **(Decision Revision – Iteration):** _If revisions are needed_, the
   orchestrator routes the feedback to the Decision Agent (or human) to update
   the decision document. The revised document is re-submitted for review
   (another `decision.review_requested` event), looping until approved.
7. **`implementation.started` – Implementation Planning:** Once the decision is
   approved (decision made), the **Implementation Agent** generates an
   implementation tracking plan. This defines milestones, success metrics, and
   monitoring setup for the execution of the decision. The issue moves into an
   “Implementation” stage.
8. **(Implementation Execution – Ongoing):** The team (or automated processes)
   executes the implementation plan. Progress and metrics are tracked (the
   system posts `implementation.metrics_update` events with YAML payloads of
   metric values, progress, blockers, etc., as defined in the plan). This stage
   continues until the project is delivered.
9. **`metrics.review_due` – ROI Analysis:** After implementation (e.g. a few
   months post-release), an **ROI Analysis Agent** is triggered to assess
   outcomes vs. predictions. It compares actual metrics to original success
   criteria, calculates ROI, and captures lessons learned in an ROI report.
10. **Cycle Complete – Knowledge Captured:** The cycle concludes. The issue can
    be closed or marked “Completed.” All artifacts (proposal, plan, decision,
    etc.) are stored in Obsidian vault and synced to the knowledge graph.
    Insights and outcomes feed back into the KG for future reference, and any
    triggers for follow-up research (e.g. new questions or identified gaps) can
    spawn new `research.requested` events, starting the cycle anew.

**Lifecycle Flowchart:** The diagram below illustrates the progression through
these stages, including decision review feedback loops:

```mermaid
flowchart LR
    subgraph Research Phase
        A[Research Requested\n(Intake)] --> B[Research Proposed\n(Planning)]
        B --> C[Research Execution\n(In Progress)]
    end
    C --> D[Decision Ready\n(Decision Drafted)]
    D --> E[Decision Review\n(Peer Review)]
    E -->|Approved| F[Implementation Started\n(Execution)]
    E -->|Needs Revision| D
    F --> G[Implementation Ongoing\n(Metrics Collection)]
    G --> H[ROI Review Due\n(Post-Implementation)]
    H --> I[Cycle Complete\n(Knowledge Captured)]
```

## GitHub Issue Workflow and Projects v2 Integration

Each research cycle is tracked via a GitHub Issue that moves through a
**Projects v2** board with columns representing the lifecycle stages. The
orchestrator and agents interact with the issue through labels and status
changes:

- **Issue Creation (Intake):** A developer or stakeholder opens a GitHub Issue
  describing the problem. The issue is tagged with a label like
  `stage: research.requested` (or a specific event label). The Projects board
  automatically places it in an **“Intake”** or **“Requested”** column. The
  issue contains the raw request details (in free text or a simple YAML
  snippet).
- **`research.requested` → `research.proposed`:** The orchestrator detects the
  new issue with the intake label. It triggers the Research Intake Agent to
  process the issue. The agent’s output – a structured research proposal – can
  be posted as a comment or committed to the repo (e.g. in an `ideas/` or
  `proposals/` directory). The issue is then updated:

  - The issue gets a label `stage: research.proposed` (indicating a proposal is
    ready for planning) and is moved to a **“Proposed”** or **“Planning”**
    column on the project board.
  - Additional labels can capture metadata from the proposal, e.g.
    `category: MEM`, `priority: high`, etc., as well as an identifier label like
    `research-id: MEM_004`. The _research ID_ (e.g. `MEM_004`) is a unique code
    that will link all artifacts of this research across stages.
- **Approval & Execution:** A human might quickly review the proposal
  (human-in-the-loop governance) or the orchestrator can auto-approve based on
  criteria. Once approved, the orchestrator transitions the issue to **“Research
  In Progress.”** (This could be done by moving it to a “In Progress” column or
  adding label `status: in_progress`.) At this point, the Research Planning
  Agent is invoked.
- **`research.proposed` (Planning):** The agent takes the proposal and produces
  a detailed research plan. The plan is committed to the repository (e.g.
  `plans/RESEARCH_ID.md`). The issue moves to a **“Planned”** or **“Execution”**
  column. The orchestrator might attach the plan or a link in the issue. The
  issue status/label might be updated to `stage: research.ready` or simply
  remain “In Progress” while work is ongoing.
- **Research Execution:** As the team or agents execute the plan, the issue
  remains in “In Progress.” If using Projects v2 fields, a **Status** field
  (e.g. a single-select) can reflect this. The orchestrator can spawn sub-tasks
  or GitHub sub-issues for key milestones (optionally linking them to the main
  issue). For parallelizable research tasks, multiple agents (“swarm agents”)
  could be spawned simultaneously (e.g. one agent benchmarking Option A, another
  Option B). The orchestrator coordinates these and consolidates findings.
  Communication can happen via issue comments or pull requests attaching
  experiment results.
- **`decision.ready` (Decision Drafting):** When research milestones are
  completed (e.g. the plan’s “Decision Gate” criteria are met), the orchestrator
  (or the researcher) updates the issue with a label `stage: decision.ready` or
  moves the card to a **“Decision Drafting”** column. This triggers the Decision
  Agent. The agent generates a `Decision.md` document (markdown) with
  recommendations and an embedded YAML status update. The orchestrator commits
  this to the repo (e.g. `decisions/RESEARCH_ID_decision.md`) and also posts a
  summary or link on the issue. The issue’s frontmatter or project fields might
  be updated (e.g. set a Status field to “Decision Ready” and a custom field for
  `decision_date`). The YAML frontmatter patch from the agent is used to update
  the Obsidian note for the research (more on this syncing below).
- **`decision.review_requested` (Peer Review):** The orchestrator (or a human)
  next triggers a peer review. This could happen automatically by adding label
  `stage: decision.review_requested` or moving the card to **“Review”** column.
  The Peer Review Agent is invoked with the decision doc as input. It returns a
  **Peer Review Report** (markdown) indicating approval or required changes. The
  orchestrator attaches this review (e.g. as an issue comment or
  `reviews/RESEARCH_ID_review.md`). It also parses the review outcome:

  - If **APPROVED**, the issue gets a label `status: decision_approved` (and
    moves to an **“Approved”** or **“Ready for Implementation”** column).
  - If **NEEDS\_REVISION** or **REQUIRES\_REWORK**, the issue remains in the
    review column and may get a label `status: needs_revision`. The orchestrator
    enters a loop: it can prompt the Decision Agent to revise the decision doc
    using the review feedback, or assign the task to the human. Revised content
    can be updated in the decision doc (committed via PR for audit). Then the
    review agent is run again until approved. All review iterations can be
    tracked via comments or sub-tasks for traceability (audit hooks).
- **`implementation.started` (Implementation):** Once the decision is approved,
  the orchestrator adds label `stage: implementation.started` or moves the issue
  to an **“Implementation”** column. The Implementation Agent generates an
  **Implementation Tracking** document (markdown) with the plan, milestones,
  monitoring setup, etc. This is saved (e.g.
  `implementation/RESEARCH_ID_implementation.md`). The issue is updated with any
  key info (start date, target end date, responsible team, etc.). The status
  might be changed to “Implementation In Progress.” The orchestrator might also
  open child issues for specific implementation tasks or link to the project
  management system if separate.
- **Metrics Tracking:** During implementation, progress is tracked. The
  Implementation plan template defines a YAML format (see **Metrics Posting
  Instructions** in the template) for posting metric updates. For example, a
  GitHub Action or script could post a comment containing:

  ```yaml
  event: implementation.metrics_update
  research_id: MEM_003
  metrics:
    phase: "Phase 2 - Data Migration Testing"
    progress_percentage: 50
    success_metrics:
      - name: Query Response Time
        current_value: 50ms
        target_value: 45ms
        status: at_risk
    blockers:
      - description: "Data indexing running slow"
        severity: high
        resolution_eta: 2025-02-10
    next_milestone: "Phase 2 complete (Testing)"
    milestone_date: 2025-02-12
  ```

  The orchestrator (or a metrics agent) can automatically gather data from
  monitoring tools and post these updates on a schedule (daily/weekly) or on
  certain triggers. Each update could also be recorded in the KG (as an event
  node or properties of the implementation node). The Projects board can reflect
  progress via a custom field (e.g. % done).
- **`metrics.review_due` (ROI Analysis):** After the implementation is completed
  and has been in use for a certain period (e.g. 3 months), a review is
  scheduled. The orchestrator can schedule a trigger (via a timed GitHub Action
  or a date field in Projects). When the date arrives, the issue gets label
  `stage: metrics.review_due` and moves to **“ROI Review”** column. The ROI
  Analysis Agent runs, using inputs such as the original success criteria and
  actual metrics (which the orchestrator collates from the metrics updates or
  monitoring systems). The output is an **Implementation ROI Analysis** document
  (markdown) stored in the vault (e.g. `analysis/RESEARCH_ID_roi.md`). The
  orchestrator posts a summary in the issue (overall success status and ROI
  percentage).

  - The issue can then be marked **“Completed”** (closed or moved to a
    **“Done”** column). A label `status: completed` or `status: archived` might
    be applied.
  - Finally, any follow-up actions recommended (e.g. “initiate new research to
    improve X”) can spawn new issues (the orchestrator might automatically open
    a new `research.requested` issue referencing this one, achieving continuous
    improvement cycles).

**Label & Status Conventions:** Throughout this flow, consistent labels help
automate transitions. For example:

- Use `stage:` labels for the current lifecycle stage (only one such label at a
  time per issue). e.g. `stage:research.requested`, `stage:decision.review`,
  etc. This makes it easy for a GitHub Action or bot to detect and trigger the
  right agent.
- Use `status:` labels or a Project field for high-level status (e.g.
  `status: in_progress`, `status: pending_review`, `status: completed`). This
  can complement the stage label.
- Use category and priority labels from the YAML (e.g. `category: ARC` for
  architecture, or `priority: high`) to help triage and filter issues on the
  board.
- Use the unique `research_id` as a label or in the issue title to link all
  artifacts. For example, issue title might be prefixed with `[MEM_003]` or a
  label `MEM_003` is applied; this allows searching the repo/board for all items
  related to that research.

**Agent & Orchestrator Roles:** The orchestrator (Claude) listens to GitHub
webhooks or periodically polls the repository for these triggers. Each time a
stage transition occurs (new label, column change, or comment event), the
orchestrator decides whether to:

- invoke an **LLM agent** (with the appropriate prompt template and context) for
  the next step,
- or wait for human input (if a manual gate is desired).

GitHub Projects v2 provides a structured way to track this: as the orchestrator
moves the item through columns, it reflects the current state to all
participants. The **Projects board** becomes a high-level dashboard of all
research items and their stage (Intake, Planning, In Progress, Decision, Review,
Implementation, ROI, Done).

## Obsidian Markdown Templates for Each Stage (with YAML Frontmatter)

Each stage in the lifecycle has a corresponding **Obsidian markdown template**
that defines the input, output format, and guidance for the agent handling that
stage. We add a YAML frontmatter section at the top of each template file to
facilitate integration with the knowledge graph and orchestrator. The YAML
frontmatter includes metadata like the event name, related entity types, and
schema references for validation. The **body** of each template provides the
detailed instructions and output schema that the agent will follow.

Below, we outline each template with its YAML frontmatter and a snippet of its
contents:

### `research_requested.md` – Research Intake Template

```markdown
---
event: research.requested
role: "Research Intake Agent"
input_schema: ResearchRequest  # Pydantic model for intake (description, requestor, etc.)
output_schema: ResearchProposal  # Pydantic model for output YAML
output_entity: research_proposal  # Entity type to be created in KG
next_event: research.proposed  # Trigger after this output
---

# research.requested Prompt

## Context

You are a Research Intake Agent helping scope and prioritize incoming research
requests from developers...
```

**Key points in template:**

- **Input:** The agent receives a raw request (problem description, requestor,
  urgency, context, etc.). The template provides a **Sample Input** in YAML, for
  example:

  ```yaml
  request:
    description: "Need to research vector database options for 10M embeddings"
    requestor: "backend-team"
    urgency: "high"
    business_context: "Performance issues with current solution"
    deadline: "2 weeks"
  ```

- **Task:** The agent is instructed to clarify the research question, estimate
  effort, identify stakeholders/impact, suggest an approach, and categorize the
  request. (The template lists these steps in detail under “Task”.)

- **Output Format:** The agent must produce a structured **Research Proposal**
  in YAML. The proposal YAML is embedded in a Markdown code block in the output.
  For example, the template defines an output structure like:

  ```yaml
  research_proposal:
    title: "[Clear, specific title]"
    research_id: "[CATEGORY_###]" # e.g. MEM_004
    question: "[Specific, answerable research question]"
    category: "[AIO|MEM|TLI|ARC|DEV|UXP]"
    priority: "[high|medium|low]"
    complexity: "[low|medium|high]"
    confidence_level: "[low|medium|high]"
    estimated_effort: "[X weeks]"
    estimated_completion: "[YYYY-MM-DD]"
    methodology: "[approach description]"
    requestor: "[team/person]"
    stakeholders: ["[list of affected parties]"]
    decision_makers: ["[who needs to approve]"]
    success_criteria:
      - "[measurable outcome 1]"
      - "[measurable outcome 2]"
    business_impact: "[impact description]"
    urgency_reason: "[why this timeline]"
    dependencies: ["[external dependencies]"]
    immediate_actions:
      - "[first next step]"
      - "[setup tasks]"
  ```

  This output YAML is designed to be **Pydantic-compatible** – it can be parsed
  by a `ResearchProposal` model to validate types (e.g. category must be one of
  the allowed codes, dates in ISO format, etc.). The `research_id` is generated
  (like `MEM_004`) and will be used as the key to connect subsequent stages.

- **Examples and Checklist:** The template includes example inputs/outputs (e.g.
  _Database Performance_ request example) and a quality checklist for the agent
  to self-verify the proposal (e.g. question is specific, success criteria
  measurable, etc.). These ensure consistency and thoroughness.

When the orchestrator runs this template, it will validate the YAML output with
a Pydantic model. If validation fails (e.g. missing field or type mismatch), the
orchestrator can prompt the agent to fix it (ensuring high-quality structured
data before moving on).

### `research_proposed.md` – Research Planning Template

```markdown
---
event: research.proposed
role: "Research Planning Agent"
input_schema: ResearchProposal (approved)
output_schema: ResearchPlan  # structure of the detailed plan
output_entity: research_plan
next_event: (research execution or decision.ready)
---

# research.proposed Prompt

## Context

You are a Research Planning Agent helping design and structure approved research
topics...
```

**Purpose:** This template guides the agent in turning an approved proposal into
a full **Research Plan**. The plan is output as a **markdown document** (not
just YAML) since it’s a rich, structured outline.

**Key Template Features:**

- **Input:** An approved proposal YAML (from the previous stage) is provided.
  The template’s Sample Input shows an example `approved_proposal` with fields
  like `research_id`, `title`, `question`, etc., plus any auto-generated tags.

- **Task Outline:** The agent must produce a comprehensive plan, including:

  1. **Methodology Design:** Break the approach into phases.
  2. **Milestone Timeline:** Weekly (or relevant interval) milestones, with
     deliverables and dependencies.
  3. **Deliverables & Artifacts:** What documents or results will be produced.
  4. **Measurement Framework:** Metrics and criteria for success (when is
     decision-ready?).
  5. **Resource Plan:** Team roles, time allocation, tools and infrastructure
     needed, external dependencies (approvals, data, etc.).

- **Output Format:** The template provides a **Markdown outline** that the agent
  fills in. For instance, it starts with:

  ```markdown
  # Research Plan: [Title]

  ## Research Overview

  - **Research ID**: [CATEGORY_###]
  - **Question**: [Original research question]
  - **Duration**: [X weeks, start to decision]
  - **Team**: [Lead researcher + contributors]
  - **Priority**: [High/Medium/Low]
  - **Tags**: [tag1, tag2, ...]
  - **Similar Research**: [Link to related past research if any]

  ## Success Criteria

  - [ ] [Measurable outcome 1]
  - [ ] [Measurable outcome 2] ...
  ```

  and so on through **Research Methodology** (Phase 1, Phase 2, ...), **Resource
  Requirements**, **Risk Management** (with a table of risks), **Communication
  Plan**, **Decision Framework** (what evidence is needed to consider the
  research complete), **Quality Assurance**, etc. The agent populates each
  section.

  The resulting plan is a human-readable markdown document but also structured
  enough that key data can be parsed if needed:

  - E.g., the **Research ID**, **Team**, **Priority**, etc., are in a consistent
    format (key: value lists) which can be parsed or converted to YAML
    frontmatter in the saved file. We recommend saving the final plan file with
    a YAML frontmatter containing at least the `research_id`, `title`,
    `status: planning_completed` (or similar), and cross-references (like list
    of stakeholder IDs, if available). This frontmatter can be inserted by the
    orchestrator based on the content.

- **Examples:** The template contains example filled plans (for the database
  performance and architecture decisions cases), demonstrating what a good plan
  looks like.

- **Validation:** The orchestrator doesn’t need to fully parse this markdown
  output via Pydantic, but it can ensure key fields like **Research ID** match
  the earlier stage, and perhaps check that all sections are filled (no
  placeholder brackets left). The presence of checkboxes and tables can also be
  programmatically checked to ensure format correctness.

Once the plan is produced, the orchestrator or human confirms it, then research
execution proceeds. The plan document (in Obsidian) serves as a living reference
during execution, and progress (e.g., completion of phases) can be checked off
in this document.

### (Research Execution Stage)

_No single template file – this stage involves carrying out the plan._ Execution
might involve multiple tasks (code, experiments, data analysis) that are outside
the scope of static templates. However, the orchestrator can leverage agent
assistance here too:

- **Swarm agents vs sequential steps:** If the research plan has independent
  research questions or comparisons (e.g. evaluating three database options),
  the orchestrator can spawn multiple agents in parallel, each tasked with
  researching one option. They might use specialized prompts (not user-provided
  here) for gathering data or simulating experiments. For example, an agent
  might use a tool-augmented prompt to fetch performance stats or relevant
  literature.
- The orchestrator monitors these parallel tasks (each could log findings to the
  issue or a temporary file). Once all are done, the orchestrator (or a
  synthesis agent) compiles the results into a cohesive findings summary. This
  could even be done by another prompt or by using the decision agent directly
  if it’s capable of synthesizing raw findings.
- **Human-in-the-loop:** The solo entrepreneur can intervene here – running
  experiments, feeding results back to the system. The framework is flexible to
  allow manual data gathering and then resuming the automated workflow by
  triggering the next event when ready.

For the purpose of the knowledge graph, the execution phase might generate
**Insight** entities (pieces of knowledge). The system could capture significant
findings as they arise (e.g., an agent/human might create an Obsidian note for a
key insight, with YAML tagging it as an `insight` and linking it to relevant
concepts or experiments). These insight nodes can later be connected to
decisions as justification.

### `decision_ready.md` – Decision Documentation Template

```markdown
---
event: decision.ready
role: "Research Decision Agent"
input_schema: ResearchSummary  # (research_id, question, findings, experiments, constraints)
output_schema: DecisionDocument  # structured decision doc
output_entity: decision
next_event: decision.review_requested
---

# decision.ready Prompt

## Context

You are a Research Decision Agent helping synthesize research findings into
clear, actionable decisions...
```

**Function:** This template instructs the agent to produce a **Decision
Document** given the compiled research results. It’s a critical handoff where
research knowledge is converted into a decision for implementation.

**Key Template Elements:**

- **Input:** The agent is given a **research summary** – typically a YAML or
  structured payload including the `research_id`, original question, a list of
  key findings, summary of experiments, and constraints or criteria. For
  example:

  ```yaml
  research_summary:
    research_id: "MEM_003"
    question: "What vector database solution best meets our 10M embedding requirements?"
    findings:
      - "PostgreSQL+pgvector: 45ms avg query time, $800/month"
      - "Qdrant: 23ms avg query time, $1200/month"
      - "Pinecone: 18ms avg query time, $1500/month"
    experiments:
      - "Load test results with synthetic data"
      - "Integration complexity assessment"
    constraints:
      - "Budget limit: $1000/month"
      - "Query time requirement: <50ms p95"
      - "Team expertise: Strong PostgreSQL, limited with newer systems"
  ```

  This summary could be prepared by the orchestrator from the research outputs
  (possibly by another agent or manually). The structured format ensures the
  decision agent has the key facts and context.

- **Task:** The agent must:

  1. Synthesize findings into 2–3 **clear options** (each option with pros,
     cons, evidence).
  2. Make a **clear recommendation** of one option with rationale.
  3. Provide an **implementation plan** outline for the recommendation (phased
     steps).
  4. Define **success metrics** for post-implementation and a
     monitoring/rollback plan.
  5. Document the decision rationale and context (why it meets business needs,
     addresses constraints, etc.).
  6. **Update research status** – _important:_ the agent is instructed to
     include a YAML frontmatter patch to update the status of the research.

- **Output Format:** The template defines a Markdown structure for the decision
  doc. It starts with a YAML frontmatter block (marked as a “patch block” since
  it might be merged into an existing note’s frontmatter in Obsidian):

  ```yaml
  ---
  status: decision_made
  decision_date: 2025-01-10  # (example date)
  recommendation: "Adopt PostgreSQL + pgvector"
  implementation_weeks: 6
  confidence_level: high
  ---
  ```

  This is followed by the main document content, for example:

  ```markdown
  # Decision: [Research Title]

  ## Executive Summary

  **Recommendation**: [One-sentence decision]\
  **Rationale**: [Why this option meets the criteria best]\
  **Implementation Timeline**: [X weeks/months]\
  **Success Metrics**: [Key metrics for success]

  ## Research Context

  - **Question**: [Original research question]
  - **Success Criteria**: [from original proposal]
  - **Constraints**: [any constraints considered]

  ## Options Analysis

  ### Option 1: [Name] ⭐ _RECOMMENDED_

  **Summary**: ...\
  **Pros**: …\
  **Cons**: …\
  **Evidence**: …\
  **Implementation Effort**: …

  ### Option 2: [Name]

  **Summary**: ...\
  **Pros**: … **Cons**: …\
  **Evidence**: …\
  **Why Not Recommended**: …

  (... similarly for Option 3 if applicable ...)
  ```

  Then the **Implementation Plan** section (Phase 1, 2, 3 with tasks), a **Risk
  Assessment** table, **Success Metrics** (immediate, short-term, long-term
  checklists), **Monitoring Plan**, **Rollback Plan**, **Decision Authority**
  (who approves what), and an **Appendix** with links to detailed findings. The
  template is quite detailed to ensure all aspects of the decision are covered.

- After the human-readable content, the template requires a **machine-readable
  summary** in JSON at the end (within a code block):

  ```json
  {
    "event": "decision.ready.completed",
    "research_id": "[research_id]",
    "decision": {
      "recommendation": "[option name]",
      "implementation_weeks": [number],
      "estimated_cost": "[monthly/one-time cost]",
      "confidence_level": "[low|medium|high]",
      "key_metrics": {
        "performance": "[primary metric]",
        "cost": "[budget impact]",
        "timeline": "[weeks to complete]"
      }
    },
    "alternatives_considered": [
      {
        "option": "[name]",
        "reason_not_selected": "[key blocker]"
      }
    ],
    "next_action": "[immediate next step]",
    "approval_needed_from": ["[role1]", "[role2]"]
  }
  ```

  This JSON allows the orchestrator or any external dashboard to easily ingest
  the decision outcome. It duplicates some info from the main doc
  (recommendation, etc.) in a structured form. We ensure this is consistent with
  the doc content. A Pydantic model (e.g. `DecisionSummary`) can validate this
  JSON. If the agent’s output JSON is invalid or missing fields, the
  orchestrator can catch it as an error.

- **Examples & Quality Checks:** The template offers example outputs (for the
  vector DB and architecture scenarios) and a checklist to ensure quality: e.g.
  “Clear recommendation stated”, “Evidence supports recommendation”, “Risks
  identified with mitigations”, etc. The agent should verify these before
  finalizing.

When this stage is complete, the research’s Obsidian note frontmatter is updated
(status: decision\_made, etc.), and the decision node in the KG can be created
with links to chosen solution concepts and stakeholders (details in KG section
below).

### `decision_review.md` – Decision Peer Review Template

```markdown
---
event: decision.review_requested
role: "Research Peer Reviewer Agent"
input_schema: DecisionDocument
output_schema: DecisionReview  # structured review output
output_entity: review_report
next_event: (approval -> implementation.started, revision -> decision.ready)
---

# decision.review_requested Prompt

## Context

You are a Research Peer Reviewer conducting quality assurance for research
decisions...
```

**Purpose:** This template has the agent act as a **peer reviewer**, evaluating
the decision document against a rigorous checklist.

**Highlights:**

- **Input:** The complete decision document (likely as a markdown text or a path
  to it) along with context like who the reviewer is (could be provided or
  assumed by the agent persona), implementation urgency, stakeholders, etc. The
  sample input YAML illustrates a `review_request` with fields for
  `research_id`, link to `decision_document`, reviewer identity, deadline,
  urgency, stakeholders involved. This provides context to the agent about how
  critical/timely the review is and who the audience is.

- **Task:** The agent goes through a **10-point checklist** covering:

  1. Decision Clarity
  2. Evidence Quality
  3. Methodology Rigor
  4. Statistical Validity
  5. Limitations Acknowledged
  6. Business Impact clarity
  7. Implementation Feasibility
  8. Success Metrics appropriateness
  9. Risk Management completeness
  10. Dependencies documented

- The template instructs the agent to assess each and then produce structured
  feedback.

- **Output Format:** A Markdown **Peer Review Report** with sections:

  - **Review Summary:** The reviewer name/role, date, an **Overall Status** (✅
    APPROVED / ⚠️ NEEDS REVISION / ❌ REQUIRES REWORK), a Confidence Level, and
    critically a `review_result` field that is clearly either `APPROVED`,
    `NEEDS_REVISION`, or `REQUIRES_REWORK` (for easy parsing). In the template,
    `review_result` is shown next to the status for machine-readability.
  - **Quality Assessment:** Divided into Strengths (✅), Areas for Improvement
    (⚠️), and Critical Issues (❌). The agent lists bullet points under each,
    summarizing what was good or lacking.
  - **Detailed Checklist Results:** A table with each of the 10 criteria, a
    status (✅/⚠️/❌), a score (1–10), and comments. This quantifies the review.
  - **Required Changes:** If not fully approved, this section lists what needs
    fixing, categorized by High/Medium/Low priority. Each issue has details:
    what's missing, suggested approach, impact if not fixed.
  - **Implementation Review:** Further breakdown of feasibility, risk review,
    metrics review – ensuring the plan is realistic and metrics are measurable.
  - **Additional Considerations:** Stakeholder alignment, long-term
    implications, etc.
  - **Recommendations for Author:** Concrete advice on what to do before
    resubmission and for future research.
  - **Approval Conditions:** If conditionally approved, list conditions; for an
    approved status, note that high-priority issues must be resolved, etc.
  - **Next Steps:** A short list of actions (what the author should do, timeline
    for re-review, when implementation can proceed, etc.).

- **Examples:** Provided for an approved (with minor conditions) review and a
  needs-major-revision review, to illustrate formatting.

- **Outcome Handling:** The orchestrator will parse the `Overall Status` or the
  `review_result`. If APPROVED, it proceeds. If revision needed, it can loop:

  - Possibly the orchestrator adds a comment to the issue with the summary of
    required changes, and assigns the issue back to the decision drafter (or
    triggers an agent to revise).
  - The presence of a structured list of issues in the review output can even
    let the orchestrator generate a to-do for the decision agent or open
    sub-issues for each required change.
  - The orchestrator ensures the research stays in “Review” column until the
    `review_result` is APPROVED. Approved might also trigger adding a label like
    `decision_approved` which is a prerequisite for the next stage.

This peer review stage is an **audit hook** in the process – it ensures quality
and provides a point where the human leader could also inspect the output. The
framework allows replacing or augmenting the AI peer review with a human review
if desired (e.g. the solo entrepreneur could manually fill a similar checklist
and put the results into the system).

### `implementation_started.md` – Implementation Tracking Template

```markdown
---
event: implementation.started
role: "Implementation Tracking Agent"
input_schema: Decision & Plan  # decision recommendations + context
output_schema: ImplementationPlan  # structured implementation doc
output_entity: implementation_plan
next_event: implementation.metrics_update (ongoing), then metrics.review_due
---

# implementation.started Prompt

## Context

You are an Implementation Tracking Agent helping set up monitoring and progress
tracking for approved research decisions...
```

**Function:** Once a decision is made, this template helps the agent create a
comprehensive **Implementation Plan/Tracking** document to guide the rollout of
the decision and ensure its success criteria are measured.

**Key Template Details:**

- **Input:** The agent gets the approved decision document (or key excerpts: the
  recommendation, success metrics, timeline, team info) as well as context like
  team assigned, start date, etc. The sample input YAML
  (`implementation_request`) includes `research_id`, a one-line `decision`
  summary, the `implementation_team` members, overall `timeline`, success
  metrics from the decision, and a link to a detailed implementation plan if any
  exists. This gives the agent what it needs to craft the tracking plan.

- **Task Outline:** The agent must:

  1. **Establish progress monitoring** – break implementation into milestones,
     define indicators and reporting cadence.
  2. **Create accountability framework** – assign owners, define review gates,
     escalation paths.
  3. **Design success validation** – how to test and validate that success
     metrics are met at each stage.
  4. **Plan risk mitigation** – identify and plan for risks with contingencies.
  5. **Enable stakeholder visibility** – set up status reports, check-ins,
     communications plan.

- **Output Format:** A Markdown **Implementation Tracking** document with
  sections:

  - **Implementation Overview:** key details like Research ID, Decision (summary
    of what is being implemented), Team (lead and members), Timeline (start →
    end date), and Budget (if applicable). These are listed in bullet form,
    which again can be extracted. It’s recommended to also capture these in YAML
    frontmatter of the file for the KG sync (e.g., `type: implementation_plan`,
    `status: in_progress`, `start_date:`, `target_end_date:`, `team: [...]`,
    etc.). The agent provides the content, and the orchestrator can augment with
    actual dates or IDs.
  - **Success Criteria:** A reiteration of the original success metrics
    (checkboxes with targets) that the implementation must achieve (e.g. “Query
    response time <45ms p95”). This keeps everyone focused on the goals.
  - **Implementation Phases:** Broken into Phase 1, Phase 2, Phase 3 (with weeks
    or time frames). For each phase:

    - **Objective** – what that phase accomplishes.
    - **Owner** – who is responsible.
    - **Key Activities** – tasks to do (as checklist items).
    - **Milestone Criteria** – what constitutes completion of the phase
      (deliverables, approvals).
    - **Success Metrics** – which success metrics or intermediate metrics are
      measured in that phase and their target.
    - **Review Gate** or **Dependencies** as appropriate – e.g. end-of-phase
      review date and criteria, or prerequisite from previous phase.
    - **Risk Mitigations** (for later phases) if needed.
  - **Monitoring Framework:** Defines how progress will be tracked:

    - Daily tracking metrics (progress %, issues count, etc.) and reporting
      method (standups, dashboard, etc.).
    - Weekly review meetings (attendees, agenda).
    - Escalation triggers (e.g. milestone delays > X days, metrics off target,
      critical blocker unresolved, etc.).
    - Metrics Collection details: For each category (Performance, Cost, Team)
      specifying how each metric will be measured, frequency, baseline, and
      alert conditions.
  - **Risk Management:** Active risk monitoring table (risk, probability,
    impact, early warning signs, mitigation plan). Also, **Contingency Plans**:

    - **Rollback Procedure:** triggers and step-by-step rollback steps, plus
      expected recovery time.
    - **Alternative Approaches:** plan B or C if the implementation fails (could
      mention fallbacks or parallel approaches).
  - **Stakeholder Communication:**

    - Status reporting schedule (weekly reports format and audience).
    - Review schedule (key review meetings at certain weeks).
    - Issue escalation path (levels of escalation, who and how quickly).
  - **Quality Assurance:** Testing strategies (unit, integration, performance,
    UAT), code review process, documentation requirements to ensure the
    implementation is done with high quality.
  - **Success Validation:** How to validate each milestone’s success, and how to
    validate final success in terms of business outcomes (performance
    improvement, cost savings, etc.). Also long-term monitoring plans (3-month,
    6-month, annual review after go-live).
  - **Implementation Handoff:** Criteria for closing out the implementation:

    - Documentation deliverables (updated manuals, runbooks, training
      materials).
    - Knowledge transfer tasks (ensure support teams are trained, etc.).
    - Closure criteria (all success metrics met, system stable for X days,
      post-implementation review done, etc.).

- **Metrics Update Integration:** At the end of the template, there is a section
  **Metrics Posting Instructions** which provides the YAML format (as shown
  earlier) for posting updates. This is basically telling the user/system: “for
  all tracked metrics, post updates to the research system with this structure.”
  The orchestrator will use this as a guide to listen for
  `implementation.metrics_update` events (which can simply be the presence of
  that YAML in comments or logs) to update the KG. Each update could be parsed
  into, for instance, a `MetricsUpdate` Pydantic model and inserted into the
  database (with relationships to the implementation node). The template
  suggests posting daily for high-priority projects, weekly for standard, and on
  blocker events – giving a cadence for the orchestrator or monitoring agent.

- **Examples:** The template includes a filled example for a “Vector Database
  Migration to PostgreSQL+pgvector” with real tasks and metrics, and another for
  an “API Integration Implementation”. These serve as references for the agent.

After this stage, the team executes the implementation according to this plan.
The plan document is a living artifact in Obsidian (which can be updated with
check marks as tasks complete and actual metric values as they come in, if
desired). The orchestrator’s role now is mostly monitoring: it can ensure metric
updates are coming in, and possibly raise alerts (via GitHub issues or
notifications) if escalation triggers are hit (for example, if a milestone is
delayed and an `at_risk` status appears in an update, the orchestrator could tag
the issue with `status: at_risk` or ping the stakeholder).

### `metrics_review.md` – ROI Evaluation Template

```markdown
---
event: metrics.review_due
role: "ROI Analysis Agent"
input_schema: ImplementationReport  # (completion data, actual metrics, original criteria)
output_schema: ROIReport
output_entity: roi_analysis
next_event: (potentially research.requested for follow-ups)
---

# metrics.review_due Prompt

## Context

You are a Research ROI Analysis Agent helping evaluate the success and impact of
completed implementations...
```

**Purpose:** This template guides the agent to compare post-implementation
outcomes against the original research predictions and success criteria,
calculate ROI, and extract lessons learned. It closes the loop by capturing
whether the research & decision actually delivered value.

**Key Elements:**

- **Input:** A payload with the implementation completion info, review
  timeframe, original success criteria, and actual metrics. For example, the
  sample `review_request` YAML includes:

  - `research_id`,
  - `implementation_completion` date,
  - `review_period` (e.g. "3 months post-implementation"),
  - `original_success_criteria` (list of the targets from the
    proposal/decision),
  - `actual_metrics` (the real outcomes like query\_time\_p95, monthly\_cost,
    downtime),
  - `additional_data` (like team satisfaction survey results, incident count,
    performance trend). The orchestrator likely compiles this from the metrics
    updates and any surveys or reports at the end of the period.

- **Task Outline:** The agent performs an analysis in steps:

  1. **Measure success vs original criteria:** Create a comparison of predicted
     vs actual for each success criterion, quantify variance, and mark each as
     met or not (✅/⚠️/❌).
  2. **Calculate business impact and ROI:** Quantify benefits (performance
     gains, cost savings, etc.), including intangible benefits (team efficiency,
     user satisfaction), and then compute ROI (return on investment), payback
     period, etc.
  3. **Analyze prediction accuracy:** Evaluate how accurate the initial research
     predictions were (e.g. was performance improvement as expected? Did costs
     differ?), identify why any discrepancies happened.
  4. **Document lessons learned:** What worked well in the research and
     implementation, and what could be improved next time (methodological or
     process improvements).
  5. **Plan ongoing monitoring:** Recommend how to continue tracking success
     over the long term and identify triggers that would warrant new research or
     adjustments (essentially a sustainability plan).

- **Output Format:** A **Post-Implementation Review** markdown document with
  sections:

  - **Executive Summary:** A quick overview: research ID, implementation period,
    review period, an Overall Success rating (✅ SUCCESS / ⚠️ PARTIAL / ❌ BELOW
    EXPECTATIONS), and an ROI figure (like X% or \$Y value). This summarizes in
    a management-friendly way how the project fared.
  - **Success Criteria Assessment:** A table comparing each original success
    criterion target vs actual vs variance, with a status symbol:

    ```markdown
    | Success Criteria            | Predicted        | Actual        | Variance    | Status |
    | --------------------------- | ---------------- | ------------- | ----------- | ------ |
    | Query response time <45ms   | 42ms (predicted) | 42ms (actual) | ±0%         | ✅ Met |
    | Monthly cost <$900          | $850 predicted   | $850 actual   | 0%          | ✅ Met |
    | Migration downtime <4 hours | 4 hours          | 3.5 hours     | +13% better | ✅ Met |
    ```

    (The template even suggests using a diff-style table to highlight
    improvements/regressions with arrows.)
  - **Success Analysis:** A breakdown of what exceeded expectations, what met
    expectations, and what fell short, with bullet points explaining each.
  - **Business Impact Analysis:**

    - Quantitative Benefits: e.g. performance improvements quantified (like 25ms
      latency improvement, 50% capacity increase), cost impacts (savings per
      month/year, efficiency gains in hours), totaled into an annual value.
    - Qualitative Benefits: team impact (productivity, reduced maintenance),
      user experience improvements, strategic value (e.g. enabled future
      capabilities, reduced technical debt).
  - **ROI Calculation:** Lists the **Investment** (research time, implementation
    cost, any tool or opportunity costs) and the **Return** (annual benefits in
    \$ or %). Then calculates ROI % and payback period. Also gives context like
    what is considered High/Medium/Low value (the template sets >200% ROI as
    high-value, etc.).
  - **Research Methodology Evaluation:** Looks back at the research phase:

    - Prediction accuracy: categorize predictions into highly accurate,
      moderately accurate, inaccurate, with examples and analysis of why.
    - Methodology strengths and weaknesses: what research approaches proved
      reliable, and where did the research approach fail to foresee something.
  - **Lessons Learned:**

    - **What Worked Well** in both research process and implementation approach.
    - **Improvement Opportunities** for future research and future
      implementation.
    - **Actionable Recommendations:** a short list of specific actions (e.g.
      “include edge case testing in future research” or “engage X team earlier
      next time”).
  - **Long-term Monitoring Plan:** Suggest metrics to monitor on an ongoing
    basis, review schedule (e.g. 6-month, annual post-mortems), and conditions
    that should trigger new research (like if performance degrades or
    requirements change).
  - **Knowledge Capture:** Ensure that the knowledge gained is recorded:

    - Documentation updates needed,
    - Team learning outcomes (skills developed, knowledge transfer done),
    - Best practices updated in the organization’s knowledge base.
  - **Recommendations:**

    - For similar future research projects (methodology, data collection,
      stakeholder engagement improvements),
    - For the organization (process changes, investment criteria, knowledge
      management enhancements).
  - **Conclusion:** Overall assessment, top 3 insights from this project, and
    future implications (how this influences strategy or future decisions).

- **Examples:** The template provides an example of a very successful case (with
  340% ROI) and a mixed-result case for the architecture decision (partial
  success, some criteria not met, ROI 120%). This helps the agent format
  appropriately for different outcomes.

This report is the final artifact of the cycle. The orchestrator will likely
extract key pieces for the KG:

- Create/update an **Insight** node for each major lesson or outcome (positive
  or negative).
- Update the **Decision** node with actual outcome metrics (perhaps as
  properties or related "achieved\_metric" edges).
- Mark the **Decision** as successful or not (could be an attribute or an edge
  like `validates`/`invalidates` some prior assumption).
- Possibly create a link from this research cycle to new research needs (the
  template’s triggers for future research can be automatically scanned: e.g. if
  “Team velocity reduction 25%” was an issue, the system might create a
  follow-up research request to address developer workflow, etc. This could be
  automated: if any success criteria status is ❌ or ⚠️, orchestrator flags
  those for potential action).

Finally, the issue is closed, and all relevant data is synced.

## Knowledge Graph Schema: Entities and Relations

All the artifacts and data from the above process are represented in a
**Knowledge Graph (KG)**, with Obsidian markdown files (plus YAML frontmatter)
serving as the source of truth for nodes and edges. The system uses Supabase
(with a GraphQL interface) to store a synchronized copy of the graph (making
queries and updates easier, and allowing future migration to Neo4j or similar).
We define **canonical entity types** and **relation types** for this knowledge
graph:

### Entity Types

Each entity is an Obsidian note (markdown) with a YAML frontmatter capturing its
properties (fields). We ensure the YAML keys align with property names in the
GraphQL/Neo4j schema (Cypher-compatible naming, e.g. snake\_case or camelCase
without spaces). Important entity types include:

- **Research Proposal / Project** – _The research initiative itself._ **Type
  Name:** `research_project` (or `research_proposal` for the initial proposal
  note). **Purpose:** Acts as an anchor for a research cycle. Contains
  overarching info about the research question and status. **Frontmatter
  fields:** `id` (the `research_id` code, e.g. "MEM\_003"), `title`, `question`,
  `category`, `status` (e.g. `in_progress`, `decision_made`, `completed`),
  `priority`, `requestor`, `stakeholders` (list of stakeholder ids),
  `decision_makers` (ids), `created_at`, `completed_at` etc. **Edges:** Will
  link to other entities like concept, decision, etc. via relationships (see
  below). **Note:** We might actually not need a separate “research\_project”
  node if we treat the Decision or the Question concept as the main node.
  However, having a `research_project` node makes it easy to group all events of
  one cycle. The initial proposal file and the research plan can be seen as
  documents attached to this node. In practice, we could implement it such that
  the Obsidian note for the research proposal doubles as the research\_project
  node (with status evolving via frontmatter updates).
- **Decision** – _A decision/recommendation resulting from research._ **Type
  Name:** `decision`. **Fields:** `id` (could use the same research\_id or an
  extended one if multiple decisions, but here 1 research yields 1 main
  decision), `research_id` (back-reference), `recommendation` (short text),
  `decision_date`, `confidence_level`, etc. The detailed rationale is in the
  markdown body. **Edges:** `decision` **implements** some solution
  (concept/method), is **justified\_by** insights, **addresses** some initial
  question (concept), and is **approved\_by** stakeholders or roles (if tracking
  approvals). In Obsidian, the decision is documented in a note (created by
  decision\_ready agent). The frontmatter for this note could include:
  `id: MEM_003_decision`, `type: decision`, `research_id: MEM_003`,
  `status: approved` (or `pending_review` initially),
  `recommendation: "Use PostgreSQL pgvector"`, `date: 2025-01-10`, etc. The JSON
  summary is also stored in the note for external systems to consume.
- **Concept** – _A key concept or problem area._ **Type Name:** `concept`.
  **Description:** This is a broad category for domain knowledge – could be a
  technology, technique, requirement, etc. The “research question” itself might
  be modeled as a concept node or a problem statement. For instance, “Vector
  Database Performance” or “Payment Service Architecture” could be concept
  nodes. Additionally, specific technologies like “PostgreSQL pgvector” or
  “Microservices architecture” might be concept (or `method`, see below).
  **Fields:** `id` (unique name or UID, e.g. `concept.vector_database`), `name`,
  `description`, maybe `alias` (list of alternate names), `created_at`.
  **Edges:** Concepts relate to each other (e.g. `related_to`), or to methods
  (maybe method is a subtype of concept). A research project **addresses** a
  concept (problem concept), a decision **implements** a concept (solution
  concept), an insight might **invalidates** or **supports** a concept, etc.
  This entity type helps accumulate domain knowledge outside of specific
  projects.
- **Method/Technology** – _A specific solution approach or technology._ **Type
  Name:** `method` (or `technology`). **Description:** We separate this from
  concept to emphasize actionable solutions or approaches (e.g. “Using a Vector
  DB”, “Microservice Architecture”, "Benchmarking method"). But this could be
  merged with concept in practice by using a tag or subtype field. **Fields:**
  similar to concept (name, description, aliases, maybe vendor info if tech).
  **Edges:** A `method` can be linked as an **option** in a research (like
  research considered these methods), or a decision **implements** a method. It
  might have an `instance_of` or `subcategory_of` a broader concept. For
  example, _“PostgreSQL+pgvector”_ (method) might be linked to _“Vector
  database”_ (concept category).
- **Insight** – _A discrete piece of knowledge or finding._ **Type Name:**
  `insight`. **Description:** Captures a result or observation from research.
  For example, “Qdrant had 18ms query time at higher cost” or “Team velocity
  dropped 25% with microservices”. These often justify decisions or highlight
  important discoveries. **Fields:** `id` (could be auto or a slug),
  `description` (the statement of insight), `source` (which research/experiment
  led to it), `confidence` (if applicable), `created_at`. The note body might
  contain more context or data. **Edges:** Insights are typically
  **generated\_by** a research project (or an experiment agent), and they
  **justify** or **invalidate** decisions or concepts. E.g., an insight “Latency
  18ms” **justifies** choosing Pinecone _if_ that was the fastest; another
  insight “Cost exceeds budget” **invalidates** Pinecone option. We also link
  insights to the specific concept or method they refer to (e.g. insight about
  PostgreSQL’s performance links to the PostgreSQL method node).
- **Stakeholder** – _An individual or team with interest in the research._
  **Type Name:** `stakeholder`. **Fields:** `id` (could use something like team
  name or a short handle), `name`, `type` (e.g. "team", "person", "role"),
  `area` (their domain, like "Infrastructure Team" or "CTO"), etc. **Edges:**
  Stakeholders can **request** research (stakeholder -> research\_project via
  `requested` or `requested_by` edge), are **impacted\_by** decisions
  (stakeholder <- decision via `impacts`), or **approve** decisions (if they are
  decision makers). In YAML, the research proposal had `requestor` and
  `stakeholders` fields referencing these. We would likely create stakeholder
  nodes for each unique one and then use relations to link.
- **Agent** – _An AI agent or human agent participating._ **Type Name:**
  `agent`. **Fields:** `id` (e.g. "ResearchIntakeAgent" or a human’s username),
  `name`, `agent_type` (`AI` or `human`), `role` (what they do, e.g. "Intake",
  "Planning", "Orchestrator"). **Edges:** Agents **contribute\_to** or
  **authored** artifacts: e.g. an agent node connects to a proposal via
  `authored_by` (or inverse `authored` edge from agent to proposal). We might
  log which agent (which prompt) generated which output for audit. For human, if
  the person did the review, link their stakeholder/agent node to the review
  output via `reviewed_by`.

_(Additional entity types could exist, like `Experiment` if we wanted to log
each experiment as an entity, or `Metric` as an entity type to track metric
definitions. But the above cover the major knowledge nodes.)_

Each entity note’s YAML frontmatter includes an `id` that uniquely identifies it
(we can use human-readable IDs like `MEM_003` for research project or a GUID,
but ensure uniqueness), a `type` (from above types), and properties. For
example, an **Insight** note might look like:

```markdown
---
id: "insight_MEM_003_latency"
type: insight
research_id: "MEM_003"
description: "Pinecone achieved 18ms avg query time in testing, fastest among options."
origin: "Phase 2 load testing"
created: 2025-01-05
---

**Insight:** Pinecone had the lowest query latency (18ms avg) but at higher
cost.
```

And a **Concept** note example:

```markdown
---
id: "concept_vector_database"
type: concept
name: "Vector Database"
aliases: ["Vector Search Engine", "Approximate Nearest Neighbor DB"]
description: "Database optimized for vector similarity search at scale."
---

A **Vector Database** is a data store designed for efficient similarity searches
on high-dimensional vectors...
```

By using consistent frontmatter keys (like `id`, `type`, etc.), we can easily
sync these to Supabase tables or Neo4j. For instance, we might have a table
`entities` with columns: `id (PK)`, `type`, `name`, `description`,
`properties (JSONB)`, `created_at`, etc., and a table `edges` with `id (PK)`,
`type`, `from_id (FK to entities.id)`, `to_id`, and maybe `properties` (like
timestamp or weight). The YAML can be parsed to insert these. We ensure the
field names are “Cypher-compatible,” meaning they can map to Neo4j property keys
or relationship types without conflicts (e.g., avoid starting with numbers or
using spaces).

### Relation Types

Relations define how entities connect in the graph. We list key relation types
and their semantics:

- **`requested_by`** – connects a **research\_project** to the **stakeholder**
  who requested it. (Or inverse, stakeholder `requests` research\_project.) This
  captures origin. _Example:_ stakeholder “backend-team” `requested_by` ->
  research\_project “MEM\_003”.
- **`addresses`** – links a **research\_project** to a **concept** (problem
  domain) it addresses. Alternatively, could link the question to a concept
  node. _Example:_ research\_project “MEM\_003” `addresses` -> concept “Vector
  Database Performance”.
- **`has_plan`** – links a **research\_project** to the **research\_plan**
  document (if treating the plan as separate node). Could also just treat plan
  as an attribute of the project. Similar for proposal and other stage
  documents. If using separate nodes for documents, then `has_proposal`,
  `has_plan`, `has_decision`, etc., could be used.
- **`contributes`** – links an **agent** to an artifact (proposal, plan, etc.)
  they contributed to. E.g., ResearchIntakeAgent `contributes` ->
  research\_proposal. This can track AI vs human contributions.
- **`justifies`** – links an **insight** to a **decision** (or to an **option**
  in the decision). It means that insight provides evidence supporting the
  decision. _Example:_ insight “pgvector latency 45ms meets requirement”
  `justifies` -> decision “Use pgvector”. (We might also use `supports`
  synonymously.)
- **`invalidates`** – links an **insight** to a **concept** or **option** to
  denote it disproves or challenges it. _Example:_ insight “Team velocity -25%
  with microservices” `invalidates` -> concept "Microservices solves all scaling
  issues" (if that concept existed as assumption) – or it could invalidate a
  decision if it was made on false assumptions. This relation is useful for
  capturing when new knowledge overturns previous beliefs or recommendations.
- **`implements`** – links a **decision** to a **method/technology** that is
  being implemented. Essentially, the decision chooses to implement that
  solution concept. _Example:_ decision “Use PostgreSQL+pgvector” `implements`
  -> method "PostgreSQL pgvector". We could also link decision to a higher-level
  concept via `implements` or another relation like `selects` or `chooses`. But
  `implements` or `adopts` conveys that the organization will adopt that method.
- **`impacts`** – links a **decision** to a **stakeholder** or **metric** that
  it impacts. This can denote who is affected. E.g. decision `impacts` ->
  stakeholder "Infrastructure Team" (they have to maintain it) or decision
  `impacts` -> concept "Operational Cost". This is more of a semantic link for
  analysis; might not be needed if we capture in text, but could be added for
  queryability (like "show me all decisions impacting security team").
- **`approved_by`** – links a **decision** to a **stakeholder** (or agent) who
  approved it. E.g. decision `approved_by` -> stakeholder "CTO". If formal
  sign-off is tracked.
- **`alias`** – connects two **concept** nodes (or any entity) that are
  essentially the same thing (aliases). Alternatively, we use an `aliases` list
  in node properties as shown, but in a graph, having explicit `alias_of`
  relationships can help merge nodes. E.g., concept "ANN Database" `alias_of` ->
  concept "Vector Database".
- **`related_to`** – a general relationship between **concepts** or **methods**
  that are related. E.g. concept "Vector Database" `related_to` -> concept
  "Embedding".
- **`supersedes`** – links a **decision** to a **past decision** that it
  replaces. For instance, if a new research leads to changing a prior decision,
  we could mark the old decision node as superseded by the new one. This is a
  temporal relation indicating evolution of strategy.
- **`follow_up`** – links an **insight** or **ROI report** to a new
  **research\_request** that it spawned. E.g. the ROI analysis identified a gap,
  leading to a new research request issue; connect them via `follow_up` or
  `triggers`.

These relation types are designed to cover the justification and implementation
traceability: from stakeholder needs to research to insights to decisions to
outcomes. Each relation itself can be represented as a small YAML in Obsidian
(or we can derive them from context when syncing to the DB). For example, we
might represent an edge in a file `edge_MEM_003_optionA_justifies.md`:

```yaml
---
type: justifies
from: insight_MEM_003_optionA_perf # Insight ID
to: decision_MEM_003 # Decision ID
---
Option A performance meets requirements, justifying the recommendation.
```

However, it might be simpler to not create separate files for each edge.
Instead, maintain relationships in the frontmatter of the entity notes. For
instance, in the **Decision** note frontmatter, include:

```yaml
implements: "method_postgresql_pgvector"
justified_by:
  - "insight_MEM_003_pgvector_performance"
  - "insight_MEM_003_team_expertise"
invalidates:
  - "concept_legacy_vector_solution"
stakeholders_approved: ["stakeholder_cto", "stakeholder_engineering_manager"]
```

And in an **Insight** note frontmatter, one could list:

```yaml
supports: "decision_MEM_003"
context: "MEM_003 Phase 2 testing"
```

The orchestrator (or a sync script) would parse these and create corresponding
edges in the graph database. We should choose a consistent approach for edges to
avoid duplication. A good practice is one source of truth for each edge (either
always list in the source node or always in the target node). For clarity, we
can list relations under the source node’s YAML.

All these fields are _Cypher-compatible_ (no spaces or special chars, and types
can directly map to Neo4j relationship types). For example, an edge `implements`
in the YAML can become a Neo4j relationship
`(:Decision)-[:IMPLEMENTS]->(:Method)`.

### Pydantic Models for Entities/Edges

To ensure data integrity, each entity type can have a corresponding Pydantic
model used when parsing YAML frontmatter or agent outputs. For brevity, we won’t
list all code here, but as an illustration:

```python
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
class ResearchProposal(BaseModel):
    title: str
    research_id: str
    question: str
    category: Literal["AIO","MEM","TLI","ARC","DEV","UXP"]
    priority: Literal["high","medium","low"]
    complexity: Literal["low","medium","high"]
    confidence_level: Literal["low","medium","high"]
    estimated_effort: str
    estimated_completion: str  # could use date
    methodology: str
    requestor: str
    stakeholders: List[str]
    decision_makers: List[str]
    success_criteria: List[str]
    business_impact: str
    urgency_reason: str
    dependencies: List[str]
    immediate_actions: List[str]
```

And similarly, `DecisionSummary`, `MetricsUpdate`, etc., can be defined. These
models are used to validate agent outputs (the YAML and JSON segments). The
orchestrator will log an error or ask the agent to correct output if validation
fails at any stage.

## Orchestration Logic and Workflow (Claude Orchestrator Design)

The orchestrator (Claude or a similar LLM in a loop) is the “conductor” that
manages stage transitions, calls the right agent prompts, and ensures data flows
into the KG. We outline how the orchestration works, including when to use
parallel agents vs sequential, how GitHub events trigger actions, how outputs
are validated, and how memory is used.

### Event-Driven Triggers and Handoff

The orchestrator runs a **workflow script** (for example, a file like
`claude/commands/workflow/kb-cycle.md`) that defines the logic for each event.
Pseudocode or structured logic might look like:

````markdown
**On Event: research.requested**

- If a new issue is created or labeled `research.requested`:
  1. Fetch issue details (description, etc.)
  2. Invoke `research_requested.md` template with the issue content as input.
  3. Get the `research_proposal` YAML output. Validate with `ResearchProposal`
     model.
  4. If valid, post the proposal: commit `proposals/{research_id}.md` and/or
     comment on issue.
  5. Add label `stage: research.proposed` (proposal ready) to issue, move issue
     to "Proposed" column.
  6. Optionally, notify humans (mention the requestor that proposal is ready for
     approval).
  7. Record new graph nodes: create `research_project` node and link stakeholder
     via `requested_by`.

**On Event: research.proposed**

- When issue labeled `research.proposed` (proposal approved for planning):
  1. Retrieve the proposal data (from file or issue comment).
  2. Invoke `research_proposed.md` with the approved proposal as input.
  3. Agent returns a Research Plan markdown. (If large, ensure it stays under
     token limits by maybe summarizing long lists of tasks.)
  4. Commit the plan to `plans/{research_id}_plan.md`.
  5. Update issue: add label `status: in_progress` and move to "Research In
     Progress".
  6. Update graph: link the plan to the research_project (edge `has_plan`).
     Possibly add concept nodes for any tags or techniques mentioned, link via
     `addresses` or `considers`.
  7. Notify that research execution can start.

**On Research Execution Completion (no single event)**

- When research tasks finish (this could be indicated by a human comment like
  "Research complete" or reaching the target date):
  1. Orchestrator gathers results. Possibly compile a `research_summary` YAML:
     include key findings from experiments and success criteria status if any.
     (It might parse experiment logs or prompt an agent to summarize raw data
     into bullet points for findings.)
  2. Label issue `stage: decision.ready` and move to "Decision" column.

**On Event: decision.ready**

- When issue labeled `decision.ready`:
  1. Prepare input: the compiled `research_summary` YAML as shown in template.
  2. Invoke `decision_ready.md` with that input.
  3. Receive the decision document (markdown) from agent. Validate:
     - Ensure the YAML frontmatter patch is present and well-formed (parse it to
       confirm keys like `status: decision_made` etc.).
     - Ensure JSON summary at end is parseable (validate with DecisionSummary
       model).
  4. Commit the decision doc to `decisions/{research_id}_decision.md`.
  5. If frontmatter patch has `status: decision_made`, apply that to the main
     research_project note’s YAML (this can be done by opening the file and
     updating YAML, or storing these statuses in a separate index).
  6. Update issue: add label `stage: decision.review` (or
     `decision.review_requested` to trigger next agent) and move to "Review"
     column. Possibly also add a label or field with `decision_date` and
     `recommendation` for quick reference.
  7. Graph updates: create `decision` node (if not created earlier) with
     properties from the frontmatter/JSON (recommendation, etc.), link it to
     research_project (`has_decision`) and to any chosen solution concept
     (`implements`). Create edges for each option’s outcome: for each
     alternative in the JSON `alternatives_considered`, link a concept node for
     that option to the decision node with an edge like `alternative` (or simply
     store them as properties of decision for now). Also create `justifies`
     edges from any insights listed in the decision content to the decision (the
     orchestrator might need to identify references to findings/insights – if
     insights were logged as separate notes, they can be matched by content or
     IDs).

**On Event: decision.review_requested**

- When issue enters review stage:
  1. Invoke `decision_review.md` with the decision document (either pass the
     text or a path the agent can access).
  2. Agent returns a review report. Validate that it includes an Overall Status
     and a `review_result` keyword.
  3. Post the review as a comment or commit `reviews/{research_id}_review.md`.
     Possibly also update the decision file’s frontmatter with a field
     `review_status: Needs Revision` or similar.
  4. Parse the result:
     - If `review_result: APPROVED`:
       - Add label `status: decision_approved` and move issue to "Approved".
       - Proceed to trigger implementation: label
         `stage: implementation.started`.
     - If `NEEDS_REVISION` or `REQUIRES_REWORK`:
       - Add label `status: needs_revision`.
       - Create a task for revision: The orchestrator could either directly
         prompt the Decision Agent with the review feedback (“Here are the
         issues, please update the decision doc accordingly”) or notify the
         human to revise.
         - If automated: the orchestrator can supply the original decision
           content and the review report to Claude in a new prompt that asks for
           an updated decision doc addressing the points. This may be tricky but
           feasible for minor edits. For major rework, human input might be
           better.
       - Once revisions are made (either by agent or human editing the decision
         markdown), the orchestrator triggers another
         `decision.review_requested` event (could be automatic after agent
         revision, or when human indicates update done, say by re-adding the
         label).
       - Iterate until approved. Each iteration could be tracked (like count of
         reviews). The workflow file would specify a loop with a max number of
         cycles to avoid infinite loop. If exceeding, escalate to human.
  5. Graph: log the review results. Possibly create a `review` node with
     relations like `reviews` (the decision). Or at least update the decision
     node with a property `review_status` and maybe a pointer to the review
     file. The review feedback can also be stored as an `insight` if it contains
     knowledge (or simply for traceability).

**On Event: implementation.started**

- When issue labeled `implementation.started`:
  1. Gather input: the decision doc’s key bits (recommendation, success metrics)
     and any implementation context (team assigned, start date). Possibly the
     orchestrator knows the team from earlier fields, and start date = now.
  2. Invoke `implementation_started.md` with that info.
  3. Agent returns the Implementation Tracking doc. Validate that key sections
     are present (phases, success metrics, etc.).
  4. Commit to `implementation/{research_id}_implementation.md`.
  5. Update issue: move to "Implementation" column, label
     `status: implementation_in_progress`. Possibly attach key dates or progress
     fields in the project item (Projects v2 allows custom fields like “Target
     completion date” – set that from the plan).
  6. Orchestrator sets up monitoring:
     - Maybe schedule a GitHub Action to prompt for daily/weekly status if no
       updates.
     - Or subscribe to external metrics (if available via APIs) and post
       `metrics_update` events. For example, tie into CI/CD or monitoring
       dashboards through webhooks.
  7. Graph updates: create `implementation` node and link to the decision node
     (`implements_decision` or similar). Also link any metrics defined to metric
     nodes if modeling metrics separately.

**On Event: implementation.metrics_update**

- (These are ongoing events, not one-time.)
  - When a metrics YAML is posted (the orchestrator sees an issue comment or an
    incoming webhook with that structure): parse it into a `MetricsUpdate`
    object.
  - Update the issue comment with a confirmation if needed, or reflect it in a
    progress field (e.g., update a “Progress (%)” field on the project item to
    the `progress_percentage` value).
  - In the KG, record the metrics: either update properties of the
    implementation node (e.g. current values and statuses of metrics), or create
    a `metrics_update` node with edges to the implementation.
  - If any `blockers` with high severity appear, orchestrator could
    automatically tag the issue as `status: blocked` or notify the stakeholder
    via mention.
  - Check escalation triggers: if `status: at_risk` on a metric or a blocker
    with severity high, the orchestrator can escalate (perhaps ping the solo
    entrepreneur via Slack/email, or create a sub-issue for the blocker). This
    logic would be defined in the workflow as conditions.
  - Continue listening for updates until implementation completion.

**On Implementation Completion:**

- This might be indicated by a metrics update where `progress_percentage: 100`
  or an explicit comment, or reaching the end date. The orchestrator then marks
  the issue with `status: implementation_done` and possibly automatically
  schedules the ROI review (sets a due date or directly triggers if immediate).
- The orchestrator could add a label `stage: metrics.review_due` when it's time.

**On Event: metrics.review_due**

- When time for ROI review:
  1. Orchestrator collects data: the original success criteria (from proposal or
     decision doc), and actual metrics (from latest metrics updates or
     monitoring systems). Also gather any additional qualitative data (maybe
     prompt the user for team feedback or check incident logs). These are
     compiled into the input YAML as shown in template
     (`original_success_criteria`, `actual_metrics`, etc.).
  2. Invoke `metrics_review.md` with that input.
  3. Agent returns the ROI Analysis markdown. Validate that it has the key
     sections and an Overall Success rating.
  4. Commit to `analysis/{research_id}_roi.md`.
  5. Update issue: move to "Done/Completed" column, label `status: completed`.
     Possibly close the issue (with a closing comment summarizing results and
     linking the ROI doc).
  6. Graph updates:
     - Update the decision node with outcome: e.g. add `outcome: success`
       property, `roi: 120%`.
     - Add edges for any new insights learned (the ROI doc “Lessons Learned” can
       be mined for insights which we add as nodes and link to future guidance).
     - If ROI is low or partial, maybe create an edge like decision
       `invalidated_by` insight "Outcome below expectations". If ROI is high,
       maybe link the decision to a concept "Success Story" or simply mark it in
       properties.
     - If any **Future Research Indicators** in the report are checked (the
       template lists conditions like performance degrading, costs rising,
       etc.), the orchestrator could proactively open new `research.requested`
       issues for those. For example, if one condition was "Costs increased
       above threshold", and it happened, then that's essentially a trigger
       event outside this cycle.

**Memory Integration (MCP) and Context:** Throughout this orchestration,
maintaining context for the agents is crucial:

- The orchestrator uses a **Memory system** (potentially the “Model-Context
  Protocol (MCP)”) to preserve important info across prompts. For instance, the
  research plan agent might benefit from knowing similar past research plans;
  the decision agent might use insights from earlier in the conversation or
  related decisions.
- **Memory implementation:** We can use a vector store to embed and retrieve
  relevant notes from the vault. Before calling an agent, the orchestrator can
  query the KG for related items:
  - E.g., when generating a decision, fetch past decisions in the same category
    or any insights from this research’s plan (the orchestrator can supply the
    research plan text as additional context if needed).
  - When doing ROI analysis, bring back the original proposal’s expected values
    to compare.
- The MCP likely refers to a system where the orchestrator maintains a running
  context document that can be appended or updated with new facts, and that
  context is provided to agents. For instance, after each stage, the
  orchestrator could update an “memory state” (in JSON or a summary) and include
  that as part of the prompt for the next stage agent so it doesn't hallucinate
  or forget details.
- **Example:** After research proposal is made, orchestrator might keep a
  “project memory” like:
  ```json
  {
    "research_id": "MEM_003",
    "question": "Vector DB for 10M embeddings",
    "priority": "high",
    "selected_options": [],
    "findings": []
  }
  ```
````

And fill it as the process goes: findings get added, selected option gets set
once decided. This could be passed as part of the system prompt for later
agents.

- In practice, each agent prompt already includes necessary context explicitly
  (the outputs of previous stage as input), so the memory is mostly needed if we
  want the agent to recall previous reasoning or ensure consistency.
- The orchestrator can also store conversation logs or intermediate rationale
  (Chain-of-thought) in memory but likely not needed with this structured
  approach.
- **MCP and external tools:** If using something like the Orkes Conductor MCP or
  similar, it provides a standardized way for agents to call tools and carry
  state. Our design can integrate with such a system by treating each event as a
  state in a workflow and passing the data through MCP’s context object. The
  YAML frontmatters and JSON summaries we included are MCP-friendly (they can be
  used as the structured output that MCP expects to decide next steps).

### Quality Gates and Audit Hooks

Quality is enforced at multiple points:

- **Templates with Checklists:** Each agent template has a checklist the agent
  must tick mentally. The orchestrator can verify outputs against these (e.g.,
  search for `[ ]` in outputs to see if any checkbox remains unchecked or any
  placeholder text left). If something is missing (e.g. no recommendation in
  decision doc), the orchestrator can either prompt again or mark it for human
  review.
- **Peer Review stage:** This is an explicit quality gate. It ensures no
  decision goes to implementation without a second pass. The orchestrator does
  not override a negative peer review; it will loop until the issues are
  resolved. This guarantees only vetted decisions are implemented.
- **Human-in-the-loop:** At key points, the orchestrator can pause and request
  human approval. For example, after proposal or after decision or after ROI, it
  can tag the issue `awaiting human review` and not proceed until the solo
  entrepreneur gives a go (perhaps by adding a label `approved` or commenting
  "OK"). This can be configured by adding steps in the workflow file like:

  ```markdown
  if proposal.category == "STRATEGIC": pause for human approval before proceeding
  ```

  (Pseudocode) So high-impact items get manual oversight.
- **Audit trail:** All actions are logged in Git (each artifact is committed
  with history) and in the issue comments. This provides an audit trail of who
  (which agent/human) did what and when. The KG also stores relationships like
  `authored_by` with timestamps (we could include a `created_at` in YAML).
- **Error handling:** The workflow includes error-catching. If an agent fails to
  produce output (or produces something incoherent), the orchestrator can retry
  or escalate. For instance, if Pydantic validation fails for the proposal, the
  orchestrator might re-prompt: “The output is missing `estimated_completion`.
  Please include it.” Agents (especially Claude) can handle such instructions to
  correct format.
- **Security and governance:** Since GitHub is used, we can use branch
  protections or PR reviews for final approvals. E.g., the orchestrator could
  open a PR with the decision doc and require the solo entrepreneur to approve
  merging it. This way, no implementation happens without an explicit human
  sign-off (if desired).
- **Memory checks:** The orchestrator can also audit the KG for conflicts or
  duplicates at certain checkpoints (see deduplication below). E.g. before
  adding a new concept node, check if an alias exists.

### When to Spawn Swarm Agents vs Sequential Steps

Our framework allows both **sequential** and **parallel** (swarm) agent
operation:

- **Sequential Steps:** By default, the orchestrator handles events one after
  another in the order of the lifecycle. This is suitable when each step’s
  output is needed for the next (which is the case for intake → proposal → plan
  → decision, etc.). The orchestrator ensures one stage completes successfully
  (output validated) before moving forward.
- **Swarm/Parallel Agents:** During complex research or brainstorming phases,
  parallelism can speed up knowledge gathering:

  - _Research Phase:_ If the research question can be broken down (e.g.,
    evaluate multiple solutions, or research separate sub-questions like
    performance vs cost vs compliance), the orchestrator can spawn multiple
    research sub-agents simultaneously. Each would be given a focused task
    prompt. For example:

    - Agent A: “Benchmark performance of Option X with given criteria”
    - Agent B: “Analyze cost and scalability of Option X”
    - Agent C: “Research compliance considerations for Option X”
    - Similarly for Option Y... The orchestrator can use the same
      `research_proposed.md` template but with filters, or custom smaller
      prompts, or even use tool-based queries (e.g. use an agent to search
      literature or documentation). This **swarm** of agents might produce
      multiple insight notes or raw findings. After parallel tasks finish, the
      orchestrator aggregates the insights into the main findings summary. This
      swarm approach is triggered when tasks are independent. The workflow
      script might have a rule: if the proposal lists multiple alternatives or
      if `complexity` is high, spawn parallel agents for each alternative.
  - _Evaluation Phase:_ Another use of swarm could be to get multiple
    perspectives. For instance, after a decision doc is drafted, we could have
    two review agents with different specialties (one focusing on technical
    aspects, another on business impact) and combine their feedback for a more
    robust review. Currently we defined one Peer Review Agent with a broad
    checklist, but one could imagine modular reviews.
  - The orchestrator must manage the outputs merging: it may wait for all
    parallel agents to finish and then either combine results itself or prompt a
    synthesis agent. In code, this could mean asynchronous calls to multiple
    Claude instances or interleaving tool calls. In our design, it's conceptual
    (as actual parallel calls depend on infrastructure).
- **Resource Use:** Since this is a solo user’s system, one might not always
  want many parallel agents for cost reasons. The orchestrator can be configured
  to only use swarm mode for high urgency or high complexity issues, otherwise
  do sequentially (which might actually involve the solo entrepreneur doing some
  tasks manually).
- The **workflow file** can include logic like:

  ```markdown
  if event == research.proposed: if proposal.complexity == "high" or
  len(proposal.options) > 1: parallel_tasks = [ generate_research_findings(option)
  for option in proposal.options ] wait for all parallel_tasks compile findings
  else: conduct research sequentially
  ```

  (Pseudo-code, illustrating decision points.)

### Orchestrator Template Invocation and Validation

The orchestrator uses the templates by filling in the **Input** section from the
current context and sending the whole markdown (context, input, task, output
format, etc.) to Claude (or the chosen LLM) as a prompt. It likely uses the
few-shot examples in the templates to guide output style.

After getting the output from Claude, the orchestrator will:

- Parse out any YAML or JSON sections that need machine reading. (Using regex or
  markdown parsing libraries.)
- Validate them with Pydantic models (already discussed). If validation passes,
  move on. If not:

  - If minor (like a date format issue or missing field), the orchestrator can
    gently correct by instructing the agent to fix just that part (e.g. provide
    the missing field or correct format). This can be done by a follow-up prompt
    that includes the last output and asks for corrections in YAML only.
  - If the output is significantly off, orchestrator might retry the whole
    prompt or escalate to human if it repeatedly fails.
- For text sections, the orchestrator could run sanity checks (like ensure no
  section is empty or all placeholders are replaced). This can be part of
  validation logic.

All these steps (invocation, validation, file commit, issue update) are encoded
in the workflow or in code hooking into GitHub events.

### Memory (MCP) and Multi-turn Orchestration

Claude (and other LLMs) have token limits, so the orchestrator must be selective
in what context to provide at each step:

- Use the KG to fetch _only relevant_ context notes. For example, when drafting
  a decision, the orchestrator might fetch the research plan and any insights
  from execution and include them in the prompt (possibly in a condensed form).
  If using Anthropics’ Claude, which has large context, one could include quite
  a bit. But we should avoid overload; hence leveraging the structured summary
  as input.
- The orchestrator should also preserve the state of the conversation if needed.
  If Claude is powering the orchestrator itself (i.e., the orchestrator is an AI
  agent that reads these instructions and events), it might maintain a
  chain-of-thought in hidden memory. However, it's more reliable to use explicit
  data passing (via YAML/JSON) between steps as we designed.
- MCP (Model Context Protocol) typically allows storing and retrieving
  “memories” by keys. We can store the research\_id context and retrieve it at
  each event. E.g., after proposal is done, store a memory
  “MEM\_003\_proposal\_result” that can be fetched later. The workflow could
  have steps to `MCP.store(key, data)` and `MCP.retrieve(key)` if integrated.
- The **knowledge graph itself is a form of long-term memory.** The orchestrator
  can query it using GraphQL or Cypher to get info about related past cases. For
  example, before making a decision, it could run a similarity search: find past
  decisions where `category == MEM` and see their outcomes (maybe to caution if
  similar decisions had problems). Those could be injected as advice or context
  to the agent (“Note: Past research MEM\_001 tried Pinecone and had cost
  issues.”). This uses embeddings or direct queries.
- **Vector Embeddings:** As the KG scales, we can compute embeddings for content
  of proposals, decisions, etc. The orchestrator (if it has an embedding store)
  can retrieve similar content to help the agent. If not built-in, the user can
  manually recall relevant notes.

### Multi-Tenancy and Role-Based Extensions

Our design anticipates future growth:

- **Nested Vaults:** If the user later introduces multiple Obsidian vaults (say
  one per product or one for private vs public knowledge), the system can
  accommodate that. Each vault could have its own set of markdown files and even
  its own orchestrator instance or configuration.

  - We might include a `vault` field in YAML to denote which vault or project an
    entity belongs to.
  - The orchestrator can be configured with a context of one vault at a time or
    route events to different vault-specific agents. For example,
    `vault: alpha-product` vs `vault: beta-project` in the research frontmatter,
    and orchestrator ensures to save files in the correct vault folder or
    repository.
  - Relations across vaults can still exist (like a concept might be global
    across vaults), but one might also partition the KG by vault if needed for
    multi-tenant isolation. Supabase could have a tenant id or separate schemas
    per vault.
- **Neo4j-based indexing:** When the data outgrows lightweight storage, we can
  sync all YAML to a Neo4j database for advanced querying. We have kept the
  schema compatible. For instance, `type` in YAML can map to Neo4j labels, and
  we can easily run Cypher like
  `MATCH (d:decision)-[:IMPLEMENTS]->(m:method) RETURN ...`. We should ensure
  the ID format is consistent (maybe prefix IDs with vault or use GUIDs to avoid
  collisions across vaults).

  - We could also use Neo4j’s full-text search or Graph Data Science for finding
    similar nodes (embedding similarity).
  - The orchestrator could query Neo4j directly via Cypher when needing complex
    info (Supabase GraphQL is good for CRUD but complex pattern queries are
    easier in Cypher).
- **Multi-tenant agent roles:** In future, multiple users or teams might use the
  system, each with their own agents or preferences. The framework can handle
  this by:

  - Parameterizing the prompts or workflow by team. E.g., one team might have a
    different checklist or additional steps (maybe a security review event).
  - Running separate orchestrator instances per team repository or having
    conditional logic in one orchestrator: e.g., if an issue is in Repo A (Team
    A’s repo), use Team A’s agent prompts (perhaps stored in a folder `teamA/`).
  - Agents themselves might have multi-tenant configurations. For example, a
    Claude instance fine-tuned or instructed specifically for one domain.
  - The knowledge graph could also be multi-tenant, partitioned by organization,
    but with some shared nodes (for common concepts). We include a `tenant`
    property if needed.
- **Extensibility:** Adding a new stage or role is straightforward: create a new
  template and define its trigger in the workflow. For example, if later they
  want a `security.review` event for security team to review decisions, we add
  `security_review.md` and insert it as a required event before implementation
  for certain categories.
- **Nested workflows:** If vaults or projects have sub-workflows (like a
  research project might spawn a sub-project), the orchestrator can manage that
  via sub-issues or linking the workflows. The design is modular enough to start
  a new cycle mid-way (the follow-up research scenario).
- **Continuous Learning:** Over time, the orchestrator/agents could learn from
  past cycles. For example, the ROI analysis could feed into a knowledge base of
  what methods tend to yield high ROI. The orchestrator can incorporate that
  into future decision-making prompts (“In past, similar projects had ROI
  \~120%, consider if expected ROI is sufficient.”). This would likely be a
  manual tuning unless we incorporate an automated meta-learning, but the data
  will be there to leverage.

## Deduplication, Entity Merging, and KG Scaling Strategies

As the knowledge graph grows with many research cycles, we need strategies to
keep it clean, non-redundant, and performant:

- **Duplicate Entity Detection:** When new entities (especially Concept or
  Method nodes) are added from a research, the orchestrator (or a maintenance
  script) should check if an equivalent entity already exists.

  - Use **embeddings** of entity names/definitions: For example, if a proposal
    introduces a concept "Vector DB", we generate an embedding for that and
    compare with existing concept embeddings. If a cosine similarity is above a
    threshold or if the name literally matches an alias of an existing concept
    (e.g. "Vector Database"), flag it.
  - The orchestrator can then decide to merge rather than create a new node.
    This could mean linking the research directly to the existing concept.
    Perhaps in YAML, instead of making a new `concept_vector_database`, we tag
    it as related to existing concept.
  - Possibly have a manual review of new concepts periodically to merge
    duplicates.
- **Entity Merging & Aliasing:** For any duplicates that do get created (maybe
  agents might phrase things differently), we can merge them. Merging in
  Obsidian could be done by consolidating content into one note and updating all
  references.

  - The KG can maintain an **aliases list** for each concept: if two concept
    notes are realized to be the same, one can become the primary, and the other
    note can either be removed or turned into a stub that just points to the
    primary (with `alias_of: primary_concept_id` in YAML).
  - In Neo4j, one might just create an `ALIAS_OF` relationship and always query
    the primary. But it's often easier to have one node with multiple names.
  - Our YAML schema allows an `aliases` field for this. We should ensure when
    agents generate content, they check if any known alias exists. For instance,
    if an insight mentions "pgvector", the orchestrator could tag it to concept
    "PostgreSQL pgvector extension" because it knows via alias that they're the
    same.
- **ID Versioning:** Over time, certain concepts or methods may evolve (e.g. a
  new version of a technology).

  - We recommend giving each entity a stable ID and if it changes significantly,
    create a new entity and relate them (rather than reusing one node for two
    different things). For example, if "Algorithm A" was one method and later a
    totally new approach "Algorithm B" replaces it, they should be separate
    nodes, but maybe linked by `supersedes`.
  - For decisions, if a decision is revisited after some time (maybe a new
    research in 2026 revisits the 2025 decision), we create a new decision node
    and mark the old one as superseded. Each decision has a date, so we know
    which is current. This essentially versions decisions by time.
  - The research\_id already encodes a project sequence (ARC\_005 implies it’s
    the 5th in ARC category). If a topic is revisited, it might get a new ID. We
    might consider linking them in a series (like “this research is a follow-up
    to ARC\_005, making ARC\_007”, etc.).
  - Using timestamp or incremental IDs ensures unique keys. If wanting global
    uniqueness, maybe prefix with year or org (but our short codes should be
    fine as long as category+number is unique).
- **Temporal Graph Hygiene:** The KG should reflect that knowledge validity can
  change over time:

  - We keep timestamps on edges like `implemented_on`, `invalidated_on`.
  - If an insight invalidates a concept, maybe that concept gets a status
    “deprecated” in YAML.
  - If a decision is made, and later another insight invalidates it, we mark
    that decision node as outdated (could add `status: invalidated`
    frontmatter).
  - We might even remove certain edges when they expire (like if an
    implementation is rolled back, the `implements` edge from decision to method
    might be marked inactive). However, it's usually better to mark as
    historical rather than delete, to retain history.
  - The ROI analysis essentially tells us if the decision held up. If not, it
    might prompt the creation of an edge `invalidates` from a new insight to
    that decision or its underlying assumption. The orchestrator or human should
    maintain these.
  - Periodic audits (maybe annually) can be done to review if any concept or
    method nodes are no longer relevant (e.g., technology no longer used). They
    could be tagged or archived.
- **Scaling using Embeddings and Graph Queries:** As number of notes grows,
  finding relevant information for new research might be like finding a needle
  in haystack. By storing vector embeddings for each note (taking perhaps the
  summary or key fields), we can do semantic search.

  - For example, when a new research request comes in, we vectorize its
    description and find the top 5 similar past research projects by embedding
    similarity. The orchestrator can then automatically suggest if similar
    research was done before (“Similar past research: MEM\_002 had a related
    question about vector DBs”). This prevents duplicate efforts and allows
    building on prior work.
  - The orchestrator could incorporate those links into the proposal (maybe
    auto-fill “Similar Research” field, as we had in the research plan
    template). We can leverage GPT/Claude to compare the new request to past
    ones too.
- **Graph Pruning and Partitioning:** If certain parts of the KG become very
  large or irrelevant, we might partition by date or archive old projects in a
  separate vault. For a solo entrepreneur, this might not be an issue for a
  while, but it's good practice.

  - Could have an "Archived" vault or just mark nodes as archived (and
    orchestrator tools by default search only active ones unless asked).
  - Using Neo4j, we can easily exclude archived nodes in queries via a property.
- **Performance Considerations:** Supabase (Postgres) might start to strain if
  we store every metric update as a row. We should aggregate or sample if
  updates are very frequent. However, for moderate usage, it’s fine.

  - We could store metrics in a time-series DB or leave them in markdown (the
    metrics updates YAML could be appended to the implementation note perhaps,
    though that could get long).
  - Neo4j can handle many nodes/edges, but if things grow huge, we might move
    some less critical detail (like every daily metric) out of the main KG after
    some time.

In summary, the orchestrator should incorporate a **deduplication check**
whenever new nodes are about to be created (especially concept, method,
stakeholder). And possibly have a background job or command to merge nodes.
Embeddings and alias lists are key tools for this. Maintaining clean data
ensures the agents also perform better (no confusion by duplicate concepts).
