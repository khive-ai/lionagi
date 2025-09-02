### 1 Event‑Driven Knowledge‑Graph Lifecycle

| Stage          | Canonical **event** (payload key)                         | Typical _agent roles_                  | Primary GH Issue **label** | GH Project (view/column) | Purpose                                                       |
| -------------- | --------------------------------------------------------- | -------------------------------------- | -------------------------- | ------------------------ | ------------------------------------------------------------- |
| Intake         | `research.requested`                                      | _Intake Agent_ (classifies, scopes)    | `event:request`            | **Inbox**                | Capture raw questions and turn them into structured proposals |
| Planning       | `research.proposed`                                       | _Planner_ (defines scope / milestones) | `event:propose`            | **Planning**             | Produce an executable research plan                           |
| Execution      | `research.plan.accepted` → work subtasks                  | _Researcher / Synthesizer_             | `event:execute`            | **In‑Progress**          | Run experiments, gather evidence                              |
| Synthesis      | `decision.ready`                                          | _Decision Agent_                       | `event:decision`           | **Decision Draft**       | Turn evidence into a decision doc                             |
| Peer review    | `decision.review_requested`                               | _Reviewer_                             | `event:review`             | **Review**               | Quality gate before approval                                  |
| Approval       | `decision.approved` (auto‑emits `implementation.started`) | _Governance Bot_                       | `status:approved`          | **Approved**             | Fires implementation queue                                    |
| Implementation | `implementation.started`                                  | _Implementer / Tracker_                | `event:implement`          | **Implementation**       | Deliver solution, track metrics                               |
| Metric audit   | `metrics.review_due`                                      | _ROI Analyst_                          | `event:metrics`            | **ROI Review**           | Compare outcomes vs. prediction                               |
| Graph update   | `knowledge.graph.update`                                  | _KG Maintainer_                        | `event:kg-update`          | **Graph Ops**            | Merge new entities/edges; dedup                               |
| Archive        | `cycle.complete`                                          | _Archivist_                            | `status:closed`            | **Done**                 | Freeze docs, move to cold storage                             |

> **Flow:** GitHub Issues are the **events**; moving a card (Project v2 workflow
> columns above) or applying a label triggers a GitHub Action that sends the
> payload to the next agent. Each agent writes its output into a markdown
> document stored in the repo; the Issue body is only the _pointer_ and JSON
> payload.

---

### 2 Using GitHub Issues/Projects as the Event Queue

1. **One‑Issue‑per‑Event**

   - Title convention: `[EVENT] short‑slug` (e.g.
     `[research.requested] Vector DB Options`).
   - Body = YAML payload wrapped in `yaml …` for machine parsing.
   - Labels:

     - `event:<name>` – routing key.
     - `cycle:<research_id>` – links all issues in one research thread.
     - `priority:high|med|low`, `status:*` – human filters.

2. **Project v2 automation**

   - Columns mirror the stage table above.
   - A GitHub Action listens for `issues.labeled` and moves card to the matching
     column; status transitions happen **only** by board movement
     (single‑source‑of‑truth).
   - Additional Action: when a card enters **In‑Progress**, create child tasks
     via the GraphQL API (one per experiment or implementation phase); they
     inherit the parent labels.

3. **Agent invocation**

   - Each agent watches the queue with the GitHub REST API filter
     `label:event:<stage>` + `assignee:agent‑bot‑name`.
   - Agent clones repo, reads payload, writes deliverable markdown into vault
     (`01 Projects/…`) using the templates below, commits on a branch
     `agent/<event>/<id>`, and opens a PR that auto‑links back to the Issue
     (`Fixes #123`).
   - Merge of the PR closes the Issue **and** emits the next event (Action posts
     a new Issue with the next‑stage label and payload extracted from the merged
     doc’s front‑matter).

---

### 3 Templates · Deliverables · Quality Gates

| Stage / Event               | **Document template** (markdown file)           | Mandatory _front‑matter_ keys                            | Quality gate (blocking)                            |
| --------------------------- | ----------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------- |
| `research.requested`        | _research\_request.md_ (Issue body only)        | `description, requestor, urgency`                        | Auto‑schema lint; complete fields                  |
| `research.proposed`         | `research_plan_<id>.md`                         | `research_id, question, success_criteria, timeline`      | Planner peer check; CI validates JSON schema       |
| `decision.ready`            | `decision_<id>.md`                              | `status:decision_made, recommendation, confidence_level` | 10‑point QA review must score ≥ 80                 |
| `decision.review_requested` | PR Review comment generated from _decision_ doc | `review_result, overall_score`                           | All ❌ items resolved; overall ≥ 80                |
| `implementation.started`    | `impl_track_<id>.md`                            | `team, timeline, success_metrics[]`                      | Kick‑off sign‑off; monitoring endpoints live       |
| `metrics.review_due`        | `roi_<id>.md`                                   | `actual_metrics, roi`                                    | Exec sponsor approval; lessons learned logged      |
| `knowledge.graph.update`    | `kg_patch_<timestamp>.md`                       | `entities_added, edges_added, dedup_actions`             | No duplicate IDs; link validation passes           |
| `cycle.complete`            | `archive_<id>.md`                               | `closed: true, archive_date`                             | CI check: all child issues closed, all docs linked |

CI (GitHub Actions) runs:

- **YAML schema validate** → fails status.
- **Dataview link integrity** (broken wikilinks).
- **Spell & style lint** for decision docs.
- **Embeddings‑based dup check** for entity IDs.

---

### 4 Entity & Relation Optimisation Guidelines

| Topic                         | Best‑practice                                                                                                                                               |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Canonical IDs**             | `TYPE_YYYYMMDD_NNN` (stable, sortable). Use a UID generator in Templater.                                                                                   |
| **Typed links**               | Always record relation in YAML `edges:`; body wikilink alone is insufficient.                                                                               |
| **Edge weights & provenance** | Store `weight` 0‑1 and `evidence:` field; required for automated pruning/graph algorithms.                                                                  |
| **Deduplication**             | Nightly job builds Sentence‑Transformer embeddings for `title + summary`; cosine > 0.93 triggers `potential_duplicate` label and writes a `kg_patch` event. |
| **Merge policy**              | _Soft merge_ = alias list in front‑matter; _hard merge_ = redirect stub note with `redirect_to:`.                                                           |
| **Attribute normalisation**   | Enumerate controlled vocabularies (`status`, `priority`, `relation`); enforce via JSON Schema.                                                              |
| **Graph hygiene**             | Orphan detector query (`WHERE length(file.inlinks)=0 AND length(file.outlinks)=0`) surfaces stale nodes weekly.                                             |
| **Performance**               | Keep each note <10 KB; archive closed cycles to `_archive/` and exclude from Juggl global graph.                                                            |
| **Versioning**                | Each entity note keeps `version:`; major schema change emits `kg.schema_changed` event so agents can migrate references.                                    |

---

### 5 Agent–Document Handoff Protocol

```yaml
# in every deliverable front‑matter
hand_off:
  from_agent: "research.bot"
  to_agent: "decision.bot"
  event_next: "decision.ready"
  due: "2025-07-03"
```

- **Producer agent** sets `to_agent` and opens a GitHub Issue tagged with
  `assignee:@to_agent` plus the path to the newly created document.
- **Consumer agent** starts only when assigned, reads the doc, and on completion
  **must**:

  1. Update `status` in the same doc.
  2. Commit patch, close issue, spawn next event.

This guarantees every state transition is explicit, reviewable, and
audit‑friendly.

---

#### Quick‑start Checklist

1. **Create** GitHub Project columns as per §1.
2. **Add** the six provided markdown templates to `_templates/`.
3. **Install** Actions: `ci-yaml-lint.yml`, `ci-kg-dup.yml`,
   `ci-project-move.yml`.
4. **Register** agent bots with repo access tokens; subscribe them to their
   `event:` labels.
5. **Run** first intake: open Issue `[research.requested] …` with YAML payload.

> Once this scaffolding is in place, your multi‑agent system will flow each
> research / implementation cycle through a **single event queue**—GitHub
> Issues—while the Obsidian vault remains the durable, queryable knowledge graph
> underneath.
