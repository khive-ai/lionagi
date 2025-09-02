# GitHub Event-Driven Workflow

## Overview

This document defines how the KB lifecycle is orchestrated through GitHub issues
and events, with agents communicating via comments and labels.

## GitHub Setup

### Repository Configuration

```yaml
# .github/kb-lifecycle.yml
labels:
  # Stage labels (mutually exclusive)
  stages:
    - name: "stage:research.requested"
      color: "0052CC"
      description: "New research request awaiting intake"
    - name: "stage:research.proposed"
      color: "5319E7"
      description: "Proposal created, awaiting planning"
    - name: "stage:research.active"
      color: "FBCA04"
      description: "Research in progress"
    - name: "stage:decision.ready"
      color: "B60205"
      description: "Research complete, decision needed"
    - name: "stage:decision.review"
      color: "E99695"
      description: "Decision under review"
    - name: "stage:implementation.approved"
      color: "0E8A16"
      description: "Ready for implementation"
    - name: "stage:implementation.active"
      color: "1D76DB"
      description: "Implementation in progress"
    - name: "stage:metrics.review"
      color: "C2E0C6"
      description: "ROI analysis phase"

  # Category labels
  categories:
    - name: "category:AIO"
      description: "AI/Orchestration"
    - name: "category:MEM"
      description: "Memory/Persistence"
    - name: "category:TLI"
      description: "Tools/Integration"
    - name: "category:ARC"
      description: "Architecture"
    - name: "category:DEV"
      description: "Developer Experience"
    - name: "category:UXP"
      description: "User Experience"

  # Priority labels
  priorities:
    - name: "priority:critical"
      color: "B60205"
    - name: "priority:high"
      color: "FF9800"
    - name: "priority:medium"
      color: "FBCA04"
    - name: "priority:low"
      color: "CEE0C6"

  # Status labels
  status:
    - name: "status:blocked"
      color: "D93F0B"
    - name: "status:needs-revision"
      color: "F9D0C4"
    - name: "status:ready"
      color: "0E8A16"
    - name: "status:in-progress"
      color: "1D76DB"

# Projects configuration
projects:
  kb_lifecycle:
    columns:
      - "📥 Intake"
      - "📋 Planning"
      - "🔬 Research"
      - "🤔 Decision"
      - "✅ Review"
      - "🚀 Implementation"
      - "📊 Analysis"
      - "✨ Complete"
```

### Agent Communication Protocol

```yaml
# Agent signature format
agent_signatures:
  pattern: "[${AGENT_ID}-${TIMESTAMP}]"
  examples:
    - "[INTAKE-2024-01-15T10:30:00Z]"
    - "[CRITIC-2024-01-15T10:32:15Z]"
    - "[ORCHESTRATOR-2024-01-15T10:00:00Z]"

# Comment structure
comment_template: |
  ${AGENT_SIGNATURE} ${ACTION_TYPE}

  ## Status: ${STATUS_EMOJI} ${STATUS}

  ### Results
  ${RESULTS}

  ### Next Actions
  ${NEXT_ACTIONS}

  ### Inter-Agent Messages
  ${AGENT_MESSAGES}

  ---
  Execution time: ${DURATION} | Confidence: ${CONFIDENCE}
```

## Event Processing System

### Event Scanner Script

```python
#!/usr/bin/env python3
"""
KB Lifecycle Event Scanner
Identifies all actionable events and their parallelization potential
"""

async def scan_events():
    """Scan for all processable events"""
    
    events = {
        "parallelizable": [],
        "sequential": [],
        "blocked": []
    }
    
    # Get all open issues
    issues = await gh.get_issues(state="open")
    
    for issue in issues:
        event = identify_event(issue)
        
        if event:
            if is_blocked(event):
                events["blocked"].append(event)
            elif can_parallelize(event):
                events["parallelizable"].append(event)
            else:
                events["sequential"].append(event)
    
    return events

def identify_event(issue):
    """Determine event type from issue state"""
    
    labels = issue.labels
    stage = get_stage_label(labels)
    
    EVENT_MAP = {
        "stage:research.requested": {
            "type": "intake_needed",
            "swarm": "kb-intake-swarm",
            "parallelizable": True,
            "dependencies": []
        },
        "stage:research.proposed": {
            "type": "planning_needed",
            "swarm": "kb-planning-swarm",
            "parallelizable": True,
            "dependencies": ["proposal_approved"]
        },
        "stage:decision.ready": {
            "type": "decision_needed",
            "swarm": "kb-decision-swarm",
            "parallelizable": False,
            "dependencies": ["research_complete"]
        },
        "stage:decision.review": {
            "type": "review_in_progress",
            "swarm": "kb-review-swarm",
            "parallelizable": False,
            "dependencies": ["decision_drafted"]
        }
    }
    
    if stage in EVENT_MAP:
        event = EVENT_MAP[stage].copy()
        event["issue"] = issue
        event["research_id"] = extract_research_id(issue)
        return event
    
    return None

def can_parallelize(event):
    """Check if event can be processed in parallel"""
    
    # Check dependencies
    if not all_dependencies_met(event):
        return False
    
    # Check resource conflicts
    if has_resource_conflict(event):
        return False
    
    # Check event type
    return event.get("parallelizable", False)
```

### Event Execution Framework

```python
class KBEventOrchestrator:
    """Main orchestrator for KB lifecycle events"""
    
    def __init__(self):
        self.agent_registry = {}
        self.active_swarms = {}
        self.event_queue = asyncio.Queue()
        
    async def initialize(self):
        """Initialize orchestrator with agent check"""
        
        # Load orchestrator instructions
        self.instructions = await Read(".claude/CLAUDE.md")
        
        # Register available agents
        await self.register_agents()
        
        # Start event monitoring
        asyncio.create_task(self.event_monitor())
        
    async def register_agents(self):
        """Register all available agents"""
        
        agent_specs = await Glob(".claude/resources/agents/*.md")
        
        for spec_path in agent_specs:
            agent_id = extract_agent_id(spec_path)
            self.agent_registry[agent_id] = {
                "spec_path": spec_path,
                "status": "ready",
                "current_task": None
            }
    
    async def event_monitor(self):
        """Continuous event monitoring"""
        
        while True:
            # Scan for events
            events = await scan_events()
            
            # Process parallelizable events
            parallel_tasks = []
            for event in events["parallelizable"]:
                task = asyncio.create_task(
                    self.process_event(event)
                )
                parallel_tasks.append(task)
            
            # Wait for parallel completion
            if parallel_tasks:
                await asyncio.gather(*parallel_tasks)
            
            # Process sequential events
            for event in events["sequential"]:
                await self.process_event(event)
            
            # Check blocked events
            for event in events["blocked"]:
                await self.check_unblock(event)
            
            # Brief pause before next scan
            await asyncio.sleep(30)
    
    async def process_event(self, event):
        """Process a single event"""
        
        issue = event["issue"]
        swarm_name = event["swarm"]
        
        # Create swarm execution context
        context = {
            "issue": issue,
            "research_id": event["research_id"],
            "event_type": event["type"]
        }
        
        # Post orchestrator start comment
        await self.post_comment(issue, f"""
[ORCHESTRATOR-{timestamp}] Starting Event Processing

## Event: {event['type']}
## Swarm: {swarm_name}
## Research ID: {event['research_id']}

### Initializing Agents
{self.format_swarm_agents(swarm_name)}

---
*Orchestration beginning*
        """)
        
        # Execute swarm
        try:
            results = await self.execute_swarm(swarm_name, context)
            await self.handle_success(event, results)
        except Exception as e:
            await self.handle_failure(event, e)
    
    async def execute_swarm(self, swarm_name, context):
        """Execute a specific swarm pattern"""
        
        # Load swarm definition
        swarm_def = await Read(f".claude/commands/swarm/{swarm_name}.md")
        swarm_config = parse_swarm_config(swarm_def)
        
        # Track swarm execution
        swarm_id = f"{swarm_name}-{context['issue'].number}"
        self.active_swarms[swarm_id] = {
            "status": "running",
            "agents": [],
            "start_time": datetime.now()
        }
        
        # Execute phases
        phase_results = {}
        for phase in swarm_config.phases:
            if await self.check_phase_gate(phase, phase_results):
                results = await self.execute_phase(phase, context)
                phase_results[phase.name] = results
                
                # Run critic validation
                await self.run_critic_validation(phase.name, results)
        
        # Final consolidation
        final_results = await self.consolidate_results(phase_results)
        
        # Cleanup
        del self.active_swarms[swarm_id]
        
        return final_results
    
    async def execute_phase(self, phase, context):
        """Execute a single phase"""
        
        if phase.type == "parallel":
            # Parallel execution
            tasks = []
            for agent_id in phase.agents:
                task = self.run_agent(agent_id, context)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return dict(zip(phase.agents, results))
            
        else:
            # Sequential execution
            results = {}
            for agent_id in phase.agents:
                result = await self.run_agent(agent_id, context)
                results[agent_id] = result
                context.update(result)  # Pass results forward
            
            return results
    
    async def run_agent(self, agent_id, context):
        """Run a single agent"""
        
        # Load agent specification
        agent_spec = await Read(self.agent_registry[agent_id]["spec_path"])
        
        # Create agent task prompt
        prompt = f"""
Please execute the following agent task:

1. First, read and internalize your agent specification
2. Then execute your designated task
3. Format outputs according to your specification
4. Sign all GitHub comments with your agent signature

Agent Specification:
{agent_spec}

Current Context:
{yaml.dump(context)}

Remember to:
- Follow your defined capabilities and patterns
- Validate your outputs before returning
- Communicate with other agents via GitHub comments
- Mark your agent ID in all outputs
"""
        
        # Execute via Task tool
        result = await Task(prompt)
        
        return result
    
    async def run_critic_validation(self, phase_name, results):
        """Run critic agent validation"""
        
        critic_prompt = f"""
Execute critic validation for phase: {phase_name}

Review all agent outputs and post a validation report as a GitHub comment.
Check for errors, inconsistencies, and quality issues.

Phase Results:
{yaml.dump(results)}
"""
        
        await self.run_agent("critic_agent", {
            "phase": phase_name,
            "results": results
        })
```

### Orchestrator Completion Check

```python
async def check_completion(self):
    """Orchestrator must run this before declaring completion"""
    
    # Check for pending events
    pending = await scan_events()
    
    if pending["parallelizable"] or pending["sequential"]:
        raise Exception(f"""
        Cannot complete - pending events found:
        - Parallelizable: {len(pending["parallelizable"])}
        - Sequential: {len(pending["sequential"])}
        - Blocked: {len(pending["blocked"])}
        
        Must process all events before completion.
        """)
    
    # Check for incomplete swarms
    if self.active_swarms:
        raise Exception(f"""
        Cannot complete - active swarms running:
        {list(self.active_swarms.keys())}
        """)
    
    # Validate all issues in correct end state
    issues = await gh.get_issues(state="open")
    for issue in issues:
        if not is_terminal_state(issue):
            raise Exception(f"""
            Issue #{issue.number} not in terminal state
            Current: {get_stage_label(issue.labels)}
            """)
    
    return True
```

## QA and Consolidation

### Consolidation Agent Pattern

```python
class ConsolidationAgent:
    """Consolidates outputs from multiple agents"""
    
    async def consolidate(self, phase_results):
        """Merge and validate phase results"""
        
        consolidated = {
            "summary": {},
            "conflicts": [],
            "gaps": [],
            "actions": []
        }
        
        # Merge results
        for agent_id, result in phase_results.items():
            self.merge_into(consolidated, result)
        
        # Identify conflicts
        conflicts = self.find_conflicts(phase_results)
        consolidated["conflicts"] = conflicts
        
        # Identify gaps
        gaps = self.find_gaps(phase_results)
        consolidated["gaps"] = gaps
        
        # Generate actions
        actions = self.generate_actions(consolidated)
        consolidated["actions"] = actions
        
        # Post consolidation report
        await self.post_consolidation_report(consolidated)
        
        return consolidated
```

## Long-Running Worker Pattern

```yaml
# For long-running research or implementation
long_running_pattern:
  initialization:
    - Create dedicated branch
    - Open draft PR for communication
    - Set up progress tracking issue

  execution:
    - Regular checkpoint commits
    - Progress updates via PR comments
    - Incremental deliverables

  communication:
    - Daily status in PR
    - Blocker alerts in issue
    - Results preview in PR description

  completion:
    - Final validation
    - Merge to main
    - Update tracking issue
```

## GitHub Action Triggers

```yaml
# .github/workflows/kb-lifecycle.yml
name: KB Lifecycle Automation

on:
  issues:
    types: [opened, labeled, unlabeled]
  issue_comment:
    types: [created]
  pull_request:
    types: [opened, synchronize]

jobs:
  process_event:
    runs-on: ubuntu-latest
    steps:
      - name: Check event type
        id: event_check
        run: |
          # Determine if this is a KB lifecycle event

      - name: Trigger orchestrator
        if: steps.event_check.outputs.is_lifecycle_event == 'true'
        run: |
          # Notify orchestrator of event
```
