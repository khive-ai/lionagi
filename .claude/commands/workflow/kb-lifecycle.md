# KB Lifecycle Workflow

## Overview

Master workflow orchestrating the complete knowledge base lifecycle from
research request through ROI analysis, using GitHub-driven event processing.

## Prerequisites

```yaml
requirements:
  github:
    - Repository with issue tracking enabled
    - Labels configured per github-event-workflow.md
    - Projects board for lifecycle tracking

  orchestrator:
    - Access to all agent specifications
    - GitHub API permissions
    - Memory system (MCP) access
    - Knowledge MCP for structured accumulation
    - Knowledge graph connectivity

  agents:
    - All agents in .claude/resources/agents/
    - Swarm patterns in .claude/commands/swarm/
    - Validation rules loaded
```

## Workflow Configuration

```yaml
workflow:
  name: kb_lifecycle_orchestrator
  type: event_driven
  mode: continuous

  initialization:
    - Load CLAUDE.md instructions
    - Register all agents
    - Connect to GitHub
    - Initialize memory systems
    - Start event monitor

  event_sources:
    - GitHub issues (labels, comments)
    - GitHub PRs (long-running work)
    - Manual triggers
    - Scheduled reviews

  termination:
    - No pending events
    - All swarms completed
    - Manual stop command
```

## Main Orchestration Loop

```python
async def kb_lifecycle_orchestrator():
    """
    Main orchestrator loop for KB lifecycle
    THIS IS THE ENTRY POINT
    """
    
    # Initialize
    print("[ORCHESTRATOR] Initializing KB Lifecycle Orchestrator")
    await initialize_systems()
    
    # Main event loop
    while True:
        try:
            # CRITICAL: Scan for all events
            events = await scan_kb_events()
            
            # Process parallelizable events
            if events['parallelizable']:
                print(f"[ORCHESTRATOR] Processing {len(events['parallelizable'])} parallel events")
                await process_parallel_events(events['parallelizable'])
            
            # Process sequential events
            if events['sequential']:
                print(f"[ORCHESTRATOR] Processing {len(events['sequential'])} sequential events")
                await process_sequential_events(events['sequential'])
            
            # Check blocked events
            if events['blocked']:
                print(f"[ORCHESTRATOR] Monitoring {len(events['blocked'])} blocked events")
                await monitor_blocked_events(events['blocked'])
            
            # Run periodic maintenance
            await run_maintenance()
            
            # Check termination conditions
            if await should_terminate():
                break
                
            # Brief pause
            await asyncio.sleep(30)
            
        except Exception as e:
            await handle_orchestrator_error(e)
    
    # Cleanup
    await cleanup_orchestrator()
```

## Event Processing Functions

### Parallel Event Processing

```python
async def process_parallel_events(events):
    """Process multiple events in parallel"""
    
    tasks = []
    for event in events:
        # Create task for each event
        task = asyncio.create_task(
            process_single_event(event)
        )
        tasks.append(task)
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle results
    for event, result in zip(events, results):
        if isinstance(result, Exception):
            await handle_event_error(event, result)
        else:
            await record_event_success(event, result)
```

### Sequential Event Processing

```python
async def process_sequential_events(events):
    """Process events one at a time in order"""
    
    for event in events:
        try:
            result = await process_single_event(event)
            await record_event_success(event, result)
        except Exception as e:
            await handle_event_error(event, e)
            # Decide whether to continue or stop
            if is_critical_error(e):
                break
```

### Single Event Processing

```python
async def process_single_event(event):
    """Process a single lifecycle event"""
    
    issue = event['issue']
    event_type = event['type']
    
    # Map event to swarm
    swarm_map = {
        'intake_needed': 'kb-intake-swarm',
        'planning_needed': 'kb-planning-swarm',
        'decision_needed': 'kb-decision-swarm',
        'review_in_progress': 'kb-decision-review-swarm',
        'implementation_tracking': 'kb-implementation-swarm',
        'roi_analysis_due': 'kb-roi-swarm'
    }
    
    swarm_name = swarm_map.get(event_type)
    if not swarm_name:
        raise ValueError(f"Unknown event type: {event_type}")
    
    # Post start comment
    await post_github_comment(issue, f"""
[ORCHESTRATOR-{timestamp}] Processing Event

**Event Type**: {event_type}
**Swarm**: {swarm_name}
**Research ID**: {event.get('research_id', 'TBD')}

Starting swarm execution...
    """)
    
    # Execute swarm
    swarm_config = await load_swarm_config(swarm_name)
    result = await execute_swarm(swarm_config, event)
    
    # Post completion
    await post_github_comment(issue, f"""
[ORCHESTRATOR-{timestamp}] Event Processed

**Status**: ✅ Complete
**Duration**: {result['duration']}
**Next Stage**: {result.get('next_stage', 'None')}

{format_summary(result)}
    """)
    
    return result
```

## Swarm Execution Engine

```python
async def execute_swarm(swarm_config, event):
    """Execute a swarm pattern with all phases"""
    
    context = {
        'issue': event['issue'],
        'research_id': event.get('research_id'),
        'event_type': event['type'],
        'swarm_start': datetime.now()
    }
    
    phase_results = {}
    
    # Execute each phase
    for phase in swarm_config['phases']:
        # Check phase gate
        if not await check_phase_gate(phase, phase_results):
            continue
            
        # Execute phase
        print(f"[ORCHESTRATOR] Executing phase: {phase['name']}")
        
        if phase['type'] == 'parallel':
            results = await execute_parallel_phase(phase, context)
        else:
            results = await execute_sequential_phase(phase, context)
        
        phase_results[phase['name']] = results
        
        # Run critic validation
        await run_critic_validation(phase['name'], results, event['issue'])
        
        # Update context for next phase
        context.update(extract_key_results(results))
    
    # Consolidate results
    final_results = await consolidate_swarm_results(phase_results)
    
    # Execute output actions
    await execute_output_actions(swarm_config, final_results, event)
    
    return final_results
```

## Agent Execution

````python
async def execute_agent(agent_id, task_context):
    """Execute a single agent with proper initialization"""
    
    # Build agent prompt
    agent_prompt = f"""
You are being invoked as a specialized agent in the KB lifecycle.

CRITICAL INSTRUCTIONS:
1. First, read your agent specification from: .claude/resources/agents/{agent_id}.md
2. Fully understand your role, capabilities, and output requirements
3. Execute the task according to your specification
4. Sign all outputs with [{agent_id.upper()}-{{timestamp}}]
5. Post results to GitHub issue #{task_context['issue_number']}

Task Context:
```yaml
{yaml.dump(task_context)}
````

Additional Requirements:

- Stay strictly within your defined capabilities
- Use your specialized decision logic
- Format outputs according to your schema
- Validate outputs before returning
- Communicate with other agents via GitHub comments using the protocol

Begin by reading your specification, then execute the task. """

    # Execute via Task tool
    result = await Task(agent_prompt)

    # Validate agent output
    await validate_agent_output(agent_id, result)

    return result

````
## Inter-Agent Communication Handler

```python
async def monitor_agent_communications(issue_number):
    """Monitor and route inter-agent messages"""
    
    # Get recent comments
    comments = await get_issue_comments(issue_number, since=last_check)
    
    for comment in comments:
        # Parse agent messages
        if is_agent_message(comment):
            message = parse_agent_message(comment)
            
            # Route to target agent if specified
            if message.get('target_agent'):
                await route_to_agent(message)
            
            # Handle requests
            if message['type'] == 'REQUEST':
                await queue_agent_request(message)
            
            # Handle alerts
            elif message['type'] == 'ALERT':
                await handle_agent_alert(message)
````

## Quality Assurance Integration

```python
async def run_critic_validation(phase_name, results, issue):
    """Run critic agent validation after each phase"""
    
    critic_context = {
        'phase_name': phase_name,
        'phase_results': results,
        'issue_number': issue.number,
        'validation_type': 'phase_completion'
    }
    
    # Execute critic agent
    validation_result = await execute_agent('critic_agent', critic_context)
    
    # Handle validation outcome
    if validation_result.get('critical_issues'):
        await handle_critical_issues(validation_result['critical_issues'])
    
    return validation_result
```

## Event Scanner Integration

```python
async def scan_kb_events():
    """
    CRITICAL: This must be called before any orchestrator can declare completion
    """
    
    # Get all open issues
    issues = await gh_api.get_issues(state='open')
    
    events = {
        'parallelizable': [],
        'sequential': [],
        'blocked': []
    }
    
    for issue in issues:
        # Extract stage from labels
        stage = get_stage_label(issue.labels)
        
        # Determine event type and parallelizability
        event_info = determine_event_info(stage, issue)
        
        if event_info:
            # Check dependencies
            if await dependencies_met(event_info):
                if event_info['parallelizable']:
                    events['parallelizable'].append(event_info)
                else:
                    events['sequential'].append(event_info)
            else:
                events['blocked'].append(event_info)
    
    return events
```

## Completion Check

```python
async def orchestrator_completion_check():
    """
    MANDATORY: Must be called before declaring any task complete
    """
    
    # Scan for pending events
    events = await scan_kb_events()
    
    # Check each category
    if events['parallelizable']:
        raise OrchestratorIncompleteError(
            f"Cannot complete: {len(events['parallelizable'])} parallel events pending"
        )
    
    if events['sequential']:
        raise OrchestratorIncompleteError(
            f"Cannot complete: {len(events['sequential'])} sequential events pending"
        )
    
    # Check active swarms
    if active_swarms:
        raise OrchestratorIncompleteError(
            f"Cannot complete: {len(active_swarms)} swarms still running"
        )
    
    # All clear
    return True
```

## Usage Examples

```bash
# Start the orchestrator
Task: "Execute KB lifecycle orchestrator
      Monitor all GitHub issues and process events
      Reference: .claude/commands/workflow/kb-lifecycle.md"

# Process specific issue
Task: "Process KB lifecycle event for issue #123
      Determine appropriate swarm and execute
      Reference: .claude/commands/workflow/kb-lifecycle.md"

# Run completion check
Task: "Run orchestrator completion check
      Verify no pending events before stopping
      Reference: .claude/commands/workflow/kb-lifecycle.md"
```

## Error Handling

```yaml
error_strategies:
  agent_failure:
    - Log error with full context
    - Post error to GitHub issue
    - Attempt retry with reduced scope
    - Escalate to human if critical

  swarm_timeout:
    - Save partial results
    - Post timeout notification
    - Mark issue as blocked
    - Schedule retry

  communication_failure:
    - Queue messages for retry
    - Use alternative communication
    - Consolidate messages
    - Alert orchestrator
```

## Monitoring Dashboard

```markdown
## KB Lifecycle Status

### Active Cycles

| Research ID | Stage           | Duration | Status         | Next Action        |
| ----------- | --------------- | -------- | -------------- | ------------------ |
| MEM_004     | Decision Review | 2h 15m   | 🔄 In Progress | Awaiting revision  |
| ARC_007     | Research        | 45m      | ✅ On Track    | Data collection    |
| TLI_002     | Implementation  | 3d       | ⚠️ At Risk     | Blocker resolution |

### Event Queue

- **Parallelizable**: 3 events ready
- **Sequential**: 1 event waiting
- **Blocked**: 2 events pending dependencies

### Agent Status

- **Active**: 5/20 agents
- **Idle**: 15 agents
- **Failed**: 0 agents

### Performance Metrics

- **Avg Cycle Time**: 3.2 hours
- **Success Rate**: 94%
- **Parallelization Efficiency**: 78%
```

## Critical Success Factors

1. **Always scan for events** before declaring completion
2. **Agent specifications** must be read before execution
3. **Critic validation** after every phase
4. **Consolidation** of parallel outputs required
5. **GitHub comments** for all agent communication
6. **Event-driven** processing, not scheduled
7. **Continuous monitoring** until explicit stop
