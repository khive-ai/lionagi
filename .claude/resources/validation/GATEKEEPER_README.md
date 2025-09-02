# KB Orchestrator Gatekeeper System

## Overview

The KB Orchestrator Gatekeeper is a critical component that implements the
mandatory completion check from CLAUDE.md. It ensures proper event processing
before any orchestrator task completion, preventing premature termination and
ensuring KB lifecycle integrity.

## Core Functionality

### Orchestrator Completion Check

The gatekeeper implements the core requirement from CLAUDE.md:

```python
# MANDATORY: Check for events before completion
async def orchestrator_completion_check():
    # 1. Run event scanner
    events = await scan_kb_events()
    
    # 2. Check parallelizable events
    if events['parallelizable']:
        raise Exception("Cannot complete - parallel events pending")
    
    # 3. Check sequential events  
    if events['sequential']:
        raise Exception("Cannot complete - sequential events pending")
    
    # 4. Check blocked events
    # 5. Validate all swarms completed
```

## Architecture

### Components

1. **OrchestratorGatekeeper** - Main gatekeeper class
2. **KBEventScanner** - GitHub issue/event scanner
3. **GitHubClient** - GitHub API integration
4. **Event Classification System** - Parallel/Sequential/Blocked determination
5. **Swarm Tracking** - Active swarm execution monitoring

### Event Classification

Events are classified into three categories:

- **Parallelizable**: Can be processed simultaneously
  - `research.requested`
  - `research.proposed`
  - `research.active`
  - `implementation.approved`
  - `implementation.active`

- **Sequential**: Must be processed in order
  - `decision.ready`
  - `decision.review`
  - `metrics.review`
  - `knowledge.captured`

- **Blocked**: Cannot be processed due to conditions
  - Missing dependencies
  - Resource conflicts
  - Timeout conditions
  - Explicit blocking labels

## Installation and Setup

### Prerequisites

1. Python 3.10+
2. GitHub repository with KB lifecycle labels
3. GitHub personal access token

### Dependencies

```bash
# Install via uv (preferred)
uv add aiohttp aiofiles pydantic pyyaml

# Or via pip
pip install aiohttp aiofiles pydantic pyyaml
```

### Environment Configuration

```bash
# Required environment variables
export GITHUB_TOKEN="your_github_token_here"
export GITHUB_REPOSITORY="owner/repo"

# Optional configuration
export KB_CONFIG_PATH=".claude/kb-config.yaml"
export GATEKEEPER_LOG_LEVEL="INFO"
```

### GitHub Repository Setup

Your repository must have the following labels configured:

#### Stage Labels (mutually exclusive)

- `stage:research.requested`
- `stage:research.proposed`
- `stage:research.active`
- `stage:decision.ready`
- `stage:decision.review`
- `stage:implementation.approved`
- `stage:implementation.active`
- `stage:metrics.review`

#### Category Labels

- `category:AIO` (AI/Orchestration)
- `category:MEM` (Memory/Persistence)
- `category:TLI` (Tools/Integration)
- `category:ARC` (Architecture)
- `category:DEV` (Developer Experience)
- `category:UXP` (User Experience)

#### Priority Labels

- `priority:critical`
- `priority:high`
- `priority:medium`
- `priority:low`

#### Status Labels

- `status:blocked`
- `status:needs-revision`
- `status:ready`
- `status:in-progress`

## Usage

### Basic Orchestrator Integration

```python
from gatekeeper import OrchestratorGatekeeper, GatekeeperException

async def orchestrator_workflow():
    """Example orchestrator workflow with gatekeeper integration"""
    
    # Initialize gatekeeper
    gatekeeper = OrchestratorGatekeeper()
    
    # Execute orchestrator tasks
    await execute_task_1()
    await execute_task_2()
    await execute_task_3()
    
    # CRITICAL: Check before completion
    try:
        await gatekeeper.orchestrator_completion_check()
        print("✅ All tasks completed successfully!")
        
    except GatekeeperException as e:
        print(f"❌ Cannot complete: {e.reason}")
        
        # Process pending events
        if e.events.parallelizable:
            await process_parallel_events(e.events.parallelizable)
        
        if e.events.sequential:
            await process_sequential_events(e.events.sequential)
        
        # Retry completion check
        await gatekeeper.orchestrator_completion_check()
```

### Command Line Usage

```bash
# Basic completion check
python gatekeeper.py --check

# Scan for events only
python gatekeeper.py --scan

# Emergency override
python gatekeeper.py --override "Production hotfix required"

# Run test scenarios
python gatekeeper.py --test
```

### Advanced Configuration

Create `.claude/resources/validation/gatekeeper-config.yaml`:

```yaml
github:
  api_url: "https://api.github.com"
  default_repo: "your-org/your-repo"

event_classification:
  parallelizable_events:
    - "research.requested"
    - "research.proposed"

  sequential_events:
    - "decision.ready"
    - "decision.review"

timeouts:
  stage_timeouts:
    research_requested: "1h"
    research_active: "8h"
    decision_review: "48h"

swarm_patterns:
  kb-intake-swarm:
    type: "parallel"
    max_agents: 5
    timeout: "1h"
```

## Event Processing Flow

### 1. Event Discovery

The gatekeeper scans GitHub issues for KB lifecycle events:

```python
# Scan all open issues
issues = await github.get_issues(state="open")

# Identify KB events by labels
for issue in issues:
    if has_stage_label(issue):
        event = create_kb_event(issue)
        classify_event(event)
```

### 2. Event Classification

Events are classified based on:

- **Stage Label**: Determines event type
- **Dependencies**: Checks completion markers in comments
- **Blocking Conditions**: Explicit blocks or timeouts
- **Resource Conflicts**: Agent availability and conflicts

### 3. Dependency Validation

Dependencies are validated through comment pattern matching:

```python
dependency_patterns = {
    "proposal_validated": r"\[.*INTAKE.*\].*COMPLETED.*proposal.*validated",
    "research_complete": r"\[.*RESEARCH.*\].*COMPLETED",
    "decision_approved": r"\[.*REVIEW.*\].*APPROVED"
}
```

### 4. Swarm Tracking

Active swarms are tracked through orchestrator comments:

```python
# Look for swarm start markers
swarm_pattern = r'\[ORCHESTRATOR-.*\].*Starting.*Swarm:\s*(\S+)'

# Check for completion markers
completion_pattern = r'\[ORCHESTRATOR-.*\].*COMPLETED.*{swarm_name}'
```

## Integration Points

### Orchestrator Integration

The orchestrator must call the gatekeeper before any completion:

```python
# In orchestrator code
async def complete_orchestration():
    """Complete orchestration workflow"""
    
    # ... execute all tasks ...
    
    # MANDATORY: Check for pending events
    gatekeeper = OrchestratorGatekeeper()
    await gatekeeper.orchestrator_completion_check()
    
    # Safe to complete
    return success_response()
```

### Agent Communication

Agents communicate through standardized GitHub comments:

```
[AGENT_ID-TIMESTAMP] ACTION_TYPE

## Status: STATUS_EMOJI STATUS

### Results
RESULTS_DETAILS

### Next Actions
NEXT_ACTIONS_LIST

---
Execution time: DURATION | Confidence: CONFIDENCE
```

### Swarm Orchestration

Swarms are initiated with tracking comments:

```
[ORCHESTRATOR-2024-01-15T10:00:00Z] Starting Event Processing

## Event: research_active
## Swarm: kb-research-swarm
## Research ID: AIO_001

### Initializing Agents
- codebase_analyst_agent
- memory_management_agent
- experiment_runner_agent

---
*Orchestration beginning*
```

## Error Handling

### GatekeeperException

The main exception raised when completion is not allowed:

```python
try:
    await gatekeeper.orchestrator_completion_check()
except GatekeeperException as e:
    # e.reason contains human-readable explanation
    # e.events contains full event scan results
    
    print(f"Completion blocked: {e.reason}")
    
    # Access specific event categories
    parallel_events = e.events.parallelizable
    sequential_events = e.events.sequential
    active_swarms = e.events.active_swarms
```

### Emergency Override

For exceptional circumstances:

```python
gatekeeper = OrchestratorGatekeeper()

# Override with documented reason
await gatekeeper.emergency_override(
    "Production hotfix required - bypassing KB lifecycle "
    "due to critical security vulnerability"
)
```

Emergency overrides are logged and create audit trails.

## Testing

### Unit Tests

Run the built-in test scenarios:

```bash
python gatekeeper-examples.py
```

### Integration Tests

Test with actual GitHub repository:

```bash
# Set up test repository
export GITHUB_REPOSITORY="your-org/test-repo"

# Run integration tests
python gatekeeper.py --test
```

### Mock Testing

For testing without GitHub API:

```python
from gatekeeper import KBEventScanner, EventType

# Mock GitHub client
scanner = KBEventScanner(None)

# Test event classification
mock_issue = {
    "labels": [{"name": "stage:research.requested"}],
    "title": "Test Research - AIO_001"
}

event = await scanner._identify_event(mock_issue)
assert event.event_type == EventType.RESEARCH_REQUESTED
```

## Monitoring and Logging

### Log Levels

- **INFO**: Normal operation events
- **WARNING**: Blocked events and conditions
- **ERROR**: Completion check failures
- **DEBUG**: Detailed event processing

### Audit Trail

The gatekeeper maintains audit trails:

- **Completion Reports**: `kb-completion-report-TIMESTAMP.md`
- **Override Records**: `kb-emergency-override-TIMESTAMP.json`
- **Event Logs**: `gatekeeper.log`

### Metrics

Track key metrics:

- Event processing times
- Completion check success rates
- Override frequency
- Blocking condition patterns

## Troubleshooting

### Common Issues

#### 1. GitHub API Rate Limiting

```
Error: GitHub API rate limit exceeded
```

**Solution**:

- Increase `api_calls_per_minute` in config
- Use authenticated requests with higher limits
- Implement exponential backoff

#### 2. Missing Research IDs

```
Warning: No research ID found for issue #123
```

**Solution**:

- Ensure issue titles contain research ID (e.g., AIO_001)
- Use proper issue templates
- Validate research ID format: `^[A-Z]{3}_\d{3}$`

#### 3. Dependency Detection Failures

```
Dependency not satisfied: proposal_validated
```

**Solution**:

- Check agent comment patterns
- Verify completion markers in issue comments
- Update dependency patterns in config

#### 4. Stale Events

```
Issue is stale - no recent activity
```

**Solution**:

- Update issue activity
- Adjust stale threshold in config
- Close inactive issues

### Debug Mode

Enable debug logging:

```bash
export GATEKEEPER_LOG_LEVEL="DEBUG"
python gatekeeper.py --scan
```

### Configuration Validation

Validate configuration:

```bash
python -c "
import yaml
with open('.claude/resources/validation/gatekeeper-config.yaml') as f:
    config = yaml.safe_load(f)
    print('Config valid:', bool(config))
"
```

## Security Considerations

### GitHub Token Security

- Use tokens with minimal required permissions
- Store tokens in secure environment variables
- Rotate tokens regularly
- Monitor token usage

### Access Control

- Limit override functionality to authorized operators
- Log all override attempts
- Require justification for overrides
- Review override audit trails

### API Security

- Validate all GitHub API responses
- Sanitize issue content before processing
- Prevent injection attacks in comment parsing
- Rate limit API calls appropriately

## Performance Optimization

### Caching

- Cache GitHub API responses (5-minute TTL)
- Cache event classification results
- Reuse GitHub client connections

### Batching

- Process events in batches
- Limit concurrent API calls
- Use parallel processing for independent operations

### Resource Management

- Set appropriate timeouts
- Clean up async resources
- Monitor memory usage
- Limit active agent counts

## Future Enhancements

### Planned Features

1. **Webhook Integration**: Real-time event processing
2. **Machine Learning**: Predictive event classification
3. **Dashboard**: Web UI for monitoring
4. **Notifications**: Slack/email integration
5. **Metrics Collection**: Prometheus/Grafana support

### Extension Points

- Custom event classifiers
- Additional dependency patterns
- External system integrations
- Custom completion conditions

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/kb-system

# Install dependencies
uv install

# Run tests
python gatekeeper-examples.py

# Format code
black gatekeeper.py
```

### Adding New Event Types

1. Add event type to `EventType` enum
2. Add classification rules to `event_definitions`
3. Add dependency patterns to config
4. Update tests

### Adding New Blocking Conditions

1. Extend `_check_blocking_conditions()` method
2. Add configuration options
3. Update documentation
4. Add test cases

---

**[GATEKEEPER_DESIGNER-2024-06-27T10:30:00Z]**

## Summary

The KB Orchestrator Gatekeeper provides a comprehensive solution for ensuring
proper event processing before task completion. It implements:

1. **Complete Python Script**: Ready-to-run implementation with full GitHub API
   integration
2. **Event Classification**: Robust logic for determining parallelizable vs
   sequential events
3. **Completion Validation**: All required checks with clear error messages
4. **Configuration System**: Flexible YAML-based configuration
5. **Test Framework**: Comprehensive test scenarios and examples
6. **Audit Trail**: Complete logging and override tracking
7. **Integration Ready**: Drop-in integration for orchestrator workflows

The system is production-ready and provides the critical gatekeeper
functionality required by CLAUDE.md for maintaining KB lifecycle integrity.
