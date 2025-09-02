# KB Task Master - Simple Event Queue for Orchestrator

This directory contains the simplified task tracking system that prevents the
orchestrator from completing tasks prematurely.

## The Problem

The orchestrator needs to process all KB events (GitHub issues with lifecycle
labels) before declaring task completion. Without this check, the orchestrator
might stop too early, leaving work unfinished.

## The Solution

Two simple scripts that work together:

### 1. `task_master.py` - The Task Tracker

This script scans GitHub for KB-related issues and tracks what needs to be done:

```bash
# Check if any tasks are pending (exit code tells you)
python task_master.py --check

# List all pending tasks in detail  
python task_master.py --list

# Get the next task to process
python task_master.py --next
```

**What it tracks:**

- 📥 **Intake tasks** - New research requests (`stage:research.requested`)
- 📝 **Planning tasks** - Proposals to plan (`stage:research.proposed`)
- 🔬 **Research tasks** - Active research (`stage:research.active`)
- 🤔 **Decision tasks** - Decisions to make (`stage:decision.*`)
- 🚀 **Implementation tasks** - Work in progress (`stage:implementation.*`)
- 📊 **Metrics tasks** - ROI analysis (`stage:metrics.*`)
- 🚫 **Blocked tasks** - Tasks that need help (`status:blocked`)

### 2. `simple_gatekeeper.py` - The Completion Guard

This implements the critical check from CLAUDE.md:

```bash
# Run before declaring any task complete
python simple_gatekeeper.py
```

Returns:

- **Exit 0**: ✅ Safe to complete (no pending tasks)
- **Exit 1**: ❌ Cannot complete (tasks still pending)

## How the Orchestrator Uses This

```python
# Orchestrator pseudo-code
while True:
    # Check what needs doing
    subprocess.run(["python", "task_master.py", "--next"])
    
    # Process tasks...
    
    # Before completing, MUST run gatekeeper
    result = subprocess.run(["python", "simple_gatekeeper.py"])
    if result.returncode == 0:
        print("All done!")
        break
    else:
        print("More work to do...")
        continue
```

See `orchestrator_example.py` for a complete example.

## Setup

1. Set your GitHub token:
   ```bash
   export GITHUB_TOKEN=your_github_token_here
   ```

2. Run the example:
   ```bash
   python orchestrator_example.py
   ```

## Why So Simple?

This is intentionally minimal because:

- It's just a todo list tracker for the orchestrator
- No complex dependencies or infrastructure needed
- Easy to understand and debug
- Does one thing well: prevents premature completion

## Integration with CLAUDE.md

This implements the requirement:

> **MANDATORY: Check for events before completion**
>
> ```python
> async def orchestrator_completion_check():
>     # 1. Run event scanner
>     events = await scan_kb_events()
>     
>     # 2. Check parallelizable events
>     if events['parallelizable']:
>         raise Exception("Cannot complete - parallel events pending")
>     
>     # 3. Check sequential events  
>     if events['sequential']:
>         raise Exception("Cannot complete - sequential events pending")
> ```

The `simple_gatekeeper.py` is the concrete implementation of this requirement.
