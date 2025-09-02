# Session Diary Workflow Documentation

## Overview

The session diary workflow transforms verbose conversation summaries into
concise, learning-focused daily diaries that are efficiently loaded during
session initialization.

## Workflow Steps

### 1. Session Work

- Orchestrator (lion) conducts sessions with Task agents
- Conversation summaries are automatically generated
- Summaries stored in `.claude/notes/summaries/`

### 2. Session End Processing

```bash
# Run at end of day or when multiple summaries accumulate
uv run python /Users/lion/liongate/.claude/scripts/session_end.py

# Preview without changes
uv run python /Users/lion/liongate/.claude/scripts/session_end.py --dry-run

# Force reprocess already processed summaries
uv run python /Users/lion/liongate/.claude/scripts/session_end.py --force
```

This script:

- Finds unprocessed conversation summaries
- Groups them by date
- Extracts key learnings (not just accomplishments)
- Creates condensed diaries (~100-150 lines)
- Marks summaries as processed

### 3. Session Initialization

```bash
# At start of new session
uv run python /Users/lion/liongate/.claude/scripts/session_init.py

# With specific options
uv run python /Users/lion/liongate/.claude/scripts/session_init.py --issue 123 --resume
```

This script:

- Counts unprocessed summaries (alerts if > 0)
- Loads recent diaries (extracts key points)
- Shows pending tasks and git status
- Loads orchestrator documentation inline
- Generates batched memory queries

## Diary Content Focus

### Learning Categories (Priority Order)

1. **Orchestration Learnings** - How to better coordinate agents, use tools
2. **Task-Specific Learnings** - Technical insights, patterns discovered
3. **Interaction Learnings** - Ocean's preferences, guidance received

### Diary Structure

```markdown
# Daily Diary: YYYY-MM-DD

## Sessions

1. **Session Title** (duration)
   - Key point 1
   - Key point 2

## Highlights

- Major achievement with impact

## 📊 By The Numbers

- Sessions Completed: X
- Tasks Accomplished: Y
- Insights Captured: Z

## 🔧 Key Work Done

- Work item with description

## 🔑 Key Technical Decisions

1. **Decision**: Rationale

## Tomorrow's Focus

### Immediate Priorities

1. **Task**: Reason

### Medium-term Goals

- Goal 1

## Reflections

Thoughtful paragraph about the day's progress and learnings.

## Technical Debt & Risks

- Item needing attention

## Learning & Insights

- [Orchestration] Batch tool usage improves efficiency
- [Technical] Pattern discovered in implementation
- [Ocean's Guidance] Preference noted and adapted

---

_End note focusing on key learning_
```

## Key Design Decisions

### 1. Balanced Content

- Less focus on bragging about accomplishments
- More focus on learnings and improvements
- Include challenges and technical debt

### 2. Efficient Loading

- Diaries are 100-150 lines (vs 1000+ line summaries)
- Key points extracted for quick scanning
- Structured format for easy parsing

### 3. Learning Extraction

The system specifically looks for:

- Orchestration keywords: "orchestrat", "swarm", "agent", "parallel", "batch"
- Technical patterns: Insights marked as technical or architectural
- Ocean's guidance: References to preferences, corrections, adaptations

### 4. Session Context Preservation

- Session IDs and dates maintained
- Cross-session connections identified
- Progress tracking across days

## File Locations

- **Summaries**: `/Users/lion/liongate/.claude/notes/summaries/`
- **Diaries**: `/Users/lion/liongate/.claude/notes/diaries/`
- **Scripts**: `/Users/lion/liongate/.claude/scripts/`
  - `session_init.py` - Start of session
  - `session_end.py` - End of day processing

## Best Practices

1. **Run session_end.py regularly** - Don't let summaries pile up
2. **Review diary output** - Ensure learnings are captured correctly
3. **Start sessions with init** - Always check for unprocessed summaries
4. **Focus on learning** - Diaries should help you improve, not just record

## Integration with Memory MCP

During session initialization, the generated memory queries include:

- Ocean's preferences (always first)
- KB rules and workflows
- Recent learnings from diaries
- Specific context based on flags (--issue, --resume)

This creates a comprehensive context while maintaining efficiency.

---

_Session diary workflow - Transform verbose summaries into actionable learning_
