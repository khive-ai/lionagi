# Quick Start

Get up and running with LionAGI in minutes. This guide shows you the essential concepts and gets you building multi-agent workflows immediately.

## What You'll Learn

In this quickstart, you'll discover how to:

- Create specialized AI agents with distinct roles and capabilities
- Coordinate multiple agents working in parallel
- Build workflows that leverage different AI perspectives

## Getting Started

1. **[Installation](installation.md)** - Install and configure LionAGI
2. **[Your First Flow](your-first-flow.md)** - Build your first multi-agent
   workflow
3. **[Claude Code Integration](claude-code-integration.md)** - Use with
   Anthropic's Claude Code

## Core Concept: Multiple Perspectives

The power of LionAGI lies in orchestrating multiple AI agents with different perspectives. Instead of relying on a single AI response, you can gather diverse viewpoints and synthesize them into more comprehensive insights.

```python
import asyncio
from lionagi import Branch, iModel

# Create agents with distinct roles
analyst = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
critic = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))

# Run them together in parallel
async def analyze(topic):
    analysis, critique = await asyncio.gather(
        analyst.chat(f"Analyze: {topic}"),
        critic.chat(f"Find issues with: {topic}")
    )
    return {"analysis": analysis, "critique": critique}

result = asyncio.run(analyze("AI safety"))
```

This example demonstrates the core LionAGI pattern: create specialized agents, run them in parallel using `asyncio.gather()`, and combine their outputs. The analyst provides comprehensive analysis while the critic identifies potential problems - giving you both perspectives simultaneously.

## Why This Approach Works

- **Parallel Processing**: Both agents work simultaneously, reducing total execution time
- **Specialized Perspectives**: Each agent focuses on their specific role and expertise  
- **Comprehensive Coverage**: Multiple viewpoints provide more thorough analysis than any single agent
- **Minimal Complexity**: Clean, readable code that's easy to understand and modify

This foundation scales from simple two-agent patterns to complex multi-agent orchestration workflows with dozens of specialized agents working together.
