# Quick Start

Get up and running with LionAGI in minutes.

## Steps

1. **[Installation](installation.md)** - Install and configure LionAGI
2. **[Your First Flow](your-first-flow.md)** - Build your first multi-agent
   workflow
3. **[Claude Code Integration](claude-code-integration.md)** - Use with
   Anthropic's Claude Code

## Minimal Example

```python
import asyncio
from lionagi import Branch, iModel

# Create agents
analyst = Branch(chat_model=iModel(model="openai/gpt-4.1-mini"))
critic = Branch(chat_model=iModel(model="openai/gpt-4.1-mini"))

# Run them together
async def analyze(topic):
    analysis, critique = await asyncio.gather(
        analyst.chat(f"Analyze: {topic}"),
        critic.chat(f"Find issues with: {topic}")
    )
    return {"analysis": analysis, "critique": critique}

result = asyncio.run(analyze("AI safety"))
```

That's the core idea - multiple agents, working together, with minimal
complexity.
