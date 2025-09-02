# LionAGI

**Build AI workflows you can trust by coordinating multiple agents**

## The Problem

AI reasoning is a black box, but AI workflows don't have to be.

When you ask an AI agent a complex question, you get one answer. But how do you know it's right? How do you know it considered all angles? Those "reasoning traces" you see? They're just generated text, not actual thinking.

LionAGI solves this by making AI workflows **observable**. Instead of trusting what one model tells you about its thinking, you orchestrate multiple specialists and see exactly what each one does.

[Read our full problem statement →](problem-statement.md)

## Installation

```bash
pip install lionagi
```

Set your API keys (use any or all):
```bash
# OpenAI for GPT models
export OPENAI_API_KEY=your-key

# NVIDIA NIM for Llama, Mistral (1000 free credits)
export NVIDIA_NIM_API_KEY=nvapi-your-key  # Get at build.nvidia.com

# Claude Code for workspace-aware agents
# Configured via Claude Code desktop app
```

## Your First Observable Workflow

Here's the simplest example - getting multiple perspectives using different providers:

```python
from lionagi import Branch, iModel
import asyncio

async def main():
    # Use different models for different strengths
    
    # OpenAI GPT-4.1 for analysis (1M token context)
    analyst = Branch(
        system="You analyze business opportunities",
        chat_model=iModel(provider="openai", model="gpt-4.1")
    )
    
    # NVIDIA NIM with DeepSeek V3.1 for risk assessment (latest preview model)
    critic = Branch(
        system="You identify risks and challenges",
        chat_model=iModel(provider="nvidia_nim", model="deepseek-ai/deepseek-v3.1")
    )
    
    # Claude Code for implementation planning (workspace-aware)
    planner = Branch(
        system="You create actionable implementation plans",
        chat_model=iModel(provider="claude_code", endpoint="query_cli")
    )
    
    # Ask all three about the same decision
    question = "Should our startup expand to Europe?"

    # Parallel execution - all models work simultaneously
    analysis, risks, plan = await asyncio.gather(
        analyst.chat(question),
        critic.chat(question),
        planner.chat(question)
    )
    
    print("Analysis (GPT-4.1):", analysis)
    print("Risks (DeepSeek V3.1):", risks)
    print("Plan (Claude Code):", plan)
    # Every perspective is visible, using the best model for each task

asyncio.run(main())
```

### Available Models

**OpenAI**: GPT-5, GPT-4.1, GPT-4.1-mini, GPT-4o, GPT-4o-mini  
**NVIDIA NIM**: DeepSeek V3.1 (latest preview), Llama 3.2 Vision, Mistral Large, Mixtral 8x22B  
**Claude Code**: Workspace-aware development with file access  
**Also supported**: Anthropic Claude, Google Gemini, Ollama (local), Groq, Perplexity

## Why Observable Workflows Matter

- **Trust through transparency**: See every step, not just the final answer
- **Multiple perspectives**: Different agents catch different issues
- **Audit trails**: Every decision is logged and reproducible
- **No black boxes**: You control the workflow, not agent conversations

## When to Use LionAGI

✅ **Perfect for:**
- Complex decisions needing multiple perspectives
- Production systems requiring audit trails
- Workflows where you need to see the reasoning
- Coordinating different models for different tasks

❌ **Not for:**
- Simple chatbots
- Basic Q&A
- Prototypes where you want agents to chat freely

## Core Concepts Made Simple

**Branches = Agents**  
Each Branch is an independent agent with its own context

**Explicit > Implicit**  
You control the workflow, not agent conversations

**Observable > Explainable**  
See what happened, don't trust what models claim

## Quick Patterns

### Get Multiple Perspectives
```python
# Parallel analysis from different angles
results = await asyncio.gather(
    technical_agent.chat("Technical implications?"),
    business_agent.chat("Business impact?"),
    legal_agent.chat("Legal considerations?")
)
# See all perspectives at once
```

### Use the Right Model for Each Job
```python
# Complex analysis needs powerful model
researcher = Branch(chat_model=iModel(provider="anthropic", model="claude-3-opus"))
research = await researcher.chat("Deep dive into quantum computing")

# Simple summary can use cheaper model
summarizer = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
summary = await summarizer.chat(f"Three key points: {research}")
```

## Learning Path

1. **Start Here** → [Your First Flow](quickstart/your-first-flow.md)
2. **Understand Why** → [Why LionAGI?](thinking-in-lionagi/why-lionagi.md)
3. **Learn Basics** → [Sessions and Branches](core-concepts/sessions-and-branches.md)
4. **Apply Patterns** → [Common Workflows](patterns/index.md)
5. **Go Deeper** → [Advanced Topics](advanced/index.md)

## The Key Difference

```python
# ❌ Other frameworks: Agents figure it out themselves
result = agent_conversation(agent1, agent2, agent3, problem)
# Who knows what happened?

# ✅ LionAGI: You orchestrate, agents execute
step1 = await analyst.analyze(problem)
step2 = await critic.review(step1)
step3 = await synthesizer.combine(step1, step2)
# Every step visible and verifiable
```

## Get Started

**Ready to build?** Start with [Your First Flow](quickstart/your-first-flow.md) →

**Have questions?** Check our [Problem Statement](problem-statement.md) to understand our philosophy

**Need help?** [GitHub Issues](https://github.com/lion-agi/lionagi) | [Discord](https://discord.gg/lionagi)

---

*LionAGI: Observable workflows for trustworthy AI*

Apache 2.0 License
