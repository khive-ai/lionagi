# LionAGI

Multi-agent orchestration that treats coordination as a graph problem, not a conversation problem.

## Quick Example

```python
import asyncio
from lionagi import Branch, iModel

async def multi_perspective_analysis():
    # Create specialized agents
    technical = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="You analyze technical feasibility"
    )
    business = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="You analyze business impact"
    )
    user = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="You analyze user experience"
    )
    
    question = "Should we rewrite our backend in Rust?"
    
    # Get all perspectives in parallel
    tech_view, biz_view, user_view = await asyncio.gather(
        technical.chat(question),
        business.chat(question),
        user.chat(question)
    )
    
    return {"technical": tech_view, "business": biz_view, "user": user_view}

result = asyncio.run(multi_perspective_analysis())
```

## What LionAGI Does Well

- **Parallel execution**: Run multiple agents simultaneously
- **Clear dependencies**: Define what depends on what
- **Isolated state**: Each agent maintains its own context
- **Predictable workflows**: Graphs, not conversations

## Get Started

### Install
```bash
pip install lionagi
```

### Set API Key
```bash
export OPENAI_API_KEY=your-key
```

### First Agent
```python
from lionagi import Branch, iModel
import asyncio

async def main():
    agent = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
    response = await agent.chat("Explain REST APIs")
    print(response)

asyncio.run(main())
```

## Learn More

### Start Here
- [Installation](quickstart/installation.md) - Get set up in 2 minutes
- [Your First Flow](quickstart/your-first-flow.md) - Build multi-agent workflows
- [Why LionAGI](thinking-in-lionagi/why-lionagi.md) - Technical differences that matter

### Core Concepts
- [Sessions and Branches](core-concepts/sessions-and-branches.md) - How agents are organized
- [Operations](core-concepts/operations.md) - Building blocks of workflows
- [Messages and Memory](core-concepts/messages-and-memory.md) - How context is managed

### Patterns
- [Fan-Out/In](patterns/fan-out-in.md) - Parallel analysis with synthesis
- [Sequential Analysis](patterns/sequential-analysis.md) - Building understanding step-by-step
- [Conditional Flows](patterns/conditional-flows.md) - Dynamic execution paths

### Examples
- [Cookbook](cookbook/) - Complete working examples
- [Integration Guide](integrations/) - Connect with tools and databases

## When to Use LionAGI

**Good fit for:**
- Systems that need multiple AI perspectives
- Workflows requiring parallel processing
- Production applications needing reliability
- Complex orchestration with clear dependencies

**Not ideal for:**
- Simple single-agent chatbots
- Purely sequential chains
- Experimental conversation flows

## Contributing

LionAGI is open source. [Contribute on GitHub](https://github.com/khive-ai/lionagi).

## License

Apache 2.0