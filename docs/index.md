# LionAGI

> **Central coordination engine for arbitrary orchestration patterns**

<div style="margin: 2em 0;">
<strong>LionAGI is evolving:</strong> From a collection of operations to a coordination engine where you define operations and LionAGI orchestrates them - agentic or otherwise.
</div>

---

## ⚡ The Orchestration Engine Mindset

```python
from lionagi import Branch, iModel
import asyncio

# Step 1: Create specialized branches for different analysis types
analysis_branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="You are a data analyst who finds patterns and insights in data."
)

synthesis_branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"), 
    system="You synthesize multiple analyses into actionable recommendations."
)

# Step 2: LionAGI orchestrates your workflow
my_data = "Sales data shows 30% increase in Q4..."

# Sequential orchestration - each step builds on the last
analysis = await analysis_branch.communicate(
    instruction="Analyze this business data for key trends",
    context={"data": my_data}
)

synthesis = await synthesis_branch.communicate(
    instruction="Create strategic recommendations from this analysis", 
    context={"analysis_results": analysis}
)

# Or parallel orchestration - run multiple analyses simultaneously  
results = await asyncio.gather(
    analysis_branch.communicate(instruction="Find revenue patterns", context={"data": my_data}),
    analysis_branch.communicate(instruction="Identify customer trends", context={"data": my_data}),
    analysis_branch.communicate(instruction="Analyze operational metrics", context={"data": my_data})
)

# The key: You define the workflow. LionAGI provides reliable orchestration.
```

## 🎯 What LionAGI Does Well

| Feature | What It Means |
|---------|---------------|
| **Parallel Execution** | Run multiple agents simultaneously without blocking |
| **Clear Dependencies** | Explicitly define execution order and data flow |
| **Isolated State** | Each agent maintains independent context and memory |
| **Predictable Workflows** | Deterministic graphs instead of unpredictable conversations |

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

## Your Learning Path

!!! tip "Choose Your Starting Point"
    **New to LionAGI?** → Follow the path below step-by-step  
    **Migrating from another framework?** → Jump to [Migration Guides](migration/)  
    **Need specific patterns?** → Browse [Cookbook](cookbook/) for ready-to-use examples

### 🚀 Step 1: Get Started (5 minutes)

- [Installation](quickstart/installation.md) - Get set up in 2 minutes
- [Your First Flow](quickstart/your-first-flow.md) - Build your first multi-agent workflow

### 🧠 Step 2: Understand the Paradigm (10 minutes)

- [Why LionAGI?](thinking-in-lionagi/why-lionagi.md) - Technical differences that matter
- [Thinking in LionAGI](thinking-in-lionagi/) - Learn the mental model

### 🔧 Step 3: Master Core Concepts (15 minutes)

Now that you understand the "why," learn the "how":

- [Sessions and Branches](core-concepts/sessions-and-branches.md) - How agents are organized
- [Operations](core-concepts/operations.md) - Building blocks of workflows
- [Messages and Memory](core-concepts/messages-and-memory.md) - How context is managed

### ⚡ Step 4: Apply Common Patterns (20 minutes)

Ready to build real workflows? Start with proven patterns:

- [Fan-Out/In](patterns/fan-out-in.md) - Parallel analysis with synthesis
- [Sequential Analysis](patterns/sequential-analysis.md) - Building understanding step-by-step
- [Conditional Flows](patterns/conditional-flows.md) - Dynamic execution paths

### 📚 Step 5: Explore Production Examples

Put it all together with complete, working examples:

- [Cookbook](cookbook/) - Copy-and-modify production workflows
- [Integration Guide](integrations/) - Connect with databases, tools, and services

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

LionAGI is open source.
[Contribute on GitHub](https://github.com/khive-ai/lionagi).

## License

Apache 2.0
