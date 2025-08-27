# LionAGI

> **Central coordination engine for arbitrary orchestration patterns**

<div style="margin: 2em 0;">
<strong>LionAGI is evolving:</strong> From a collection of operations to a coordination engine where you define operations and we orchestrate them - agentic or otherwise.
</div>

---

## ⚡ The Orchestration Engine Mindset

```python
from lionagi import Branch, Operation

# Step 1: Define YOUR operations (not limited to what we provide)
class YourCustomAnalysis(Operation):
    async def execute(self, data):
        # Exactly how YOU want analysis done
        # Can use ML, APIs, databases, other frameworks, anything
        return your_analysis_logic(data)

class YourCustomSynthesis(Operation):
    async def execute(self, results):
        # YOUR synthesis logic
        return your_synthesis_logic(results)

# Step 2: LionAGI orchestrates them
branch = Branch()  # Minimal interface: just chat, communicate, operate, react

# Sequential orchestration
analysis = await branch.operate(YourCustomAnalysis(), data=my_data)
synthesis = await branch.operate(YourCustomSynthesis(), results=analysis)

# Or parallel orchestration
import asyncio

# Run operations in parallel
results = await asyncio.gather(
    branch.operate(YourCustomAnalysis(), data=my_data),
    branch.operate(AnotherOperation(), data=my_data),
    branch.operate(ThirdOperation(), data=my_data)
)

# The key: You define operations. We provide the orchestration engine.
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
