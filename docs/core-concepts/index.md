# Core Concepts

!!! info "You're in Step 3 of the Learning Path"
    You've learned [why LionAGI is different](../thinking-in-lionagi/why-lionagi.md). Now let's understand the practical mechanics that make it work.

Understanding LionAGI's core abstractions is essential for building effective multi-agent workflows. These concepts work together to provide the parallel execution, memory isolation, and flexible orchestration that make LionAGI powerful.

## Key Abstractions

!!! tip "Learning Order"
    Read these in order for best understanding:  
    1. **Sessions & Branches** (the foundation)  
    2. **Operations** (how work gets done)  
    3. **Messages & Memory** (how context works)

### [Sessions and Branches](sessions-and-branches.md)

- **Session**: Workspace that coordinates multiple agents
- **Branch**: Individual agent with memory and tools

### [Operations](operations.md)

- Building blocks of workflows
- Types: chat, communicate, operate, ReAct

### [Messages and Memory](messages-and-memory.md)

- How conversation state is managed
- Memory isolation between branches

### [Tools and Functions](tools-and-functions.md)

- Extending agents with capabilities
- Built-in and custom tools

### [Models and Providers](models-and-providers.md)

- iModel abstraction for LLM providers
- Supporting OpenAI, Anthropic, Ollama, etc.

## The Mental Model

```python
# Traditional: Agents talk to each other
agent1 → agent2 → agent3 → result

# LionAGI: Agents work in parallel, results synthesized
agent1 ↘
agent2 → synthesis → result
agent3 ↗
```

## Architecture

```
Session (Workspace)
├── Branch (Agent 1)
│   ├── Messages (Memory)
│   ├── Tools
│   └── Model Config
├── Branch (Agent 2)
│   ├── Messages (Memory)
│   ├── Tools
│   └── Model Config
└── Graph (Workflow)
    ├── Operations
    └── Dependencies
```

---

!!! success "Ready to Build Workflows?"
    Now that you understand the core concepts, it's time to see them in action:
    
    **Next:** [Patterns](../patterns/) - Learn proven multi-agent workflow patterns  
    **Or:** [Cookbook](../cookbook/) - Jump to complete working examples
