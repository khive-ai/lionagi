# Thinking in LionAGI

Learn the mental model that makes LionAGI different.

## Core Ideas

### [Why LionAGI?](why-lionagi.md)
The technical differences that matter.

### [Branches as Agents](branches-as-agents.md)
Why we call them branches, not agents.

### [Graphs Over Chains](graphs-over-chains.md)
Parallel execution with dependency management.

### [The Builder Pattern](builder-pattern.md)
Constructing workflows programmatically.

## The Paradigm Shift

### From Conversations to Graphs

**Old way**: Agents have conversations
```python
agent1: "I found X"
agent2: "Based on X, I think Y"
agent3: "Given X and Y, we should Z"
```

**LionAGI way**: Agents execute operations
```python
x = await agent1.analyze()
y = await agent2.evaluate(x)
z = await agent3.synthesize([x, y])
```

### From Sequential to Parallel

**Old way**: Wait for each step
```python
step1()  # 2 sec
step2()  # 2 sec  
step3()  # 2 sec
# Total: 6 seconds
```

**LionAGI way**: Run simultaneously
```python
await gather(step1(), step2(), step3())
# Total: 2 seconds
```

### From Rigid to Flexible

**Old way**: Fixed agent roles
```python
researcher = Agent(role="researcher")
# Always a researcher, can't adapt
```

**LionAGI way**: Dynamic capabilities
```python
branch = Branch(system=context_specific_prompt)
# Adapts to the task at hand
```