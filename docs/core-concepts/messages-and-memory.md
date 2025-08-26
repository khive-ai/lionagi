# Messages and Memory

LionAGI automatically manages conversation state and memory through branches.

## Automatic Memory Management

```python
from lionagi import Branch, iModel

assistant = Branch(
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="You are a helpful assistant"
)

# Messages handled automatically
response = await assistant.chat("Explain Python list comprehensions")

# Follow-up maintains context
clarification = await assistant.chat("Show a complex example")
```

## Message History

```python
from lionagi import Branch, iModel

analyst = Branch(
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="Data analyst"
)

await analyst.chat("Analyze this data: [1000, 1200, 800, 1500]")
await analyst.chat("What's the trend?")

# Access message history
print(f"Total messages: {len(analyst.messages)}")

# Iterate through messages
for msg in analyst.messages:
    print(f"{msg.role}: {msg.content[:50]}...")
```

## Multi-Branch Memory

Each branch maintains independent memory within a session:

```python
from lionagi import Session, Branch, iModel

session = Session()

researcher = Branch(
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="Research specialist",
    name="researcher"
)

critic = Branch(
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="Critical analyst",
    name="critic"
)

session.include_branches([researcher, critic])

# Each branch maintains separate memory
await researcher.chat("Research AI safety measures")
await critic.chat("What are potential risks of AI?")

print(f"Researcher: {len(researcher.messages)} messages")
print(f"Critic: {len(critic.messages)} messages")
```

## Context Continuity

Conversation context is automatically maintained:

```python
mathematician = Branch(
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="Math tutor"
)

await mathematician.chat("I'm learning calculus")
await mathematician.chat("Explain derivatives")
await mathematician.chat("What about the chain rule?")

# References full conversation context
quiz = await mathematician.chat("Quiz me on what we've covered")
```

## Memory Serialization

Export conversation data for persistence:

```python
import json
from lionagi import Branch, iModel

assistant = Branch(
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="Helpful assistant"
)

await assistant.chat("My favorite color is blue")
await assistant.chat("I work as a software engineer")

# Export conversation
data = {
    "name": assistant.name,
    "messages": [
        {"role": msg.role, "content": msg.content}
        for msg in assistant.messages
    ]
}

with open("conversation.json", "w") as f:
    json.dump(data, f)
```

## Workflow Memory

Each branch in Builder workflows maintains independent memory:

```python
from lionagi import Session, Builder, Branch, iModel

session = Session()
builder = Builder("workflow")

collector = Branch(system="Collect information")
analyzer = Branch(system="Analyze information")

session.include_branches([collector, analyzer])

collect_op = builder.add_operation(
    "communicate",
    branch=collector,
    instruction="Collect renewable energy data"
)

analyze_op = builder.add_operation(
    "communicate",
    branch=analyzer,
    instruction="Analyze the collected data",
    depends_on=[collect_op]
)

result = await session.flow(builder.get_graph())
```

## Best Practices

**Use descriptive system prompts** for consistent memory context:

```python
specialist = Branch(
    chat_model=iModel(provider="openai", model="gpt-4"),
    system="You are Dr. Smith, a cardiologist with 20 years experience"
)
```

**Use `communicate()` for stateful conversations**, `chat()` for independent
queries:

```python
# Stateful conversation
await therapist.communicate("I had a difficult day")
await therapist.communicate("Tell me about stress management")

# Independent query
await assistant.chat("What time is it in Tokyo?")
```

**Leverage specialized branches** for domain-specific memory:

```python
research_team = {
    "reviewer": Branch(system="Literature reviewer"),
    "analyst": Branch(system="Data analyst"),
    "writer": Branch(system="Report writer")
}
```

LionAGI automatically handles conversation state, letting you focus on building
intelligent workflows without managing message details.
