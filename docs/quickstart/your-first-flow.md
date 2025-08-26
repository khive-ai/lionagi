# Your First Flow

Let's build something useful: multiple AI agents working together to analyze a
problem from different angles.

## Basic Multi-Agent Pattern

```python
import asyncio
from lionagi import Branch, iModel

async def analyze_idea():
    # Create specialized agents
    critic = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="You are a critical thinker. Find potential problems and weaknesses."
    )
    
    supporter = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="You are optimistic and supportive. Find the potential and opportunities."
    )
    
    analyst = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="You are analytical and data-driven. Focus on facts and feasibility."
    )
    
    idea = "Starting a subscription box service for AI developers"
    
    # Get three different perspectives
    critical_view = await critic.communicate(f"Evaluate this idea: {idea}")
    positive_view = await supporter.communicate(f"What's exciting about: {idea}")
    analytical_view = await analyst.communicate(f"Analyze the market for: {idea}")
    
    print("CRITIC:", critical_view)
    print("\nSUPPORTER:", positive_view)
    print("\nANALYST:", analytical_view)

asyncio.run(analyze_idea())
```

## Parallel Execution

The above runs sequentially. Let's make it faster with parallel execution:

```python
import asyncio
from lionagi import Branch, iModel

async def parallel_analysis():
    # Same agents
    critic = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="You are a critical thinker. Find problems."
    )
    
    supporter = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="You are optimistic. Find opportunities."
    )
    
    analyst = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="You are analytical. Focus on data."
    )
    
    idea = "Starting a subscription box service for AI developers"
    
    # Run all three at the same time
    results = await asyncio.gather(
        critic.communicate(f"Evaluate: {idea}"),
        supporter.communicate(f"Opportunities in: {idea}"),
        analyst.communicate(f"Market analysis: {idea}")
    )
    
    critical_view, positive_view, analytical_view = results
    
    # Now synthesize
    synthesizer = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="You synthesize multiple viewpoints into balanced recommendations."
    )
    
    synthesis = await synthesizer.communicate(
        f"Synthesize these perspectives:\n\n"
        f"Critical: {critical_view}\n\n"
        f"Positive: {positive_view}\n\n"
        f"Analytical: {analytical_view}"
    )
    
    print("FINAL SYNTHESIS:", synthesis)

asyncio.run(parallel_analysis())
```

## Using Sessions for Coordination

For more complex workflows, use Session and Builder:

```python
from lionagi import Session, Branch, Builder, iModel
import asyncio

async def coordinated_brainstorm():
    session = Session()
    builder = Builder("brainstorm")
    
    # Create a brainstorming team
    creative = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="You generate creative, unconventional ideas."
    )
    
    practical = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="You focus on practical, implementable solutions."
    )
    
    devil = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="You play devil's advocate, challenging assumptions."
    )
    
    problem = "How to make technical documentation more engaging"
    
    # Define the workflow
    creative_ideas = builder.add_operation(
        "communicate",
        branch=creative,
        instruction=f"Generate 3 creative solutions for: {problem}"
    )
    
    practical_ideas = builder.add_operation(
        "communicate",
        branch=practical,
        instruction=f"Generate 3 practical solutions for: {problem}"
    )
    
    # Devil's advocate reviews both
    challenge = builder.add_operation(
        "communicate",
        branch=devil,
        depends_on=[creative_ideas, practical_ideas],
        instruction="Challenge and improve the proposed solutions"
    )
    
    # Execute the workflow
    result = await session.flow(builder.get_graph())
    
    print("Creative Ideas:", result["operation_results"][creative_ideas])
    print("\nPractical Ideas:", result["operation_results"][practical_ideas])
    print("\nDevil's Advocate:", result["operation_results"][challenge])

asyncio.run(coordinated_brainstorm())
```

## Conversation Memory

Branches maintain conversation history, enabling follow-up questions:

```python
from lionagi import Branch, iModel
import asyncio

async def iterative_refinement():
    comedian = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="You are a comedian who makes technical concepts funny."
    )
    
    editor = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="You are an editor who improves clarity and punch."
    )
    
    # First draft
    joke = await comedian.communicate(
        "Write a joke about Python's Global Interpreter Lock"
    )
    print("First draft:", joke)
    
    # Editor feedback
    feedback = await editor.communicate(
        f"How can we improve this joke: {joke}"
    )
    print("Editor feedback:", feedback)
    
    # Comedian revises (has memory of first joke)
    revision = await comedian.communicate(
        f"Revise the joke based on this feedback: {feedback}"
    )
    print("Revised joke:", revision)
    
    # Editor can also remember the context
    final_check = await editor.communicate(
        "Is this version better than the original?"
    )
    print("Final check:", final_check)

asyncio.run(iterative_refinement())
```

## Common Patterns

### Research Team

```python
researcher = Branch(system="You find relevant information")
fact_checker = Branch(system="You verify accuracy")
summarizer = Branch(system="You create concise summaries")
```

### Creative Team

```python
writer = Branch(system="You write engaging content")
editor = Branch(system="You improve clarity and flow")
reviewer = Branch(system="You ensure quality standards")
```

### Analysis Team

```python
data_analyst = Branch(system="You analyze quantitative data")
strategist = Branch(system="You identify strategic implications")
risk_assessor = Branch(system="You identify potential risks")
```

## Tips for Effective Branches

1. **Clear Roles**: Give each branch a specific, well-defined role
2. **Complementary Skills**: Create branches that complement each other
3. **Iterative Refinement**: Use conversation memory for multi-round improvement
4. **Parallel When Possible**: Use `asyncio.gather()` for independent tasks
5. **Sequential When Needed**: Use dependencies for tasks that build on each
   other

## Next Steps

- Explore [patterns](../patterns/) for proven multi-agent workflows
- Check the [cookbook](../cookbook/) for complete examples
- Learn about [Sessions and Branches](../core-concepts/sessions-and-branches.md)
  in depth

## Try It Yourself

Start with the basic multi-agent pattern above and experiment:

- Change the system prompts to create different expert types
- Add more branches for additional perspectives
- Try different coordination patterns
- Build something useful for your actual work
