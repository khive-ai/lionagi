# Brainstorming Workflows

Creative ideation using parallel agents with divergent → convergent thinking
patterns.

## Basic Brainstorming Pattern

```python
from lionagi import Branch, Session, Builder, iModel

session = Session()
builder = Builder("brainstorming")

# Create diverse creative agents
innovator = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-sonnet-20240229"),
    system="Innovative thinker generating bold, unconventional ideas."
)

pragmatist = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Focus on practical, implementable solutions."
)

contrarian = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Challenge assumptions and think from opposite perspectives."
)

synthesizer = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-sonnet-20240229"),
    system="Combine and refine ideas into coherent solutions."
)

session.include_branches([innovator, pragmatist, contrarian, synthesizer])

challenge = "How can we reduce plastic waste in urban environments?"

# Parallel ideation phase
innovator_ideas = builder.add_operation(
    "communicate",
    branch=innovator,
    instruction=f"Generate 3 innovative solutions: {challenge}"
)

pragmatist_ideas = builder.add_operation(
    "communicate", 
    branch=pragmatist,
    instruction=f"Generate 3 practical solutions: {challenge}"
)

contrarian_ideas = builder.add_operation(
    "communicate",
    branch=contrarian,
    instruction=f"3 unconventional approaches: {challenge}"
)

# Synthesis phase
synthesis = builder.add_aggregation(
    "communicate",
    branch=synthesizer,
    source_node_ids=[innovator_ideas, pragmatist_ideas, contrarian_ideas],
    instruction="Create 3 refined solutions combining best aspects"
)

result = await session.flow(builder.get_graph())
```

## Sequential Brainstorming Process

```python
# Sequential stages: Explore → Generate → Evaluate → Refine
explorer = Branch(system="Explore problems deeply, identify root causes")
generator = Branch(system="Generate many diverse ideas quickly")  
evaluator = Branch(system="Evaluate ideas for feasibility and impact")
refiner = Branch(system="Refine and improve promising ideas")

problem = "Remote team creative collaboration challenges"

# Chain dependent operations
explore = builder.add_operation("communicate", branch=explorer, 
                                instruction=f"Analyze problem: {problem}")

generate = builder.add_operation("communicate", branch=generator,
                                 instruction="Generate 10 solution approaches",
                                 depends_on=[explore])

evaluate = builder.add_operation("communicate", branch=evaluator,
                                 instruction="Evaluate and rank top 5 ideas",
                                 depends_on=[generate])

refine = builder.add_operation("communicate", branch=refiner,
                               instruction="Develop top ideas into solutions",
                               depends_on=[evaluate])

result = await session.flow(builder.get_graph())
```

## Multiple Perspectives

```python
# Different viewpoint agents
user_advocate = Branch(system="Represent end-user needs and experiences")
tech_expert = Branch(system="Focus on technical feasibility")
business_analyst = Branch(system="Consider business impact and ROI") 
creative_director = Branch(system="Focus on innovative user experiences")

challenge = "Design mobile app for sustainable living"

# Parallel perspective generation
import lionagi as ln

results = {}

async def get_perspective(name, agent):
    prompt = f"From {name} perspective, 3 key ideas for: {challenge}"
    results[name] = await agent.communicate(prompt)

async with ln.create_task_group() as tg:
    tg.start_soon(get_perspective, "user_advocate", user_advocate)
    tg.start_soon(get_perspective, "tech_expert", tech_expert)
    tg.start_soon(get_perspective, "business_analyst", business_analyst)
    tg.start_soon(get_perspective, "creative_director", creative_director)

# Synthesize perspectives
synthesizer = Branch(system="Synthesize diverse perspectives into solutions")
all_perspectives = "\n\n".join([f"{k}: {v}" for k, v in results.items()])
synthesis = await synthesizer.communicate(f"Synthesize: {all_perspectives}")
```

## Rapid Ideation Sprint

```python
# Quick parallel idea generation
generators = [
    Branch(system="Generate wild, unconventional ideas"),
    Branch(system="Focus on simple, elegant solutions"),
    Branch(system="Think scalable, systematic approaches"),
    Branch(system="Consider user-centered solutions")
]

topic = "Make coding accessible to beginners"

# Parallel rapid generation
all_ideas = []

async def quick_ideas(generator):
    ideas = await generator.communicate(f"5 quick ideas: {topic}")
    all_ideas.append(ideas)

async with ln.create_task_group() as tg:
    for generator in generators:
        tg.start_soon(quick_ideas, generator)

# Curate best ideas
curator = Branch(system="Identify and combine best ideas")
curation = await curator.communicate(f"Top 7 from: {all_ideas}")
```

## Best Practices

### Diverse Agent Personalities

```python
# Different thinking styles
agents = [
    Branch(system="Think analytically and systematically"),
    Branch(system="Think creatively and associatively"), 
    Branch(system="Think practically and implementally"),
    Branch(system="Think critically and skeptically")
]
```

### Clear Ideation Prompts

```python
# Good: Specific, actionable
"Generate 5 solutions for X under $Y budget in Z time"

# Avoid: Vague
"Think of some ideas"
```

### Structured Synthesis

```python
synthesis_prompt = f"""
Review ideas: {all_ideas}
Create 3 refined concepts that:
1. Combine best aspects
2. Address concerns  
3. Are actionable
"""
```

### Balance Divergence and Convergence

**Pattern:** Divergent (generate many) → Convergent (refine/combine) → Select
(develop)

## When to Use

**Perfect for:** Product development, problem solving, strategic planning,
content creation, process improvement

AI brainstorming leverages parallel processing and diverse perspectives for
faster, higher-quality ideation through structured synthesis phases.
