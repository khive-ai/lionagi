# Tournament Validation Pattern

Competitive refinement where multiple agents propose solutions, then compete for
the best result.

## When to Use This Pattern

Use tournament validation when:

- Quality matters more than speed
- Multiple valid approaches exist
- Objective evaluation criteria can be defined
- Stakes are high (important decisions, critical code, etc.)
- You want to minimize bias from single perspectives

## Pattern Structure

1. **Generation**: Multiple agents create different solutions
2. **Evaluation**: Judge agents rate each solution
3. **Tournament**: Solutions compete head-to-head
4. **Refinement**: Winners refine their approach
5. **Selection**: Best solution emerges

## Basic Tournament Pattern

```python
from lionagi import Branch, Session, Builder, iModel

session = Session()
builder = Builder("solution_tournament")

# Create diverse problem solvers
creative_solver = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-sonnet-20240229"),
    system="Approach problems creatively with unconventional solutions."
)

analytical_solver = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Solve problems with systematic, analytical approaches."
)

practical_solver = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Focus on practical, implementable solutions."
)

# Judge for evaluation
judge = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-opus-20240229"),
    system="Evaluate solutions objectively on feasibility and effectiveness."
)

session.include_branches([creative_solver, analytical_solver, practical_solver, judge])

problem = "Design a system to reduce food waste while maintaining profitability"

# Phase 1: Generate competing solutions
creative_solution = builder.add_operation(
    "communicate",
    branch=creative_solver,
    instruction=f"Propose creative solution: {problem}"
)

analytical_solution = builder.add_operation(
    "communicate", 
    branch=analytical_solver,
    instruction=f"Propose systematic solution: {problem}"
)

practical_solution = builder.add_operation(
    "communicate",
    branch=practical_solver,
    instruction=f"Propose practical solution: {problem}"
)

# Execute solution generation
result = await session.flow(builder.get_graph())

solutions = {
    "creative": result["operation_results"][creative_solution],
    "analytical": result["operation_results"][analytical_solution],
    "practical": result["operation_results"][practical_solution]
}

# Phase 2: Judge evaluates all solutions
evaluation = await judge.communicate(f"""
Evaluate these solutions for: {problem}

Creative: {solutions['creative']}
Analytical: {solutions['analytical']}
Practical: {solutions['practical']}

Rate each 1-10 on feasibility, effectiveness, innovation.
Declare winner with reasoning.
""")
```

## Multi-Round Tournament

Elimination rounds with refinement:

```python
# Create multiple competitors with different approaches
competitors = [
    Branch(system="Prioritize user experience", chat_model=iModel(provider="openai", model="gpt-4o-mini")),
    Branch(system="Focus on technical excellence", chat_model=iModel(provider="openai", model="gpt-4o-mini")),
    Branch(system="Emphasize cost-effectiveness", chat_model=iModel(provider="openai", model="gpt-4o-mini"))
]

# Panel of specialized judges
judges = [
    Branch(system="Judge business viability", chat_model=iModel(provider="anthropic", model="claude-3-sonnet-20240229")),
    Branch(system="Judge technical feasibility", chat_model=iModel(provider="anthropic", model="claude-3-sonnet-20240229"))
]

challenge = "Create a mobile app for better sleep habits"

# Round 1: Initial proposals
proposals = {}
for i, competitor in enumerate(competitors):
    proposal = await competitor.communicate(f"Propose solution: {challenge}")
    proposals[f"competitor_{i}"] = proposal

# Round 1: Judging
scores = {}
for judge in judges:
    judge_scores = await judge.communicate(f"""
    Score these proposals 1-10: {proposals}
    Format: competitor_0: X/10, competitor_1: Y/10, etc.
    """)
    scores[judge] = judge_scores

# Round 2: Top performers refine solutions
# (Parse scores to select finalists)
finalists = competitors[:2]  # Top 2
for finalist in finalists:
    refined = await finalist.communicate("Refine your solution based on judge feedback")
    
# Final judging
winner = await judges[0].communicate("Select winner from refined proposals")
```

## Code Tournament

Specialized competition for code solutions:

```python
# Different coding philosophies
performance_coder = Branch(
    system="Write optimized, performance-focused code",
    chat_model=iModel(provider="openai", model="gpt-4o-mini")
)

readable_coder = Branch(
    system="Write clean, maintainable code",
    chat_model=iModel(provider="openai", model="gpt-4o-mini")
)

secure_coder = Branch(
    system="Write secure, robust code with error handling",
    chat_model=iModel(provider="anthropic", model="claude-3-sonnet-20240229")
)

code_judge = Branch(
    system="Senior developer evaluating code quality and best practices",
    chat_model=iModel(provider="anthropic", model="claude-3-opus-20240229")
)

coding_challenge = "Write Python function to get top 10 users by score from list of dicts"

# Generate solutions
solutions = {}
solutions["performance"] = await performance_coder.communicate(f"Optimize for speed: {coding_challenge}")
solutions["readable"] = await readable_coder.communicate(f"Optimize for readability: {coding_challenge}")  
solutions["secure"] = await secure_coder.communicate(f"Optimize for security: {coding_challenge}")

# Judge evaluates all solutions
evaluation = await code_judge.communicate(f"""
Rate these solutions 1-10 on correctness, performance, readability, security:
{solutions}

Declare winner and suggest hybrid approach.
""")
```

## Collaborative Tournament

Competition with cross-pollination:

```python
# Competing approaches
innovative = Branch(system="Focus on disruption", chat_model=iModel(provider="openai", model="gpt-4o-mini"))
practical = Branch(system="Focus on execution", chat_model=iModel(provider="openai", model="gpt-4o-mini"))
user_focused = Branch(system="Focus on user experience", chat_model=iModel(provider="anthropic", model="claude-3-sonnet-20240229"))

project = "Sustainable urban transportation"

# Initial proposals
proposals = {}
for team in [innovative, practical, user_focused]:
    proposals[team] = await team.communicate(f"Propose approach: {project}")

# Cross-pollination round
refined_proposals = {}
for team in [innovative, practical, user_focused]:
    refined_proposals[team] = await team.communicate(f"""
    Review other proposals: {proposals}
    Refine your approach by incorporating best elements from others.
    """)

# Collaborative synthesis
mediator = Branch(system="Identify synergies between approaches", chat_model=iModel(provider="anthropic", model="claude-3-opus-20240229"))

final_solution = await mediator.communicate(f"""
Create hybrid solution combining strengths: {refined_proposals}
""")
```

## Best Practices

### Clear Evaluation Criteria

```python
# Define specific, measurable criteria
evaluation_criteria = {
    "feasibility": "Can this be realistically implemented?",
    "effectiveness": "Will this solve the problem effectively?", 
    "innovation": "How creative/novel is this approach?",
    "scalability": "Can this work at larger scales?"
}
```

### Diverse Perspectives

```python
# Different specialties and approaches
competitors = [
    Branch(system="Focus on technical excellence"),
    Branch(system="Prioritize user experience"),
    Branch(system="Emphasize cost-effectiveness"),
    Branch(system="Consider sustainability")
]
```

### Objective Judging

```python
# Use specific scoring rubrics
judge_prompt = """
Rate 1-10 on:
1. Technical feasibility
2. Market viability  
3. Implementation complexity
4. Expected impact

Provide scores and reasoning.
"""
```

### Iterative Refinement

```python
# Allow winners to improve based on feedback
refinement_prompt = f"""
Your solution scored highest but judges noted: {feedback}
Refine to address concerns while maintaining strengths.
"""
```

## When to Use

**Perfect for:** High-stakes decisions, creative problems, quality-critical
tasks, complex analysis, innovation challenges

**Key advantage:** Competitive dynamics drive higher quality through diverse
perspectives, objective evaluation, and iterative refinement.

Tournament validation creates the highest quality solutions by leveraging
competition and collaborative improvement.
