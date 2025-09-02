# Multi-Agent Code Review

Use specialized agents to review code from multiple perspectives.

## Basic Multi-Agent Review

```python
from lionagi import Session, Branch, iModel

session = Session()

# Create specialized reviewers
security = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Security expert. Focus only on security issues."
)

performance = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Performance expert. Focus only on performance issues."
)

maintainability = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Code quality expert. Focus on maintainability and readability."
)

session.include_branches([security, performance, maintainability])

# Code to review
code = '''
def login(user, pwd):
    query = f"SELECT * FROM users WHERE name='{user}' AND pass='{pwd}'"
    return db.execute(query).fetchone()
'''

# Parallel reviews using LionAGI TaskGroup
import lionagi as ln

reviews = {}

async def security_review():
    reviews["security"] = await security.chat(f"Security review: {code}")

async def performance_review():
    reviews["performance"] = await performance.chat(f"Performance review: {code}")

async def maintainability_review():
    reviews["maintainability"] = await maintainability.chat(f"Code quality review: {code}")

async with ln.create_task_group() as tg:
    tg.start_soon(security_review)
    tg.start_soon(performance_review)
    tg.start_soon(maintainability_review)

# All tasks complete when TaskGroup context exits
review_results = reviews
```

## Builder Pattern Review

```python
from lionagi import Session, Builder, Branch, iModel

session = Session()
builder = Builder("code_review")

# Reviewers
security_branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Security code reviewer"
)
quality_branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Code quality reviewer"
)

session.include_branches([security_branch, quality_branch])

# Code snippet to review (example with SQL injection vulnerability)
user_input = "1 OR 1=1"  # Example malicious input
code_snippet = "SELECT * FROM users WHERE id=" + user_input

# Parallel review operations
security_review = builder.add_operation(
    "communicate",
    branch=security_branch,
    instruction=f"Review for security issues: {code_snippet}"
)

quality_review = builder.add_operation(
    "communicate", 
    branch=quality_branch,
    instruction=f"Review for code quality: {code_snippet}"
)

# Synthesis
synthesis = builder.add_aggregation(
    "communicate",
    branch=security_branch,
    source_node_ids=[security_review, quality_review],
    instruction="Summarize all review findings"
)

result = await session.flow(builder.get_graph())
```

## Review with Final Decision

```python
session = Session()
builder = Builder("comprehensive_review")

# Create multiple reviewers
review_types = ["security", "performance", "maintainability", "correctness"]
reviewers = {}
review_ops = []

for review_type in review_types:
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system=f"{review_type.title()} code reviewer"
    )
    reviewers[review_type] = branch
    
    op_id = builder.add_operation(
        "communicate",
        branch=branch,
        instruction=f"{review_type} review of submitted code"
    )
    review_ops.append(op_id)

# Senior reviewer for final decision
senior = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-sonnet-20240229"),
    system="Senior code reviewer who makes final approval decisions"
)

session.include_branches([*reviewers.values(), senior])

# Final synthesis
final_decision = builder.add_aggregation(
    "communicate",
    branch=senior,
    source_node_ids=review_ops,
    instruction="Based on all reviews, provide final APPROVE/REJECT decision"
)

result = await session.flow(builder.get_graph())
```

## Best Practices

### Specialized Systems

```python
# Good: Clear specialization
security = Branch(system="Focus only on security vulnerabilities")
performance = Branch(system="Focus only on performance bottlenecks")

# Avoid: Generic reviewers
generic = Branch(system="Review all aspects of code")
```

### Structured Output

```python
instruction = """
Review this code for security issues:

Format response as:
- Issues Found: [list]
- Severity: [high/medium/low]  
- Recommendations: [list]
"""
```

### Advanced Parallel Execution

```python
# LionAGI TaskGroup (recommended)
async with ln.create_task_group() as tg:
    tg.start_soon(security_task)
    tg.start_soon(performance_task)
    tg.start_soon(quality_task)

# Or asyncio.gather() for simple cases
reviews = await asyncio.gather(
    security.chat(prompt),
    performance.chat(prompt)
)
```

Multi-agent code review leverages specialized expertise in parallel, catching
issues that single reviewers might miss.
