# Conditional Flows Pattern

Dynamic workflow paths based on runtime conditions and decision points.

## When to Use This Pattern

Use conditional flows when:

- Different inputs need different processing paths
- Decision points determine next steps
- Workflow adapts based on intermediate results
- Error conditions need special handling
- Quality gates determine continuation

## Basic Conditional Flow

```python
from lionagi import Branch, Session, Builder, iModel

session = Session()
builder = Builder("conditional_example")

# Classifier branch
classifier = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Classify requests as: question, task, or creative."
)

# Specialized handlers  
qa_expert = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Answer questions clearly and accurately."
)

task_expert = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Provide step-by-step task guidance."
)

creative_expert = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-sonnet-20240229"),
    system="Help with creative projects and brainstorming."
)

session.include_branches([classifier, qa_expert, task_expert, creative_expert])

user_input = "How do I learn Python programming?"

# Step 1: Classify the request
classify_op = builder.add_operation(
    "communicate",
    branch=classifier,
    instruction=f"Classify as 'question', 'task', or 'creative': {user_input}"
)

# Step 2: Route based on classification
result = await session.flow(builder.get_graph())
classification = result["operation_results"][classify_op]

# Route to appropriate handler
if "question" in classification.lower():
    handler = qa_expert
elif "task" in classification.lower():
    handler = task_expert
else:
    handler = creative_expert

# Execute chosen path
response = await handler.communicate(f"Handle this request: {user_input}")
```

## Quality Gate Pattern

Content processing with quality thresholds:

```python
# Content creation and review branches
creator = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-sonnet-20240229"),
    system="Create high-quality content."
)

reviewer = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Review content quality and provide scores 1-10."
)

editor = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-sonnet-20240229"),
    system="Improve and refine content."
)

topic = "The future of renewable energy"

# Stage 1: Create initial content
initial_content = await creator.communicate(f"Write article about: {topic}")

# Stage 2: Quality review 
quality_review = await reviewer.communicate(
    f"Rate this content 1-10: {initial_content}"
)

# Quality gate: improve if below threshold
if any(str(i) in quality_review for i in range(1, 7)):
    # Below threshold - improve content
    final_content = await editor.communicate(
        f"Improve based on review: {initial_content}\n\nReview: {quality_review}"
    )
else:
    # Quality acceptable
    final_content = initial_content
```

## Error Handling with Fallbacks

Graceful degradation when primary processing fails:

```python
# Main and fallback processors
processor = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-sonnet-20240229"),
    system="Process complex requests thoroughly."
)

fallback = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"), 
    system="Handle requests with simplified approaches."
)

# Process with fallback
try:
    # Try main processor first
    result = await processor.communicate(request)
except Exception:
    # Fallback to simpler processor
    try:
        result = await fallback.communicate(f"Simplified: {request}")
    except Exception:
        result = "Unable to process request"
```

## Best Practices

### Clear Decision Criteria

```python
# Good: Specific, measurable criteria
"Rate complexity as 'simple' (1-3 steps) or 'complex' (4+ steps)"

# Avoid: Vague criteria  
"Is this hard?"
```

### Explicit Routing Logic

```python
# Clear routing based on parsed results
if "urgent" in priority_assessment.lower():
    handler = urgent_processor
else:
    handler = standard_processor
```

### Always Have Fallbacks

```python
# Graceful degradation
try:
    result = await complex_processor.communicate(request)
except Exception:
    result = await simple_processor.communicate(f"Simplified: {request}")
```

## When to Use

**Perfect for:** Content routing, difficulty adaptation, quality control, error
recovery, resource optimization

**Key advantage:** Runtime decision-making optimizes processing paths and
handles varying input conditions intelligently.

Conditional flows create adaptive workflows that route processing based on
content, quality thresholds, and error conditions.
