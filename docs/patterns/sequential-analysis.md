# Sequential Analysis Pattern

Build complex understanding step-by-step through dependent operations.

## When to Use This Pattern

Use sequential analysis when:

- Each step builds upon previous findings
- Processing requires logical progression
- Context accumulation improves quality
- Complex documents need structured analysis

## Basic Pattern Structure

```python
from lionagi import Branch, Session, Builder, iModel

session = Session()
builder = Builder("document_analysis")

# Create analyzer branch
analyzer = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="You are a document analysis expert."
)
session.include_branches([analyzer])

document = "Your document content here..."

# Step 1: Extract key topics
extract_topics = builder.add_operation(
    "communicate",
    branch=analyzer,
    instruction=f"Extract 3-5 key topics from this document: {document}"
)

# Step 2: Analyze each topic (depends on step 1)
analyze_topics = builder.add_operation(
    "communicate",
    branch=analyzer,
    instruction="For each topic, provide detailed analysis",
    depends_on=[extract_topics]
)

# Step 3: Synthesize insights (depends on step 2)
synthesize = builder.add_operation(
    "communicate",
    branch=analyzer,
    instruction="What are the 3 most important insights?",
    depends_on=[analyze_topics]
)

# Execute the sequential workflow
result = await session.flow(builder.get_graph())
```

## Multi-Step Analysis

Research paper analysis with sequential dependency:

```python
# Specialized research analyzer
researcher = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-sonnet-20240229"),
    system="Research analyst specializing in academic papers."
)

paper_text = "Your research paper content..."

# Step 1: Structure identification
identify_structure = builder.add_operation(
    "communicate",
    branch=researcher,
    instruction=f"Identify and summarize each section: {paper_text}"
)

# Step 2: Technical analysis
analyze_technical = builder.add_operation(
    "communicate",
    branch=researcher,
    instruction="Analyze technical contributions and methodology",
    depends_on=[identify_structure]
)

# Step 3: Evaluate novelty
evaluate_novelty = builder.add_operation(
    "communicate",
    branch=researcher,
    instruction="Assess novelty and significance of contributions",
    depends_on=[analyze_technical]
)

# Step 4: Final assessment
final_assessment = builder.add_operation(
    "communicate",
    branch=researcher,
    instruction="Provide comprehensive evaluation",
    depends_on=[evaluate_novelty]
)

result = await session.flow(builder.get_graph())
```

## Context Building

Each step accumulates context for deeper analysis:

```python
investigator = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Thorough investigator building understanding incrementally."
)

# Sequential investigation steps
observe = builder.add_operation(
    "communicate",
    branch=investigator,
    instruction="Make initial observations about the data"
)

hypothesize = builder.add_operation(
    "communicate", 
    branch=investigator,
    instruction="Generate 3 hypotheses based on observations",
    depends_on=[observe]
)

analyze = builder.add_operation(
    "communicate",
    branch=investigator, 
    instruction="Analyze each hypothesis for evidence",
    depends_on=[hypothesize]
)

conclude = builder.add_operation(
    "communicate",
    branch=investigator,
    instruction="Draw conclusions with confidence levels",
    depends_on=[analyze]
)

result = await session.flow(builder.get_graph())
```

## Best Practices

### Clear Dependencies

```python
# Good: Clear progression
step1 = builder.add_operation("communicate", instruction="Extract facts")
step2 = builder.add_operation("communicate", instruction="Analyze facts", depends_on=[step1])
step3 = builder.add_operation("communicate", instruction="Draw conclusions", depends_on=[step2])
```

### Consistent Context

```python
# Use same branch for context continuity
analyzer = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Maintain context across analysis steps."
)
```

### Quality Assessment

```python
# Include data assessment as first step
assess_data = builder.add_operation(
    "communicate",
    instruction="Assess data quality and identify limitations"
)
```

## When to Use

**Perfect for:** Document analysis, research workflows, investigations, decision
making, problem solving

**Key advantage:** Each step builds meaningfully on previous work, leading to
more thorough and accurate results than parallel analysis.

Sequential analysis creates structured understanding through logical progression
and context accumulation.
