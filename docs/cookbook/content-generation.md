# Content Generation Pipeline

Multi-stage content creation with drafting, review, and refinement using
parallel branches and graph workflows.

## Blog Post Pipeline

```python
import asyncio
from pydantic import BaseModel, Field
from lionagi import Branch, Session, Builder, iModel

class BlogOutline(BaseModel):
    title: str
    hook: str
    sections: list[str]
    target_audience: str
    tone: str

class BlogPost(BaseModel):
    title: str
    content: str
    meta_description: str = Field(max_length=160)
    tags: list[str]

async def generate_blog_post(topic: str) -> BlogPost:
    session = Session()
    builder = Builder("BlogPipeline")

    # Stage 1: Outline
    outliner = session.new_branch(
        system="Create detailed blog post outlines. Focus on structure and flow.",
        name="outliner",
    )
    outline_step = builder.add_operation(
        "communicate",
        branch=outliner,
        instruction=f"Create an outline for a blog post about: {topic}",
        response_format=BlogOutline,
    )

    # Stage 2: Draft (depends on outline)
    writer = session.new_branch(
        system=(
            "Write engaging blog posts from outlines. "
            "Use clear language, concrete examples, and smooth transitions."
        ),
        name="writer",
    )
    draft_step = builder.add_operation(
        "communicate",
        branch=writer,
        depends_on=[outline_step],
        instruction="Write the full blog post based on the outline above.",
        inherit_context=True,
    )

    # Stage 3: Edit (depends on draft)
    editor = session.new_branch(
        system=(
            "You are a senior editor. Improve clarity, fix errors, "
            "tighten prose, and ensure the piece is publication-ready."
        ),
        name="editor",
    )
    edit_step = builder.add_operation(
        "communicate",
        branch=editor,
        depends_on=[draft_step],
        instruction=(
            "Edit this blog post. Fix errors, improve clarity, "
            "tighten the prose. Return the final version."
        ),
        response_format=BlogPost,
        inherit_context=True,
    )

    result = await session.flow(builder.get_graph())
    return result["operation_results"][edit_step]

# Usage
post = asyncio.run(generate_blog_post("Why async Python matters for AI apps"))
print(f"Title: {post.title}")
print(f"Tags: {post.tags}")
print(f"Meta: {post.meta_description}")
print(f"\n{post.content[:500]}...")
```

## Parallel Review Pipeline

Multiple reviewers evaluate a draft simultaneously:

```python
class ReviewFeedback(BaseModel):
    strengths: list[str]
    issues: list[str]
    suggestions: list[str]
    score: float = Field(ge=0, le=10)

async def review_and_refine(draft: str) -> str:
    """Send a draft through parallel reviewers, then refine."""
    session = Session()
    builder = Builder("ReviewPipeline")

    # Parallel reviewers with different perspectives
    reviewers = {
        "technical": session.new_branch(
            system="Review for technical accuracy and correctness.",
        ),
        "readability": session.new_branch(
            system="Review for clarity, readability, and audience fit.",
        ),
        "seo": session.new_branch(
            system="Review for SEO, headlines, and web discoverability.",
        ),
    }

    review_nodes = []
    for perspective, branch in reviewers.items():
        node = builder.add_operation(
            "communicate",
            branch=branch,
            instruction=f"Review this draft from a {perspective} perspective:\n\n{draft}",
            response_format=ReviewFeedback,
        )
        review_nodes.append(node)

    # Aggregation: refine based on all reviews
    refiner = session.new_branch(
        system=(
            "You refine content based on reviewer feedback. "
            "Address all valid issues while preserving the author's voice."
        ),
    )
    refine_node = builder.add_aggregation(
        "communicate",
        branch=refiner,
        source_node_ids=review_nodes,
        instruction=(
            "Here is the original draft and reviewer feedback. "
            "Produce a refined version addressing all valid points.\n\n"
            f"Original:\n{draft}"
        ),
    )

    result = await session.flow(builder.get_graph())
    return result["operation_results"][refine_node]
```

## Email Campaign Generator

Generate variants and pick the best:

```python
class EmailVariant(BaseModel):
    subject_line: str
    preview_text: str = Field(max_length=90)
    body: str
    cta: str = Field(description="Call to action text")

async def generate_email_campaign(
    product: str,
    audience: str,
    num_variants: int = 3,
) -> list[EmailVariant]:
    """Generate multiple email variants in parallel."""
    branches = [
        Branch(
            chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
            system=f"Write marketing emails. Style: {style}",
        )
        for style in [
            "Direct and urgent — create FOMO",
            "Friendly and conversational — build rapport",
            "Data-driven and authoritative — cite benefits",
            "Story-driven — use a narrative hook",
        ][:num_variants]
    ]

    tasks = [
        branch.communicate(
            f"Write a marketing email for {product} targeting {audience}.",
            response_format=EmailVariant,
        )
        for branch in branches
    ]
    return await asyncio.gather(*tasks)

# Usage
variants = asyncio.run(generate_email_campaign(
    product="AI code review tool",
    audience="engineering managers",
    num_variants=3,
))
for i, v in enumerate(variants, 1):
    print(f"Variant {i}: {v.subject_line}")
    print(f"  CTA: {v.cta}\n")
```

## Documentation Generator

Generate docs from code with structured output:

```python
class FunctionDoc(BaseModel):
    name: str
    description: str
    parameters: list[dict]
    returns: str
    example: str
    notes: str | None = None

class ModuleDoc(BaseModel):
    module_name: str
    overview: str
    functions: list[FunctionDoc]
    usage_example: str

async def document_code(source_code: str, module_name: str) -> ModuleDoc:
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
        system=(
            "Generate clear, accurate API documentation from source code. "
            "Include practical examples for every function."
        ),
    )
    return await branch.communicate(
        f"Document this module:\n\n```python\n{source_code}\n```",
        response_format=ModuleDoc,
    )

# Usage
code = '''
def retry(func, attempts=3, delay=1.0):
    """Retry a function with exponential backoff."""
    ...

def batch(items, size=10):
    """Split items into batches of given size."""
    ...
'''

docs = asyncio.run(document_code(code, "utils"))
print(f"# {docs.module_name}\n\n{docs.overview}\n")
for fn in docs.functions:
    print(f"## {fn.name}\n{fn.description}\n")
```

## When to Use

**Perfect for:** Blog posts, documentation, email campaigns, social media
content, marketing copy, technical writing.

**Key patterns:**

- Use sequential `depends_on` for draft → review → refine pipelines
- Use parallel branches for multiple reviewers or style variants
- Use `add_aggregation()` to combine feedback into a refined output
- Use `response_format` to enforce structure (word counts, required fields)
- Use `inherit_context=True` to pass previous stage output to the next
