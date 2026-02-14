# Multi-Provider Evaluation

Compare LLM outputs across providers using the same prompts. Useful for
benchmarking, selecting models, and building consensus from diverse sources.

## Side-by-Side Comparison

```python
import asyncio
from pydantic import BaseModel, Field
from lionagi import Branch, iModel

models = {
    "gpt-4.1-mini": iModel(provider="openai", model="gpt-4.1-mini"),
    "claude-sonnet": iModel(provider="anthropic", model="claude-sonnet-4-20250514"),
    "gemini-flash": iModel(provider="gemini", model="gemini-2.5-flash"),
}

async def compare(prompt: str) -> dict[str, str]:
    """Run the same prompt across multiple providers."""
    async def run_one(name: str, model: iModel) -> tuple[str, str]:
        branch = Branch(chat_model=model)
        result = await branch.communicate(prompt)
        return name, result

    pairs = await asyncio.gather(*[
        run_one(name, model) for name, model in models.items()
    ])
    return dict(pairs)

# Usage
results = asyncio.run(compare("Explain quantum entanglement in 2 sentences."))
for name, answer in results.items():
    print(f"[{name}]\n{answer}\n")
```

## Structured Evaluation with Scoring

Use a judge model to score each response:

```python
class ResponseScore(BaseModel):
    accuracy: float = Field(ge=0, le=10)
    clarity: float = Field(ge=0, le=10)
    completeness: float = Field(ge=0, le=10)
    overall: float = Field(ge=0, le=10)
    reasoning: str

class EvalResult(BaseModel):
    model: str
    response: str
    score: ResponseScore

async def evaluate_models(
    prompt: str,
    judge_model: iModel = None,
) -> list[EvalResult]:
    """Compare models with a judge scoring each response."""
    judge_model = judge_model or iModel(provider="openai", model="gpt-4.1")

    # Step 1: Get responses from all models
    responses = await compare(prompt)

    # Step 2: Score each response with the judge
    judge = Branch(
        chat_model=judge_model,
        system=(
            "You are an impartial evaluator. Score responses on accuracy, "
            "clarity, completeness, and overall quality (0-10). Be specific "
            "in your reasoning."
        ),
    )

    results = []
    for name, response in responses.items():
        score = await judge.communicate(
            f"Original prompt: {prompt}\n\n"
            f"Response to evaluate:\n{response}",
            response_format=ResponseScore,
        )
        results.append(EvalResult(model=name, response=response, score=score))

    # Sort by overall score
    results.sort(key=lambda r: r.score.overall, reverse=True)
    return results

# Usage
evals = asyncio.run(evaluate_models(
    "What are the trade-offs between microservices and monoliths?"
))
for e in evals:
    print(f"{e.model}: {e.score.overall}/10 — {e.score.reasoning[:80]}...")
```

## Consensus Building

Aggregate diverse model outputs into a consensus answer:

```python
async def consensus(prompt: str) -> str:
    """Get responses from multiple models, then synthesize a consensus."""
    responses = await compare(prompt)

    synthesizer = Branch(
        chat_model=iModel(provider="openai", model="gpt-4.1"),
        system=(
            "You synthesize multiple AI responses into a single, "
            "high-quality answer. Combine the best parts, resolve "
            "contradictions, and note disagreements."
        ),
    )

    formatted = "\n\n".join(
        f"**{name}:**\n{resp}" for name, resp in responses.items()
    )

    result = await synthesizer.communicate(
        f"Prompt: {prompt}\n\nResponses:\n{formatted}\n\n"
        "Synthesize a consensus answer."
    )
    return result

# Usage
answer = asyncio.run(consensus("Best practices for API rate limiting?"))
print(answer)
```

## Tournament Evaluation

Run a bracket-style tournament where models compete head-to-head:

```python
class MatchResult(BaseModel):
    winner: str
    reasoning: str
    margin: float = Field(ge=0, le=10, description="How decisive the win was")

async def head_to_head(
    prompt: str,
    model_a: tuple[str, str],
    model_b: tuple[str, str],
) -> MatchResult:
    """Compare two responses head-to-head."""
    name_a, resp_a = model_a
    name_b, resp_b = model_b

    judge = Branch(
        chat_model=iModel(provider="openai", model="gpt-4.1"),
        system="Compare two responses. Pick the better one and explain why.",
    )

    result = await judge.communicate(
        f"Prompt: {prompt}\n\n"
        f"Response A ({name_a}):\n{resp_a}\n\n"
        f"Response B ({name_b}):\n{resp_b}\n\n"
        f"Which is better? Set winner to '{name_a}' or '{name_b}'.",
        response_format=MatchResult,
    )
    return result

async def tournament(prompt: str) -> list[MatchResult]:
    """Round-robin tournament across all configured models."""
    responses = await compare(prompt)
    names = list(responses.keys())
    matches = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            result = await head_to_head(
                prompt,
                (names[i], responses[names[i]]),
                (names[j], responses[names[j]]),
            )
            matches.append(result)

    return matches

# Usage
matches = asyncio.run(tournament("Explain the CAP theorem with examples."))
for m in matches:
    print(f"Winner: {m.winner} (margin: {m.margin}) — {m.reasoning[:60]}...")
```

## Fastest Response

Use `race()` to get the first response from any provider:

```python
from lionagi.ln.concurrency import race

async def fastest_answer(prompt: str) -> str:
    """Return whichever provider responds first."""
    branches = [
        Branch(chat_model=iModel(provider="openai", model="gpt-4.1-mini")),
        Branch(chat_model=iModel(provider="anthropic", model="claude-sonnet-4-20250514")),
        Branch(chat_model=iModel(provider="gemini", model="gemini-2.5-flash")),
    ]
    result = await race(*[b.communicate(prompt) for b in branches])
    return result
```

## When to Use

**Perfect for:** Model selection, quality assurance, A/B testing prompts,
building consensus from diverse AI perspectives, latency benchmarking.

**Key patterns:**

- Use `compare()` for quick side-by-side output
- Use `evaluate_models()` with a judge for systematic scoring
- Use `consensus()` when you want one high-quality answer from many sources
- Use `race()` when latency matters more than choosing a specific provider
