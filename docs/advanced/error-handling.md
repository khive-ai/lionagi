# Error Handling

Handle failures gracefully in LionAGI workflows.

## Basic Error Handling

Catch and handle common errors:

```python
from lionagi import Branch, iModel

try:
    # Invalid model configuration
    branch = Branch(
        chat_model=iModel(provider="invalid", model="gpt-4")
    )
    result = await branch.communicate("Analyze market trends")
    
except Exception as e:
    print(f"Error: {e}")
    # Fallback to working model
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4")
    )
    result = await branch.communicate("Analyze market trends")
```

## Retry Pattern

Retry failed operations with backoff:

```python
import asyncio
from lionagi import Branch, iModel

branch = Branch(chat_model=iModel(provider="openai", model="gpt-4"))

async def safe_communicate(prompt: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return await branch.communicate(prompt)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

result = await safe_communicate("Analyze market trends")
```

## Fallback Pattern

Try multiple models as fallbacks:

```python
from lionagi import Branch, iModel

models = [
    {"provider": "openai", "model": "gpt-4"},
    {"provider": "anthropic", "model": "claude-3-5-sonnet-20240620"}
]

prompt = "What are current AI trends?"

for i, config in enumerate(models):
    try:
        branch = Branch(chat_model=iModel(**config))
        result = await branch.communicate(prompt)
        print(f"✓ Success with model {i+1}")
        break
    except Exception as e:
        print(f"✗ Model {i+1} failed: {e}")
        if i == len(models) - 1:
            raise Exception("All models failed")
```

## Workflow Fallbacks

Simplify workflows when complex ones fail:

```python
from lionagi import Session, Builder, Branch, iModel

async def complex_workflow():
    session = Session()
    builder = Builder("analysis")
    
    branch = Branch(chat_model=iModel(provider="openai", model="gpt-4"))
    session.include_branches([branch])
    
    step1 = builder.add_operation(
        "communicate", branch=branch,
        instruction="Research market trends"
    )
    step2 = builder.add_operation(
        "communicate", branch=branch,
        instruction="Analyze competition",
        depends_on=[step1]
    )
    
    return await session.flow(builder.get_graph())

async def simple_workflow():
    branch = Branch(chat_model=iModel(provider="openai", model="gpt-4"))
    result = await branch.communicate("Provide market and competitive analysis")
    return {"operation_results": {"analysis": result}}

# Try complex, fallback to simple
try:
    result = await complex_workflow()
except Exception as e:
    print(f"Complex workflow failed: {e}")
    result = await simple_workflow()
```

## Partial Results

Accept partial success when some operations fail:

```python
from lionagi import Branch, iModel

branch = Branch(chat_model=iModel(provider="openai", model="gpt-4"))

tasks = ["market analysis", "competitor research", "risk assessment"]
results = []

for task in tasks:
    try:
        result = await branch.communicate(f"Brief {task}")
        results.append({"task": task, "result": result, "status": "success"})
        print(f"✓ {task} completed")
    except Exception as e:
        results.append({"task": task, "error": str(e), "status": "failed"})
        print(f"✗ {task} failed: {e}")

# Use partial results if sufficient
successful = [r for r in results if r["status"] == "success"]
if len(successful) >= 2:
    print(f"Using {len(successful)} successful results")
else:
    print("Too many failures to proceed")
```

## Error Tracking

Track errors for monitoring:

```python
from lionagi import Branch, iModel

branch = Branch(chat_model=iModel(provider="openai", model="gpt-4"))

operations = ["market analysis", "competitor research", "risk assessment"]
errors = []
successes = 0

for operation in operations:
    try:
        result = await branch.communicate(f"Brief {operation}")
        successes += 1
        print(f"✓ {operation} completed")
    except Exception as e:
        errors.append({"operation": operation, "error": str(e)})
        print(f"✗ {operation} failed: {e}")

total = len(operations)
error_rate = len(errors) / total * 100
print(f"Results: {successes}/{total} successful ({error_rate:.1f}% error rate)")
```

## Best Practices

**Handle errors at appropriate levels**:

```python
# Operation level: basic try/catch
try:
    result = await branch.communicate(instruction)
except Exception as e:
    print(f"Operation failed: {e}")

# Workflow level: fallbacks
try:
    result = await complex_workflow()
except Exception:
    result = await simple_fallback()
```

**Use simple retry logic**:

```python
for attempt in range(3):
    try:
        return await operation()
    except Exception as e:
        if attempt == 2:
            raise
        await asyncio.sleep(2 ** attempt)
```

**Accept partial results**:

```python
results = []
for task in tasks:
    try:
        result = await process_task(task)
        results.append(result)
    except Exception:
        continue

if len(results) >= minimum_required:
    return results
```

Error handling focuses on practical patterns: basic try/catch blocks, retry
logic, fallback strategies, and accepting partial results.
