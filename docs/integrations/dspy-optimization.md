# DSPy Integration

lionagi does not have a native DSPy integration. The two frameworks solve different problems: DSPy focuses on prompt optimization and compilation, while lionagi focuses on multi-model orchestration and tool calling.

## Using DSPy with lionagi

You can wrap DSPy modules as lionagi tools or call them within your application code alongside lionagi branches:

```python
import dspy
from lionagi import Branch

# Your DSPy module
class Summarizer(dspy.Signature):
    text = dspy.InputField()
    summary = dspy.OutputField()

summarizer = dspy.ChainOfThought(Summarizer)

# Wrap as a lionagi tool
def dspy_summarize(text: str) -> str:
    """Summarize text using an optimized DSPy module.

    Args:
        text: The text to summarize.
    """
    result = summarizer(text=text)
    return result.summary

branch = Branch(tools=[dspy_summarize])
```

Both frameworks can share the same LLM API keys and run in the same process.
