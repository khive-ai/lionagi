# Your First Flow

This quickstart shows LionAGI's core pattern: **multiple specialized branches working together**.

We'll build a self-critiquing joke generator - one branch writes jokes, another critiques them, then they iterate to improve.

## Complete Example

```python
from lionagi import Branch, iModel

chat_model = iModel(provider="openai", model="gpt-4.1-mini")

async def generate_joke(joke_request):
    # Create two specialized branches
    comedian = Branch(
        chat_model=chat_model,
        system="You are a comedian who makes technical concepts funny."
    )

    editor = Branch(
        chat_model=chat_model,
        system="You are an editor who improves clarity and punch."
    )

    # Flow: Generate → Critique → Revise → Verify
    joke = await comedian.communicate(
        "Write a short joke",
        context={"user_input": joke_request}
    )

    feedback = await editor.communicate(
        "Give humorous critical feedback to improve this joke by addressing clarity and punch.",
        context={"joke": joke}
    )

    revision = await comedian.communicate(
        "Revise the joke based on this feedback.",
        context={"feedback": feedback}
    )

    final_check = await editor.communicate(
        "Is this version better than the original? If yes, reply the following token (including the square brackets) '[YES]' only, otherwise elaborate on why not without including the token.",
        context={"revised_joke": revision}
    )

    # Return improved version if approved
    if "[yes]" in final_check.lower():
        return revision
    return joke  # Fallback to original if not improved

# Run it
if __name__ == "__main__":
    import anyio
    result = anyio.run(generate_joke, "machine learning")
    print(result)
```

## What Just Happened

**Branch**: Independent conversation context with its own system prompt and memory
- `comedian` = creative joke writing
- `editor` = critical feedback

**Flow**: Sequential operations across branches
1. Comedian generates initial joke
2. Editor provides feedback
3. Comedian revises based on feedback
4. Editor validates improvement

**Context**: Each `communicate()` call can include context from previous steps

## Try It

Save the code above and run:
```bash
uv run your_script.py
```

Expected output: A refined joke about machine learning, improved through iteration.

## Next Steps

- **Structured Output**: [Cheat Sheet](cheat-sheet.md#structured-output) - Get Pydantic models instead of strings
- **Tools**: [Cheat Sheet](cheat-sheet.md#tools) - Add web search, code execution, custom functions
- **Tutorials**: [L1 Tutorial](../tutorials/L1-your-first-agent.md) - Deep dive into Branch fundamentals
