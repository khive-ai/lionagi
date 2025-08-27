# Installation

!!! tip "Recommended: Using uv"
    LionAGI works best with [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management:
    
    ```bash
    uv add lionagi
    ```

!!! info "Alternative: pip"
    You can also install with pip, but uv is faster:
    
    ```bash
    pip install lionagi
    ```

## Set Your API Key

!!! warning "API Key Required"
    LionAGI needs an LLM provider API key to work. Choose one:

=== "OpenAI (Recommended)"
    ```bash
    export OPENAI_API_KEY=your-key-here
    ```

=== "Anthropic"
    ```bash
    export ANTHROPIC_API_KEY=your-key-here  
    ```

=== "Local (Ollama)"
    No API key needed - just install [Ollama](https://ollama.ai/)

## Test Installation

!!! success "Verify Everything Works"
    ```python
    from lionagi import Branch, iModel
    import asyncio
    
    async def test():
        agent = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
        response = await agent.communicate("Hello")
        print(f"LionAGI says: {response}")
    
    asyncio.run(test())
    ```

## Provider Options

```python
# OpenAI
iModel(provider="openai", model="gpt-4o-mini")

# Anthropic  
iModel(provider="anthropic", model="claude-3-5-sonnet-20241022")

# Local (Ollama)
iModel(provider="ollama", model="llama3")
```
