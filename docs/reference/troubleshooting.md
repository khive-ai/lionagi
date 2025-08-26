# Troubleshooting Guide

Quick solutions to common LionAGI issues.

## Installation Issues

**Import Error: Module not found**

```python
# Error: ModuleNotFoundError: No module named 'lionagi'
pip install lionagi

# For latest development version
pip install git+https://github.com/khive-ai/lionagi.git
```

**Missing Optional Dependencies**

```python
# Error: docling not available
pip install lionagi[pdf]  # For PDF processing

# Error: matplotlib not available  
pip install matplotlib

# Error: networkx not available
pip install networkx
```

**Python Version Issues**

```bash
# LionAGI requires Python 3.10+
python --version  # Check version
pip install lionagi  # Only works on 3.10+
```

## API Key Errors

**OpenAI Authentication**

```python
import os
from lionagi import Branch, iModel

# Set API key
os.environ["OPENAI_API_KEY"] = "your-key-here"

branch = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
```

**Multiple Providers**

```python
# Different ways to set keys
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# Or in iModel directly
model = iModel(provider="openai", model="gpt-4o-mini", api_key="your-key")
```

## Async/Await Problems

**Missing await**

```python
# ❌ Wrong - missing await
branch = Branch()
result = branch.communicate("Hello")  # Returns coroutine

# ✅ Correct
result = await branch.communicate("Hello")
```

**Running in Jupyter**

```python
# Jupyter handles async automatically
branch = Branch()
result = await branch.communicate("Hello")  # Works in Jupyter

# For scripts, use asyncio.run()
import asyncio

async def main():
    branch = Branch()
    result = await branch.communicate("Hello")
    return result

# ❌ Wrong in script
result = await main()

# ✅ Correct in script  
result = asyncio.run(main())
```

**Mixing sync/async**

```python
# ❌ Wrong - can't await in sync function
def sync_function():
    branch = Branch()
    return await branch.communicate("Hello")  # SyntaxError

# ✅ Correct - make function async
async def async_function():
    branch = Branch()
    return await branch.communicate("Hello")
```

## Performance Issues

**Slow Parallel Execution**

```python
# ❌ Slow - sequential execution
results = []
for topic in topics:
    result = await branch.communicate(f"Research {topic}")
    results.append(result)

# ✅ Fast - parallel execution
import asyncio

tasks = [branch.communicate(f"Research {topic}") for topic in topics]
results = await asyncio.gather(*tasks)
```

**Graph vs Direct Calls**

```python
# Use graphs for complex workflows
from lionagi import Session, Builder

session = Session()
builder = Builder()

# Parallel operations
for topic in topics:
    builder.add_operation("communicate", instruction=f"Research {topic}")

results = await session.flow(builder.get_graph())
```

**Model Rate Limits**

```python
# Add delays between requests
import asyncio

async def rate_limited_call():
    try:
        result = await branch.communicate("Hello")
        return result
    except Exception as e:
        if "rate limit" in str(e).lower():
            await asyncio.sleep(1)  # Wait 1 second
            return await branch.communicate("Hello")
        raise e
```

## Memory Issues

**Token Limit Exceeded**

```python
# ❌ Error: Context too long
long_message = "very long text..." * 1000
result = await branch.communicate(long_message)  # May fail

# ✅ Solution: Chunk large inputs
def chunk_text(text, chunk_size=4000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

chunks = chunk_text(long_message)
results = []
for chunk in chunks:
    result = await branch.communicate(f"Process this: {chunk}")
    results.append(result)
```

**Branch Memory Accumulation**

```python
# Branch remembers all messages
branch = Branch()
for i in range(1000):
    await branch.communicate(f"Message {i}")  # Memory keeps growing

# Solution: Create new branch when needed
def get_fresh_branch():
    return Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))

# Or clear messages (if implemented)
# branch.messages.clear()  # Check if available
```

## Graph Execution Issues

**Circular Dependencies**

```python
# ❌ Error: Circular dependency
node_a = builder.add_operation("communicate", depends_on=[node_b])
node_b = builder.add_operation("communicate", depends_on=[node_a])  # Error

# ✅ Solution: Linear dependencies
node_a = builder.add_operation("communicate", instruction="Step 1")
node_b = builder.add_operation("communicate", depends_on=[node_a], instruction="Step 2")
```

**Empty Graph Results**

```python
# Check graph execution
try:
    result = await session.flow(builder.get_graph())
    print("Graph result:", result)
    
    # Access specific nodes
    graph = builder.get_graph()
    for node_id, node in graph.internal_nodes.items():
        print(f"Node {node_id}: {node}")
        
except Exception as e:
    import traceback
    traceback.print_exc()
```

## Cost Tracking Issues

**Missing Cost Data**

```python
def get_costs(node_id, builder, session):
    try:
        graph = builder.get_graph()
        node = graph.internal_nodes[node_id]
        branch = session.get_branch(node.branch_id, None)
        
        if branch and len(branch.messages) > 0:
            msg = branch.messages[-1]
            if hasattr(msg, 'model_response'):
                return msg.model_response.get("total_cost_usd", 0)
    except Exception as e:
        print(f"Cost tracking error: {e}")
    return 0
```

## Error Diagnosis

**Enable Verbose Logging**

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific operations
result = await session.flow(builder.get_graph(), verbose=True)
```

**Debugging Graph State**

```python
# Check graph structure
graph = builder.get_graph()
print(f"Nodes: {len(graph.internal_nodes)}")
for node_id, node in graph.internal_nodes.items():
    print(f"  {node_id}: {node}")

# Check session state
print(f"Branches: {len(session.branches)}")
```

## Getting Help

**GitHub Issues**: Report bugs at
[khive-ai/lionagi/issues](https://github.com/khive-ai/lionagi/issues)

**Check Version**: `pip show lionagi` for installed version

**Minimal Reproduction**: Include minimal code that reproduces the issue
