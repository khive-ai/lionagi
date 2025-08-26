# MCP Server Integration

Connecting LionAGI to Model Context Protocol servers for external capabilities.

## Memory MCP

```python
from lionagi import Branch
# Memory MCP is automatically available

# Save information to memory
branch = Branch(name="researcher")
await branch.communicate("Remember that user prefers concise responses")

# Memory is automatically saved via MCP
# Access with mcp__memory__search(), mcp__memory__save(), etc.
```

## External API Integration Pattern

```python
from lionagi import iModel

# Exa search integration
exa_model = iModel(
    provider="exa",
    endpoint="search",
    queue_capacity=5,
    capacity_refresh_time=1,
    invoke_with_endpoint=False,
)

async def research_with_external_api(query: str):
    """Use external MCP servers for enhanced research"""
    
    researcher = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="Research specialist with external search capabilities"
    )
    
    # Use external search API
    search_request = {
        "query": query,
        "num_results": 5,
        "include_domains": ["arxiv.org", "openai.com"],
        "is_cached": True
    }
    
    # Make API call
    api_call = exa_model.create_api_calling(**search_request)
    search_result = await exa_model.invoke(api_call)
    
    # Process with LionAGI
    analysis = await researcher.communicate(
        "Analyze search results and provide insights",
        context=search_result.response
    )
    
    return analysis

# Usage
insights = await research_with_external_api("AI safety research trends 2024")
```

## Custom MCP Server Pattern

```python
# Example: Custom knowledge base MCP
class KnowledgeBaseMCP:
    """Custom MCP server for domain knowledge"""
    
    def __init__(self, knowledge_db_path: str):
        self.db_path = knowledge_db_path
    
    async def query_knowledge(self, topic: str) -> str:
        """Query domain-specific knowledge base"""
        # Implementation would connect to actual knowledge base
        return f"Knowledge about {topic}: ..."
    
    async def add_knowledge(self, topic: str, content: str) -> bool:
        """Add new knowledge to base"""
        # Implementation would store in knowledge base
        return True

# Integration with LionAGI
knowledge_server = KnowledgeBaseMCP("./knowledge.db")

domain_expert = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    tools=[knowledge_server.query_knowledge, knowledge_server.add_knowledge],
    system="Domain expert with access to specialized knowledge base"
)

# Use custom MCP
result = await domain_expert.ReAct(
    instruct={"instruction": "Research quantum computing applications"},
    max_extensions=3
)
```

## MCP Communication Patterns

```python
import json
import asyncio

class MCPClient:
    """Generic MCP client for custom protocols"""
    
    def __init__(self, server_endpoint: str):
        self.endpoint = server_endpoint
    
    async def call_mcp_method(self, method: str, params: dict):
        """Generic MCP method call"""
        # Implementation would handle MCP protocol
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }
        
        # Mock response for example
        return {"result": f"MCP response for {method}"}

# Use MCP client in agents
mcp_client = MCPClient("http://localhost:8000/mcp")

async def mcp_enhanced_workflow():
    """Workflow using multiple MCP servers"""
    
    coordinator = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="Coordinates work using MCP servers"
    )
    
    # Call multiple MCP servers
    tasks = [
        mcp_client.call_mcp_method("search", {"query": "AI research"}),
        mcp_client.call_mcp_method("analyze", {"data": "research_data"}),
        mcp_client.call_mcp_method("summarize", {"content": "long_text"})
    ]
    
    # Execute MCP calls in parallel
    mcp_results = await asyncio.gather(*tasks)
    
    # Process results with LionAGI
    synthesis = await coordinator.communicate(
        "Synthesize MCP server results",
        context=mcp_results
    )
    
    return synthesis

# Usage
result = await mcp_enhanced_workflow()
```

## Resource Management

```python
from contextlib import asynccontextmanager

class MCPResourceManager:
    """Manage MCP server connections and resources"""
    
    def __init__(self):
        self.connections = {}
    
    @asynccontextmanager
    async def get_mcp_connection(self, server_name: str):
        """Get managed MCP connection"""
        try:
            if server_name not in self.connections:
                # Initialize connection
                self.connections[server_name] = MCPClient(f"http://{server_name}:8000")
            
            yield self.connections[server_name]
        except Exception as e:
            print(f"MCP connection error: {e}")
            raise
        finally:
            # Cleanup if needed
            pass

# Usage in production
mcp_manager = MCPResourceManager()

async def production_mcp_workflow():
    """Production workflow with managed MCP resources"""
    
    async with mcp_manager.get_mcp_connection("memory_server") as memory_mcp:
        async with mcp_manager.get_mcp_connection("search_server") as search_mcp:
            # Use multiple MCP servers safely
            memory_result = await memory_mcp.call_mcp_method("recall", {"query": "user_preferences"})
            search_result = await search_mcp.call_mcp_method("search", {"query": "latest_ai_news"})
            
            # Process with LionAGI
            processor = Branch(
                chat_model=iModel(provider="openai", model="gpt-4o-mini")
            )
            
            return await processor.communicate(
                "Process MCP results",
                context={"memory": memory_result, "search": search_result}
            )
```

## Error Handling and Retry

```python
async def robust_mcp_call(mcp_client, method: str, params: dict, max_retries: int = 3):
    """Robust MCP call with retry logic"""
    
    for attempt in range(max_retries):
        try:
            result = await mcp_client.call_mcp_method(method, params)
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"MCP call failed after {max_retries} attempts: {e}")
                return None
            
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)
    
    return None

# Usage in workflows
async def fault_tolerant_workflow():
    """Workflow with MCP fault tolerance"""
    
    mcp_client = MCPClient("http://unreliable-server:8000")
    
    # Robust MCP calls
    results = await asyncio.gather(
        robust_mcp_call(mcp_client, "method1", {"param": "value1"}),
        robust_mcp_call(mcp_client, "method2", {"param": "value2"}),
        return_exceptions=True
    )
    
    # Filter successful results
    successful_results = [r for r in results if r is not None]
    
    # Process with LionAGI even if some MCP calls failed
    if successful_results:
        processor = Branch(
            chat_model=iModel(provider="openai", model="gpt-4o-mini")
        )
        
        return await processor.communicate(
            "Process available MCP results",
            context=successful_results
        )
    
    return "No MCP results available"
```

## Best Practices

**Connection Management:**

- Use connection pooling for high-throughput scenarios
- Implement proper timeout handling
- Cache MCP responses when appropriate

**Error Handling:**

- Implement retry logic with exponential backoff
- Graceful degradation when MCP servers unavailable
- Comprehensive logging for debugging

**Performance:**

- Parallel MCP calls when possible
- Optimize payload sizes for network efficiency
- Monitor MCP server response times

**Security:**

- Validate all MCP server responses
- Use secure connections (TLS) for production
- Implement proper authentication mechanisms
