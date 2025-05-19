# Quick Start

This guide will help you get started with Pynector quickly, showing you how to
set up a basic client and make requests.

## Basic Usage

```python
import asyncio
from pynector.client import Client
from pynector.config import ClientConfig

async def main():
    # Create a client configuration
    config = ClientConfig(
        base_url="https://api.example.com",
        timeout=30.0,  # 30 seconds timeout
    )
    
    # Initialize a client
    client = Client(config)
    
    # Make a request
    response = await client.get("/endpoint")
    
    # Process the response
    if response.is_success:
        data = response.json()
        print(f"Received data: {data}")
    else:
        print(f"Request failed with status: {response.status_code}")
        print(f"Error: {response.text}")
    
    # Don't forget to close the client when done
    await client.close()

# Run the async main function
asyncio.run(main())
```

## Using HTTP Transport

Pynector uses HTTP transport by default, but you can explicitly configure it:

```python
from pynector.client import Client
from pynector.config import ClientConfig
from pynector.transport.http import HTTPTransport

async def main():
    # Create a client with HTTP transport explicitly
    config = ClientConfig(
        base_url="https://api.example.com",
        transport=HTTPTransport(),
    )
    
    client = Client(config)
    # ... use client as before
```

## Working with SDK Adapters

If you're working with APIs that have Python SDKs, you can use Pynector's SDK
transport:

```python
from pynector.client import Client
from pynector.config import ClientConfig
from pynector.transport.sdk import SDKTransport
from pynector.transport.sdk.adapter import OpenAIAdapter

async def main():
    # Create an SDK transport with the OpenAI adapter
    transport = SDKTransport(adapter=OpenAIAdapter(api_key="your-api-key"))
    
    # Create client configuration
    config = ClientConfig(
        transport=transport,
    )
    
    client = Client(config)
    
    # Now you can use the client with the SDK adapter
    response = await client.post(
        "/completions", 
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello, AI!"}]
        }
    )
    
    if response.is_success:
        completion = response.json()
        print(f"AI response: {completion['choices'][0]['message']['content']}")
```

## Error Handling

Pynector provides comprehensive error handling:

```python
from pynector.client import Client
from pynector.config import ClientConfig
from pynector.errors import RequestError, ResponseError

async def main():
    client = Client(ClientConfig(base_url="https://api.example.com"))
    
    try:
        response = await client.get("/endpoint")
        data = response.json()
        
    except RequestError as e:
        # Handle connection errors, timeouts, etc.
        print(f"Request failed: {e}")
        
    except ResponseError as e:
        # Handle server errors, bad responses, etc.
        print(f"Response error: {e}")
        print(f"Status code: {e.status_code}")
        
    except Exception as e:
        # Handle other errors
        print(f"Unexpected error: {e}")
```

## Advanced Concurrency

Pynector supports various concurrency patterns:

```python
from pynector.client import Client
from pynector.config import ClientConfig
from pynector.concurrency.patterns import gather, race

async def main():
    client = Client(ClientConfig(base_url="https://api.example.com"))
    
    # Make multiple requests in parallel
    responses = await gather(
        client.get("/endpoint1"),
        client.get("/endpoint2"),
        client.get("/endpoint3"),
    )
    
    # Use the first response that completes (race pattern)
    first_response = await race(
        client.get("/fast-but-unreliable"),
        client.get("/slow-but-reliable"),
    )
```

For more detailed information, check out the following sections:

- [Client Documentation](../client.md)
- [Transport Overview](../transport.md)
- [Concurrency Patterns](../concurrency.md)
- [Observability Features](../observability.md)
