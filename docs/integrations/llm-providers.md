# LLM Provider Integration

Comprehensive guide to all supported LLM providers in LionAGI.

## OpenAI

### Setup

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or use a `.env` file:

```env
OPENAI_API_KEY=your-api-key-here
```

### Basic Usage

```python
from lionagi import Branch, iModel
import asyncio

async def main():
    # Using GPT-4o-mini (recommended for development)
    branch = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
    
    response = await branch.communicate(
        "Explain the difference between async and sync programming in Python"
    )
    
    print(response)

asyncio.run(main())
```

### Advanced Configuration

```python
from lionagi import Branch, iModel

# Custom OpenAI configuration
config = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 2000,
    "top_p": 0.9,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1
}

async def main():
    session = Session(
        imodel="gpt-4o-mini",
        **config
    )
    
    # Multi-turn conversation with memory
    await session.chat("I'm building a web scraper in Python.")
    response = await session.chat("What libraries should I use for handling JavaScript?")
    
    print(response)
```

### Streaming Responses

```python
from lionagi.session import Session

async def stream_chat():
    session = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
    
    async for chunk in session.chat_stream(
        "Write a short story about AI agents working together"
    ):
        print(chunk.content, end="", flush=True)
    
    print()  # New line at the end

asyncio.run(stream_chat())
```

### Function Calling

```python
from lionagi.session import Session
from lionagi.tools.base import Tool
from pydantic import Field
from typing import Dict, Any

class WeatherTool(Tool):
    """Get current weather for a location."""
    
    location: str = Field(description="City name")
    
    async def call(self) -> Dict[str, Any]:
        # Simulate API call
        return {
            "location": self.location,
            "temperature": "22°C",
            "condition": "Sunny",
            "humidity": "45%"
        }

async def function_calling_example():
    session = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
    session.register_tool(WeatherTool)
    
    response = await session.chat(
        "What's the weather like in Tokyo? Format it nicely."
    )
    
    print(response)

asyncio.run(function_calling_example())
```

### Troubleshooting

- **Rate Limits**: Use exponential backoff (built into LionAGI)
- **Token Limits**: Monitor token usage with `session.get_token_count()`
- **API Errors**: Check API key validity and billing status
- **Timeout Issues**: Increase timeout in session configuration

## Anthropic

### Setup

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Basic Usage

```python
from lionagi.session import Session

async def claude_example():
    # Using Claude 3.5 Sonnet
    session = Branch(chat_model=iModel(provider="anthropic", model="claude-3-5-sonnet-20241022"))
    
    response = await session.chat(
        "Analyze this code and suggest improvements:",
        context="""
        def process_data(data):
            result = []
            for item in data:
                if item > 0:
                    result.append(item * 2)
            return result
        """
    )
    
    print(response)

asyncio.run(claude_example())
```

### Long Context Processing

```python
from lionagi.session import Session
from lionagi.tools.file.reader import ReaderTool

async def long_document_analysis():
    session = Branch(chat_model=iModel(provider="anthropic", model="claude-3-5-sonnet-20241022"))
    
    # Claude excels at long context
    reader = ReaderTool()
    document = await reader.read("long_document.pdf")
    
    response = await session.chat(
        "Summarize the key points and identify any inconsistencies in this document:",
        context=document.content
    )
    
    print(response)
```

### Structured Output with Claude

```python
from lionagi.session import Session
from pydantic import BaseModel, Field
from typing import List

class CodeReview(BaseModel):
    overall_score: int = Field(description="Score from 1-10")
    strengths: List[str] = Field(description="Code strengths")
    weaknesses: List[str] = Field(description="Areas for improvement")
    recommendations: List[str] = Field(description="Specific recommendations")

async def structured_code_review():
    session = Branch(chat_model=iModel(provider="anthropic", model="claude-3-5-sonnet-20241022"))
    
    response = await session.instruct(
        instruction="Review this Python code and provide structured feedback",
        context="""
        class DataProcessor:
            def __init__(self, data):
                self.data = data
                self.processed = False
            
            def process(self):
                if not self.processed:
                    self.data = [x * 2 for x in self.data if x > 0]
                    self.processed = True
                return self.data
        """,
        response_format=CodeReview
    )
    
    print(f"Score: {response.overall_score}/10")
    print(f"Strengths: {response.strengths}")
    print(f"Recommendations: {response.recommendations}")

asyncio.run(structured_code_review())
```

### Troubleshooting

- **Content Policy**: Claude has strict content policies
- **API Limits**: Monitor usage through Anthropic console
- **Context Windows**: Claude 3.5 Sonnet supports up to 200K tokens

## Ollama (Local)

### Setup

Install Ollama and dependencies:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Install Python package
uv add lionagi[ollama]

# Pull a model
ollama pull llama3.2:3b
```

### Basic Usage

```python
from lionagi.session import Session

async def local_llama_example():
    # Local Llama model
    session = Session(
        imodel="llama3.2:3b",
        provider="ollama",
        base_url="http://localhost:11434"
    )
    
    response = await session.chat(
        "Explain how transformers work in machine learning"
    )
    
    print(response)

asyncio.run(local_llama_example())
```

### Model Management

```python
from lionagi.service.connections.providers.ollama_ import OllamaEndpoint
import ollama

async def manage_models():
    client = ollama.AsyncClient()
    
    # List available models
    models = await client.list()
    print("Available models:")
    for model in models['models']:
        print(f"- {model['name']} ({model['size']} bytes)")
    
    # Pull a new model
    await client.pull("codellama:7b")
    
    # Use the new model
    session = Session(
        imodel="codellama:7b",
        provider="ollama"
    )
    
    response = await session.chat(
        "Write a Python function to calculate Fibonacci numbers"
    )
    print(response)

asyncio.run(manage_models())
```

### Custom Model Configuration

```python
from lionagi.session import Session

# Fine-tuned local model settings
config = {
    "temperature": 0.1,  # Lower for code generation
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "num_ctx": 4096,  # Context window
    "num_predict": 1000,  # Max tokens to generate
}

async def custom_ollama_config():
    session = Session(
        imodel="codellama:7b",
        provider="ollama",
        **config
    )
    
    response = await session.chat(
        "Optimize this SQL query:",
        context="SELECT * FROM users WHERE age > 18 AND city = 'NYC' ORDER BY name"
    )
    
    print(response)
```

### Troubleshooting

- **Ollama Not Running**: Check service: `ollama serve`
- **Model Not Found**: Pull model first: `ollama pull model_name`
- **Memory Issues**: Use smaller models (3B vs 7B vs 13B parameters)
- **Slow Performance**: Ensure GPU drivers are installed for acceleration

## Google (Gemini)

### Setup

```bash
export GOOGLE_API_KEY="your-google-api-key"
# or
export GEMINI_API_KEY="your-gemini-api-key"
```

### Basic Usage

```python
from lionagi.session import Session

async def gemini_example():
    session = Session(
        imodel="gemini-1.5-flash",
        provider="google"
    )
    
    response = await session.chat(
        "Compare Python and Rust for system programming"
    )
    
    print(response)

asyncio.run(gemini_example())
```

### Multimodal Capabilities

```python
from lionagi.session import Session
from lionagi.tools.file.reader import ReaderTool

async def gemini_vision_example():
    session = Session(
        imodel="gemini-1.5-pro",
        provider="google"
    )
    
    # Image analysis
    reader = ReaderTool()
    image_data = await reader.read("diagram.png")
    
    response = await session.chat(
        "Describe what you see in this image and explain any technical concepts shown",
        attachments=[image_data]
    )
    
    print(response)
```

### Large Context Processing

```python
from lionagi.session import Session

async def large_context_analysis():
    # Gemini 1.5 Pro supports up to 2M tokens
    session = Session(
        imodel="gemini-1.5-pro",
        provider="google"
    )
    
    # Process entire codebase
    codebase = await load_entire_codebase()  # Your function
    
    response = await session.chat(
        "Analyze this entire codebase for security vulnerabilities and architectural issues",
        context=codebase
    )
    
    print(response)
```

### Troubleshooting

- **API Quota**: Check Google Cloud Console for quota limits
- **Safety Settings**: Adjust safety settings for content filtering
- **Regional Availability**: Gemini availability varies by region

## Custom Providers

### Adding New Providers

```python
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.service.connections.providers.types import ProviderConfig
from typing import Dict, Any

class CustomLLMEndpoint(Endpoint):
    """Custom LLM provider endpoint."""
    
    def __init__(self, api_key: str, base_url: str, **kwargs):
        config = EndpointConfig(
            name="custom_llm",
            provider="custom",
            base_url=base_url,
            endpoint="v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            **kwargs
        )
        super().__init__(config)
    
    async def call(self, messages: list, **kwargs) -> Dict[str, Any]:
        payload = {
            "messages": messages,
            "model": kwargs.get("model", "default"),
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
        
        response = await self.client.post(
            self.config.endpoint,
            json=payload
        )
        
        return response.json()

# Register the custom provider
from lionagi.service.manager import ServiceManager

class CustomLLMProvider:
    def __init__(self, api_key: str, base_url: str):
        self.endpoint = CustomLLMEndpoint(api_key, base_url)
    
    async def create_completion(self, messages: list, **kwargs):
        return await self.endpoint.call(messages, **kwargs)

# Usage
async def custom_provider_example():
    provider = CustomLLMProvider(
        api_key="your-custom-api-key",
        base_url="https://api.customllm.com"
    )
    
    # Register with LionAGI
    ServiceManager.register_provider("custom_llm", provider)
    
    # Use in session
    session = Session(
        imodel="custom-model-name",
        provider="custom_llm"
    )
    
    response = await session.chat("Hello from custom provider!")
    print(response)
```

### OpenAI-Compatible Providers

```python
from lionagi.session import Session

# Many providers are OpenAI-compatible
async def openai_compatible_providers():
    # Together AI
    session_together = Session(
        imodel="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        provider="openai",  # OpenAI-compatible
        base_url="https://api.together.xyz/v1",
        api_key="your-together-key"
    )
    
    # Groq
    session_groq = Session(
        imodel="llama-3.1-8b-instant",
        provider="openai",  # OpenAI-compatible
        base_url="https://api.groq.com/openai/v1",
        api_key="your-groq-key"
    )
    
    # Perplexity
    session_pplx = Session(
        imodel="llama-3.1-sonar-small-128k-online",
        provider="perplexity",
        api_key="your-perplexity-key"
    )
    
    # Use any of them
    response = await session_groq.chat("Fast inference test")
    print(response)

asyncio.run(openai_compatible_providers())
```

### Provider Configuration Templates

```python
# config/custom_providers.py
from typing import Dict, Any

CUSTOM_PROVIDER_CONFIGS = {
    "huggingface": {
        "base_url": "https://api-inference.huggingface.co/models",
        "headers": {
            "Authorization": "Bearer {api_key}",
            "Content-Type": "application/json"
        },
        "endpoint_template": "{model_name}",
        "request_format": "huggingface",
        "response_format": "huggingface"
    },
    
    "replicate": {
        "base_url": "https://api.replicate.com/v1",
        "headers": {
            "Authorization": "Token {api_key}",
            "Content-Type": "application/json"
        },
        "endpoint_template": "predictions",
        "request_format": "replicate",
        "response_format": "replicate"
    }
}

class ProviderFactory:
    @staticmethod
    def create_provider(provider_name: str, config: Dict[str, Any]):
        if provider_name in CUSTOM_PROVIDER_CONFIGS:
            template = CUSTOM_PROVIDER_CONFIGS[provider_name]
            # Merge template with user config
            final_config = {**template, **config}
            return CustomLLMProvider(**final_config)
        
        raise ValueError(f"Unknown provider: {provider_name}")
```

## Provider Comparison

### Performance Characteristics

| Provider              | Latency | Context | Strengths                 | Best For                |
| --------------------- | ------- | ------- | ------------------------- | ----------------------- |
| **GPT-4o-mini**       | Fast    | 128K    | Balanced, cost-effective  | General development     |
| **GPT-4**             | Medium  | 128K    | Reasoning, complex tasks  | Critical applications   |
| **Claude 3.5 Sonnet** | Medium  | 200K    | Code, analysis, safety    | Code review, analysis   |
| **Gemini 1.5 Pro**    | Medium  | 2M      | Multimodal, large context | Document processing     |
| **Ollama/Local**      | Varies  | Varies  | Privacy, no API costs     | Sensitive data, offline |

### Cost Optimization

```python
from lionagi.session import Session
from lionagi.service.token_calculator import TokenCalculator

async def cost_aware_routing():
    # Simple tasks → GPT-4o-mini
    simple_session = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
    
    # Complex reasoning → GPT-4
    complex_session = Branch(chat_model=iModel(provider="openai", model="gpt-4"))
    
    # Code tasks → Claude
    code_session = Branch(chat_model=iModel(provider="anthropic", model="claude-3-5-sonnet-20241022"))
    
    def route_by_complexity(task: str):
        # Simple heuristics
        if any(word in task.lower() for word in ["code", "debug", "review"]):
            return code_session
        elif any(word in task.lower() for word in ["analyze", "complex", "reasoning"]):
            return complex_session
        else:
            return simple_session
    
    # Route task to appropriate model
    task = "Fix this Python bug in my authentication system"
    session = route_by_complexity(task)
    
    response = await session.chat(task)
    return response
```

### Best Practices

1. **Model Selection**: Choose based on task complexity and cost requirements
2. **API Key Management**: Use environment variables, never hardcode keys
3. **Error Handling**: Implement retry logic with exponential backoff
4. **Rate Limiting**: Respect provider rate limits to avoid blocking
5. **Token Management**: Monitor token usage to control costs
6. **Caching**: Cache responses for identical queries
7. **Streaming**: Use streaming for long responses to improve UX
8. **Fallback Providers**: Have backup providers for high availability
