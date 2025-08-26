# Tool Integration

Built-in and custom tools for extending agent capabilities.

## Basic Tool Creation

```python
from lionagi import Branch, iModel

# Simple function becomes a tool
def calculate_sum(a: float, b: float) -> float:
    """Add two numbers together"""
    return a + b

def search_web(query: str) -> str:
    """Search the web for information"""
    return f"Web search results for: {query}"

# Create tool-enabled branch (direct function passing)
agent = Branch(
    tools=[calculate_sum, search_web],
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="You have access to calculation and web search tools"
)

# Usage
result = await agent.ReAct(
    instruct={"instruction": "Search for Python tutorials and calculate 15 + 27"},
    max_extensions=3
)
```

## ReaderTool Integration

```python
from lionagi.tools.types import ReaderTool

# Document analysis agent
doc_analyzer = Branch(
    tools=[ReaderTool],
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Document analysis specialist with reading capabilities"
)

# Analyze documents with ReaderTool
analysis = await doc_analyzer.ReAct(
    instruct={
        "instruction": "Read and analyze the document, extract key findings",
        "context": {"document_path": "research_paper.pdf"}
    },
    tools=["reader_tool"],
    max_extensions=4,
    verbose=True
)
```

## API Integration Tools

```python
import httpx
from typing import Dict, Any

def make_api_request(url: str, method: str = "GET", data: Dict[str, Any] = None) -> str:
    """Make HTTP API requests"""
    try:
        with httpx.Client() as client:
            if method.upper() == "POST":
                response = client.post(url, json=data)
            else:
                response = client.get(url, params=data)
            
            response.raise_for_status()
            return response.text
    except Exception as e:
        return f"API request failed: {e}"

# API-enabled agent
api_agent = Branch(
    tools=[make_api_request],
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="API integration specialist with HTTP request capabilities"
)

# Usage
result = await api_agent.ReAct(
    instruct={"instruction": "Fetch weather data from api.weather.com"},
    max_extensions=2
)
```

## Database Tool Integration

```python
import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db_connection(db_path: str):
    """Database connection context manager"""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

def query_database(query: str, db_path: str = "database.db") -> str:
    """Execute SQL query and return results"""
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            return str(results)
    except Exception as e:
        return f"Database error: {e}"

def insert_data(table: str, data: Dict[str, Any], db_path: str = "database.db") -> str:
    """Insert data into database table"""
    try:
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, list(data.values()))
            conn.commit()
            return f"Inserted data into {table}"
    except Exception as e:
        return f"Insert error: {e}"

# Database agent
db_agent = Branch(
    tools=[query_database, insert_data],
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Database specialist with SQL query and data insertion capabilities"
)

# Usage
result = await db_agent.ReAct(
    instruct={"instruction": "Query user table and add a new user record"},
    max_extensions=3
)
```

## Code Execution Tools

```python
import subprocess
import tempfile
import os

def execute_python_code(code: str) -> str:
    """Safely execute Python code in isolated environment"""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            
            # Execute with timeout
            result = subprocess.run(
                ['python', f.name],
                capture_output=True,
                text=True,
                timeout=10  # 10 second timeout
            )
            
            # Cleanup
            os.unlink(f.name)
            
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error: {result.stderr}"
                
    except subprocess.TimeoutExpired:
        return "Code execution timeout"
    except Exception as e:
        return f"Execution error: {e}"

def validate_code_syntax(code: str) -> str:
    """Validate Python code syntax without execution"""
    try:
        compile(code, '<string>', 'exec')
        return "Syntax is valid"
    except SyntaxError as e:
        return f"Syntax error: {e}"

# Code execution agent
code_agent = Branch(
    tools=[execute_python_code, validate_code_syntax],
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Code execution specialist with Python runtime capabilities"
)

# Usage
result = await code_agent.ReAct(
    instruct={"instruction": "Write and execute code to calculate fibonacci sequence"},
    max_extensions=3
)
```

## File System Tools

```python
import os
import json
from pathlib import Path

def read_file(file_path: str) -> str:
    """Read content from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(file_path: str, content: str) -> str:
    """Write content to file"""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"

def list_directory(dir_path: str) -> str:
    """List contents of directory"""
    try:
        items = os.listdir(dir_path)
        return json.dumps(items, indent=2)
    except Exception as e:
        return f"Error listing directory: {e}"

# File system agent
fs_agent = Branch(
    tools=[read_file, write_file, list_directory],
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="File system specialist with read/write capabilities"
)

# Usage
result = await fs_agent.ReAct(
    instruct={"instruction": "Create a project structure with README and config files"},
    max_extensions=4
)
```

## Custom Tool Factory

```python
from typing import Callable, Any
import inspect

class ToolFactory:
    """Factory for creating standardized tools"""
    
    @staticmethod
    def create_api_tool(base_url: str, api_key: str = None) -> Callable:
        """Create API tool with base URL and authentication"""
        
        def api_tool(endpoint: str, method: str = "GET", data: Dict = None) -> str:
            """Generated API tool"""
            url = f"{base_url}/{endpoint.lstrip('/')}"
            headers = {}
            
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            try:
                with httpx.Client() as client:
                    response = client.request(method, url, json=data, headers=headers)
                    response.raise_for_status()
                    return response.text
            except Exception as e:
                return f"API error: {e}"
        
        return api_tool
    
    @staticmethod
    def create_validation_tool(validation_func: Callable) -> Callable:
        """Create validation tool from validation function"""
        
        def validation_tool(data: str) -> str:
            """Generated validation tool"""
            try:
                is_valid = validation_func(data)
                return f"Validation result: {'Valid' if is_valid else 'Invalid'}"
            except Exception as e:
                return f"Validation error: {e}"
        
        return validation_tool
    
    @staticmethod
    def create_processing_tool(processor_func: Callable) -> Callable:
        """Create data processing tool"""
        
        def processing_tool(input_data: str) -> str:
            """Generated processing tool"""
            try:
                result = processor_func(input_data)
                return str(result)
            except Exception as e:
                return f"Processing error: {e}"
        
        return processing_tool

# Usage example
def validate_email(email: str) -> bool:
    """Email validation function"""
    return "@" in email and "." in email

def process_text(text: str) -> str:
    """Text processing function"""
    return text.upper().replace(" ", "_")

# Create custom tools
email_validator = ToolFactory.create_validation_tool(validate_email)
text_processor = ToolFactory.create_processing_tool(process_text)
api_tool = ToolFactory.create_api_tool("https://api.example.com", "your_api_key")

# Use in agent
custom_agent = Branch(
    tools=[email_validator, text_processor, api_tool],
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Multi-purpose agent with custom tools"
)
```

## Tool Composition Patterns

```python
from typing import List
import asyncio

class ToolComposer:
    """Compose multiple tools into workflows"""
    
    def __init__(self, tools: List[Callable]):
        self.tools = {tool.__name__: tool for tool in tools}
    
    def create_pipeline_tool(self, tool_names: List[str]) -> Callable:
        """Create a pipeline of tools"""
        
        def pipeline_tool(input_data: str) -> str:
            """Execute tools in sequence"""
            current_data = input_data
            results = []
            
            for tool_name in tool_names:
                if tool_name in self.tools:
                    result = self.tools[tool_name](current_data)
                    results.append(f"{tool_name}: {result}")
                    current_data = result  # Pass result to next tool
                else:
                    results.append(f"Tool {tool_name} not found")
            
            return "\n".join(results)
        
        return pipeline_tool
    
    def create_parallel_tool(self, tool_names: List[str]) -> Callable:
        """Create parallel execution of tools"""
        
        def parallel_tool(input_data: str) -> str:
            """Execute tools in parallel"""
            results = []
            
            for tool_name in tool_names:
                if tool_name in self.tools:
                    result = self.tools[tool_name](input_data)
                    results.append(f"{tool_name}: {result}")
                else:
                    results.append(f"Tool {tool_name} not found")
            
            return "\n".join(results)
        
        return parallel_tool

# Usage
composer = ToolComposer([validate_email, text_processor, calculate_sum])

# Create composite tools
pipeline = composer.create_pipeline_tool(["text_processor", "validate_email"])
parallel = composer.create_parallel_tool(["text_processor", "validate_email"])

# Use in agent
composite_agent = Branch(
    tools=[pipeline, parallel],
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Agent with composite tool capabilities"
)
```

## Tool Development Best Practices

**Error Handling:**

- Always wrap tool functions in try/except blocks
- Return meaningful error messages
- Implement graceful degradation

**Type Safety:**

- Use type hints for all parameters and return values
- Validate input parameters before processing
- Document expected input/output formats

**Security:**

- Validate all external inputs
- Use secure methods for file operations
- Implement proper authentication for API tools

**Performance:**

- Use connection pooling for database/API tools
- Implement caching where appropriate
- Set reasonable timeouts for external operations

**Testing:**

- Test tools independently before integration
- Mock external dependencies for testing
- Verify error handling scenarios
