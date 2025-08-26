# Tools and Functions

Give agents custom capabilities with functions and tools.

## Function Tools

```python
from lionagi import Branch

def calculate_sum(a: float, b: float) -> float:
    return a + b

agent = Branch(tools=[calculate_sum])
result = await agent.ReAct(instruct={"instruction": "What is 15 + 27?"})
```

## Built-in Tools

### ReaderTool

```python
from lionagi.tools.file.reader import ReaderTool

agent = Branch(tools=[ReaderTool])
await agent.communicate("Read docs/README.md and summarize it")
```

## Tool Validation

For complex tools with input validation:

```python
from pydantic import BaseModel
from lionagi.protocols.action.tool import Tool

class WeatherRequest(BaseModel):
    city: str
    units: str = "celsius"

def get_weather(city: str, units: str = "celsius") -> dict:
    return {"city": city, "temperature": 22, "units": units}

weather_tool = Tool(func_callable=get_weather, request_options=WeatherRequest)
agent = Branch(tools=[weather_tool])
```

## Multiple Tools

```python
def search(query: str) -> str:
    return f"Results for: {query}"

def calculate(expression: str) -> float:
    return eval(expression)  # In production, use safe evaluation

agent = Branch(tools=[search, calculate])
```

That's it. LionAGI makes tools simple - just pass your functions to the `tools`
parameter.
