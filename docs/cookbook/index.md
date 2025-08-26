# Cookbook

Complete, working examples you can copy and modify.

## Available Recipes

### Analysis & Research
- [Claim Extraction](claim-extraction.md) - Extract and validate claims from documents
- [Research Synthesis](research-synthesis.md) - Aggregate multiple sources into insights

### Business Applications  
- [HR Automation](hr-automation.md) - Multi-agent HR workflow system
- [Code Review Crew](code-review-crew.md) - Parallel code analysis with quality gates

### Creative Work
- [Brainstorming](brainstorming.md) - Generate and refine ideas collaboratively

### Technical
- [Data Persistence](data-persistence.md) - Save agent state to databases

## Quick Templates

### Basic Multi-Agent Analysis
```python
from lionagi import Branch, iModel
import asyncio

agents = {
    "analyst": Branch(system="Analyze data", chat_model=iModel(provider="openai", model="gpt-4o-mini")),
    "critic": Branch(system="Find issues", chat_model=iModel(provider="openai", model="gpt-4o-mini")),
    "advisor": Branch(system="Give recommendations", chat_model=iModel(provider="openai", model="gpt-4o-mini"))
}

async def analyze(topic):
    results = await asyncio.gather(*[
        agent.chat(f"Analyze: {topic}") 
        for agent in agents.values()
    ])
    return dict(zip(agents.keys(), results))
```

### Sequential Pipeline
```python
from lionagi import Session, Branch, Builder

async def pipeline(input_data):
    session = Session()
    builder = Builder("pipeline")
    
    extract = builder.add_operation("chat", instruction=f"Extract key points from: {input_data}")
    analyze = builder.add_operation("chat", depends_on=[extract], instruction="Analyze the extracted points")
    summarize = builder.add_operation("chat", depends_on=[analyze], instruction="Create executive summary")
    
    return await session.flow(builder.get_graph())
```

## Contributing Examples

Have a useful pattern? Submit a PR with:
1. Working code
2. Clear use case
3. Expected output
4. Performance metrics