# Quick Start: Orchestration-First Mindset

Learn to think in terms of orchestration, not operations.

## The Mental Model

Instead of thinking "what can LionAGI do?", think "what do I want to orchestrate?"

## Example: Building a Research System

### Traditional Approach (What Others Do)
```python
# Look for built-in research operations
result = framework.research(topic)  # Hope it does what you need
summary = framework.summarize(result)  # Hope the summary is good
report = framework.create_report(summary)  # Hope the format is right
```

### LionAGI Approach (Orchestration Engine)
```python
from lionagi import Branch, Operation

# Define YOUR research operation
class MyResearch(Operation):
    async def execute(self, topic):
        # Exactly how YOU want research done
        sources = await self.gather_sources(topic)
        validated = await self.validate_sources(sources)
        analyzed = await self.deep_analysis(validated)
        return analyzed

# Define YOUR summary operation  
class MySummary(Operation):
    async def execute(self, research):
        # Exactly how YOU want summaries
        key_points = self.extract_key_points(research)
        structured = self.structure_findings(key_points)
        return structured

# Define YOUR report operation
class MyReport(Operation):
    async def execute(self, summary):
        # Exactly the format YOU need
        return self.generate_custom_report(summary)

# Now orchestrate them
branch = Branch()
research = await branch.operate(MyResearch(), topic="quantum computing")
summary = await branch.operate(MySummary(), research)
report = await branch.operate(MyReport(), summary)
```

## The Power of This Approach

### 1. You're Not Limited
```python
# Want to use a specific API?
class WeatherAPI(Operation):
    async def execute(self, city):
        return await my_weather_service.get(city)

# Want to use a specific ML model?
class CustomModel(Operation):
    async def execute(self, input):
        return self.model.predict(input)

# Want to combine multiple frameworks?
class MultiFramework(Operation):
    async def execute(self, task):
        langchain_result = await self.langchain_chain.run(task)
        crew_result = await self.crewai_crew.kickoff(task)
        return self.combine_results(langchain_result, crew_result)

# ALL work the same way
result = await branch.operate(any_operation, **params)
```

### 2. Natural Composition
```python
import asyncio

# Build complex from simple - Sequential pipeline
async def research_pipeline(branch, topic):
    # Execute operations sequentially, passing data through pipeline
    data = topic
    for operation in [GatherSources(), ValidateSources(), AnalyzeData(), GenerateInsights(), CreateReport()]:
        data = await branch.operate(operation, data=data)
    return data

# Or parallel processing
analyses = await asyncio.gather(
    branch.operate(StatisticalAnalysis(), data=market_data),
    branch.operate(SentimentAnalysis(), data=market_data),
    branch.operate(TrendAnalysis(), data=market_data),
    branch.operate(CompetitorAnalysis(), data=market_data)
)

# Execute the pipeline
result = await research_pipeline(branch, "AI trends")
```

### 3. Cross-Branch Orchestration
```python
class MultiExpertSystem(Operation):
    async def execute(self, problem):
        # Create specialized branches (spaces)
        experts = [
            Branch(system="Security expert"),
            Branch(system="Performance expert"),
            Branch(system="UX expert")
        ]
        
        # Get all perspectives
        perspectives = await asyncio.gather(*[
            expert.communicate(problem) 
            for expert in experts
        ])
        
        # Synthesize
        synthesizer = Branch(system="Synthesizer")
        return await synthesizer.communicate(
            instruction="Combine all perspectives",
            context=perspectives
        )

# Complex multi-agent orchestration is just an operation
expert_system = MultiExpertSystem()
solution = await branch.operate(expert_system, problem="System design")
```

## Quick Patterns

### Sequential Pipeline
```python
# Define your pipeline
steps = [LoadData(), Clean(), Transform(), Analyze(), Report()]

# Execute
for step in steps:
    data = await branch.operate(step, data)
```

### Parallel Execution
```python
# Define parallel operations
ops = [SecurityCheck(), PerformanceTest(), QualityReview()]

# Execute all at once
results = await asyncio.gather(*[
    branch.operate(op, code=my_code) for op in ops
])
```

### Conditional Flow
```python
# Define conditional logic
if await branch.operate(NeedsDetailedAnalysis(), data):
    result = await branch.operate(DeepAnalysis(), data)
else:
    result = await branch.operate(QuickSummary(), data)
```

### Map-Reduce
```python
# Process chunks in parallel
chunks = split_data(large_dataset)
processed = await asyncio.gather(*[
    branch.operate(ProcessChunk(), chunk) for chunk in chunks
])

# Reduce to final result
result = await branch.operate(CombineResults(), processed)
```

## Integration Examples

### Use Any LLM
```python
class GPT4Operation(Operation):
    async def execute(self, prompt):
        return await openai.complete(prompt)

class ClaudeOperation(Operation):
    async def execute(self, prompt):
        return await anthropic.complete(prompt)

class LocalLLM(Operation):
    async def execute(self, prompt):
        return self.local_model.generate(prompt)

# All work the same way
await branch.operate(GPT4Operation(), prompt="...")
await branch.operate(ClaudeOperation(), prompt="...")
await branch.operate(LocalLLM(), prompt="...")
```

### Use Any Framework
```python
# Wrap any framework as an operation
langchain_op = LangChainWrapper(my_chain)
crew_op = CrewAIWrapper(my_crew)
autogen_op = AutoGenWrapper(my_agents)

# Orchestrate across frameworks - Sequential execution
async def multi_framework_pipeline(branch, data):
    # Execute framework operations sequentially
    current_data = data
    for op in [langchain_op, crew_op, autogen_op]:
        current_data = await branch.operate(op, data=current_data)
    return current_data

result = await multi_framework_pipeline(branch, data)
```

## Key Takeaways

1. **Operations are arbitrary** - Any computation can be an operation
2. **Branch is minimal** - Just `chat()`, `communicate()`, `operate()`, `react()`
3. **You define behavior** - Not limited to what LionAGI provides
4. **Composition is powerful** - Build complex from simple
5. **Orchestration is universal** - Works with anything

## Next Steps

- Learn about [Creating Operations](../core-concepts/operations.md)
- Explore [Orchestration Patterns](../patterns/index.md)
- See [Framework Integration](../integrations/index.md)

Remember: **You bring the operations. LionAGI is the orchestration engine.**
