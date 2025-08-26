# DSPy Integration

Use DSPy for prompt optimization within LionAGI workflows.

## Why DSPy + LionAGI?

**DSPy**: Best-in-class prompt optimization, automatic compilation, few-shot
learning **LionAGI**: Superior orchestration, parallel execution, multi-agent
coordination

**Together**: Optimized prompts with intelligent orchestration.

## Basic Integration

Use your existing DSPy signatures as custom operations:

```python
from lionagi import Branch, Builder, Session
import dspy

# Your existing DSPy setup - no changes needed!
class AnalyzeMarket(dspy.Signature):
    """Analyze market conditions and provide insights."""
    market_data = dspy.InputField(desc="Raw market data")
    analysis = dspy.OutputField(desc="Market analysis with key insights")

class AssessRisk(dspy.Signature):
    """Assess risks based on market analysis."""
    analysis = dspy.InputField(desc="Market analysis")
    risk_assessment = dspy.OutputField(desc="Risk assessment with recommendations")

# Your DSPy modules - unchanged!
market_analyzer = dspy.ChainOfThought(AnalyzeMarket)
risk_assessor = dspy.ChainOfThought(AssessRisk)

# Wrap DSPy modules as LionAGI custom operations
async def dspy_market_analysis(branch: Branch, market_data: str, **kwargs):
    """Custom operation using optimized DSPy prompt"""
    # Your existing DSPy logic - no changes
    result = market_analyzer(market_data=market_data)
    
    # Return to LionAGI workflow
    return result.analysis

async def dspy_risk_assessment(branch: Branch, analysis: str, **kwargs):
    """Custom operation using optimized DSPy prompt"""  
    # Your existing DSPy logic - no changes
    result = risk_assessor(analysis=analysis)
    
    # Return to LionAGI workflow
    return result.risk_assessment

# Orchestrate DSPy modules with LionAGI
session = Session()
builder = Builder("dspy_workflow")

# Phase 1: DSPy market analysis
analysis_op = builder.add_operation(
    operation=dspy_market_analysis,
    market_data="Q3 earnings data, market trends, competitor analysis..."
)

# Phase 2: DSPy risk assessment (depends on analysis)
risk_op = builder.add_operation(
    operation=dspy_risk_assessment,
    analysis="{analysis_op}",  # Use result from previous operation
    depends_on=[analysis_op]
)

result = await session.flow(builder.get_graph())
```

## Multi-Signature Workflow

Orchestrate multiple DSPy signatures in parallel:

```python
import dspy
from lionagi import Session, Builder

# Your existing DSPy signatures - unchanged!
class TechnicalAnalysis(dspy.Signature):
    data = dspy.InputField()
    technical_insights = dspy.OutputField()

class FundamentalAnalysis(dspy.Signature): 
    data = dspy.InputField()
    fundamental_insights = dspy.OutputField()

class SentimentAnalysis(dspy.Signature):
    data = dspy.InputField() 
    sentiment_insights = dspy.OutputField()

# Your optimized DSPy modules
tech_analyzer = dspy.ChainOfThought(TechnicalAnalysis)
fundamental_analyzer = dspy.ChainOfThought(FundamentalAnalysis) 
sentiment_analyzer = dspy.ChainOfThought(SentimentAnalysis)

# Wrap each as custom operations
async def tech_analysis_op(branch: Branch, data: str, **kwargs):
    result = tech_analyzer(data=data)
    return result.technical_insights

async def fundamental_analysis_op(branch: Branch, data: str, **kwargs):
    result = fundamental_analyzer(data=data)
    return result.fundamental_insights
    
async def sentiment_analysis_op(branch: Branch, data: str, **kwargs):
    result = sentiment_analyzer(data=data)
    return result.sentiment_insights

# Orchestrate all analyses in parallel
session = Session()
builder = Builder("multi_analysis")

market_data = "AAPL Q3 earnings, market conditions, news sentiment..."

# Parallel execution of optimized DSPy modules
tech_op = builder.add_operation(operation=tech_analysis_op, data=market_data)
fundamental_op = builder.add_operation(operation=fundamental_analysis_op, data=market_data)
sentiment_op = builder.add_operation(operation=sentiment_analysis_op, data=market_data)

# Synthesize all optimized insights
synthesis_op = builder.add_aggregation(
    "communicate",
    source_node_ids=[tech_op, fundamental_op, sentiment_op],
    instruction="Combine technical, fundamental, and sentiment analysis into investment recommendation"
)

result = await session.flow(builder.get_graph())
```

## DSPy Optimization in LionAGI

Show how to optimize DSPy modules within LionAGI workflows:

```python
import dspy
from dspy.datasets import HotPotQA
from lionagi import Session, Builder

# Your DSPy module to optimize
class GenerateAnswer(dspy.Signature):
    """Answer questions based on context."""
    context = dspy.InputField(desc="Background context")  
    question = dspy.InputField(desc="Question to answer")
    answer = dspy.OutputField(desc="Comprehensive answer")

class RAGPipeline(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

# Optimization within LionAGI workflow
async def optimize_dspy_pipeline(branch: Branch, **kwargs):
    """Custom operation that optimizes DSPy pipeline"""
    
    # Your existing DSPy optimization - unchanged!
    trainset = [{'question': q, 'answer': a} for q, a in training_data]
    
    # Compile/optimize the pipeline
    teleprompter = dspy.BootstrapFewShot(metric=answer_correctness)
    compiled_rag = teleprompter.compile(RAGPipeline(), trainset=trainset)
    
    # Save optimized pipeline
    compiled_rag.save("optimized_rag.json")
    
    return "Pipeline optimization complete"

async def use_optimized_pipeline(branch: Branch, question: str, **kwargs):
    """Custom operation using optimized DSPy pipeline"""
    
    # Load your optimized pipeline
    optimized_rag = RAGPipeline()
    optimized_rag.load("optimized_rag.json")
    
    # Use optimized pipeline
    result = optimized_rag(question)
    return result.answer

# Orchestrate optimization and usage
session = Session()
builder = Builder("dspy_optimization")

# Phase 1: Optimize DSPy pipeline
optimize_op = builder.add_operation(
    operation=optimize_dspy_pipeline
)

# Phase 2: Use optimized pipeline (depends on optimization)
questions = ["What is quantum computing?", "How does AI work?", "Explain blockchain"]
for i, question in enumerate(questions):
    builder.add_operation(
        operation=use_optimized_pipeline,
        question=question,
        depends_on=[optimize_op]
    )

result = await session.flow(builder.get_graph())
```

## Hybrid DSPy + LionAGI Agents

Combine DSPy optimization with LionAGI's multi-agent capabilities:

```python
# DSPy signatures for different agent types
class ResearchSignature(dspy.Signature):
    topic = dspy.InputField()
    research_findings = dspy.OutputField()

class AnalysisSignature(dspy.Signature):
    findings = dspy.InputField()
    analysis = dspy.OutputField()

class CritiqueSignature(dspy.Signature):
    analysis = dspy.InputField()
    critique = dspy.OutputField()

# Optimized DSPy modules
research_module = dspy.ChainOfThought(ResearchSignature)
analysis_module = dspy.ChainOfThought(AnalysisSignature)
critique_module = dspy.ChainOfThought(CritiqueSignature)

# LionAGI branches using optimized DSPy prompts
class DSPyBranch(Branch):
    def __init__(self, dspy_module, **kwargs):
        super().__init__(**kwargs)
        self.dspy_module = dspy_module
    
    async def dspy_execute(self, **inputs):
        """Execute DSPy module and return to LionAGI context"""
        result = self.dspy_module(**inputs)
        # Return as standard LionAGI communication
        return str(result)

# Create specialized branches with optimized prompts
session = Session()
builder = Builder("hybrid_workflow")

researcher = DSPyBranch(research_module, system="Research specialist with optimized prompts")
analyst = DSPyBranch(analysis_module, system="Analysis specialist with optimized prompts") 
critic = DSPyBranch(critique_module, system="Critique specialist with optimized prompts")

session.include_branches([researcher, analyst, critic])

# Workflow using optimized DSPy prompts in LionAGI orchestration
topic = "Impact of AI on healthcare"

research_op = builder.add_operation(
    "dspy_execute", 
    branch=researcher,
    topic=topic
)

analysis_op = builder.add_operation(
    "dspy_execute",
    branch=analyst, 
    findings="{research_op}",
    depends_on=[research_op]
)

critique_op = builder.add_operation(
    "dspy_execute",
    branch=critic,
    analysis="{analysis_op}",
    depends_on=[analysis_op]
)

result = await session.flow(builder.get_graph())
```

## Performance with DSPy

Optimize both prompts and orchestration:

```python
import asyncio
from lionagi import Session, Builder

# Batch processing with optimized DSPy prompts
async def batch_dspy_analysis(branch: Branch, questions: list, **kwargs):
    """Process multiple questions with optimized DSPy in parallel"""
    
    async def single_analysis(question):
        # Your optimized DSPy module - unchanged
        result = optimized_analyzer(question=question)
        return result.analysis
    
    # Parallel execution within the operation
    results = await asyncio.gather(*[
        single_analysis(q) for q in questions
    ])
    
    return results

# High-throughput DSPy processing
questions = [f"Question {i}" for i in range(100)]
batch_size = 10

builder = Builder("high_throughput_dspy")

for i in range(0, len(questions), batch_size):
    batch = questions[i:i + batch_size]
    builder.add_operation(
        operation=batch_dspy_analysis,
        questions=batch
    )

# Execute with controlled concurrency
result = await session.flow(
    builder.get_graph(),
    max_concurrent=5  # Control resource usage
)
```

## A/B Testing DSPy Models

Compare different DSPy optimizations:

```python
# Different DSPy optimizations to compare
async def dspy_model_a(branch: Branch, input_data: str, **kwargs):
    # Your Model A optimization
    return model_a_result

async def dspy_model_b(branch: Branch, input_data: str, **kwargs): 
    # Your Model B optimization
    return model_b_result

# A/B test different optimizations in parallel
builder = Builder("dspy_ab_test")

test_data = "Sample input for comparison"

model_a_op = builder.add_operation(operation=dspy_model_a, input_data=test_data)
model_b_op = builder.add_operation(operation=dspy_model_b, input_data=test_data)

# Compare results
comparison_op = builder.add_aggregation(
    "communicate",
    source_node_ids=[model_a_op, model_b_op],
    instruction="Compare Model A and Model B results. Which performs better and why?"
)

result = await session.flow(builder.get_graph())
```

## Key Benefits

1. **Zero Migration**: Keep your existing DSPy code unchanged
2. **Superior Orchestration**: LionAGI handles parallel execution, dependencies
3. **Optimization + Orchestration**: Best of both worlds
4. **A/B Testing**: Easy comparison of different optimizations
5. **Scalable**: Built-in performance controls
6. **Hybrid Workflows**: Mix DSPy with other AI operations

DSPy provides the prompt optimization, LionAGI provides the orchestration
intelligence.
