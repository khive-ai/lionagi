# Research Synthesis

Parallel research with expert synthesis - the natural pattern for comprehensive
analysis.

## Basic Research Synthesis

```python
from lionagi import Branch, Builder, Session, iModel
from lionagi.fields import LIST_INSTRUCT_FIELD_MODEL, Instruct
from lionagi.protocols.types import AssistantResponse

# Setup orchestrator
orchestrator = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Break research into parallel assignments and synthesize findings"
)
session = Session(default_branch=orchestrator)
builder = Builder("ResearchSynthesis")

topic = "AI safety in production systems"

# Decomposition phase
root = builder.add_operation(
    "operate",
    instruct=Instruct(
        instruction=f"Create 3-4 research assignments for: {topic}",
        context=topic
    ),
    reason=True,
    field_models=[LIST_INSTRUCT_FIELD_MODEL]
)

# Execute decomposition
result = await session.flow(builder.get_graph())
instruct_models = result["operation_results"][root].instruct_models

# Fan-out: Create researchers
research_nodes = []
for i, instruction in enumerate(instruct_models):
    researcher = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system=f"Research specialist #{i+1} - focused domain expert"
    )
    
    node = builder.add_operation(
        "communicate",
        depends_on=[root],
        branch=researcher,
        **instruction.to_dict()
    )
    research_nodes.append(node)

# Execute research
await session.flow(builder.get_graph())

# Extract findings with cost tracking
costs = 0
def get_context(node_id):
    global costs
    graph = builder.get_graph()
    node = graph.internal_nodes[node_id]
    branch = session.get_branch(node.branch_id, None)
    if (branch and len(branch.messages) > 0 and 
        isinstance(msg := branch.messages[-1], AssistantResponse)):
        costs += msg.model_response.get("total_cost_usd") or 0
        return f"""
Response: {msg.model_response.get("result") or "Not available"}
Summary: {msg.model_response.get("summary") or "Not available"}
        """.strip()

ctx = [get_context(i) for i in research_nodes]

# Fan-in: Synthesize
synthesis = builder.add_operation(
    "communicate",
    depends_on=research_nodes,
    branch=orchestrator,
    instruction="Synthesize research findings into comprehensive analysis",
    context=[i for i in ctx if i is not None]
)

final_result = await session.flow(builder.get_graph())
print(f"Research complete. Total cost: ${costs:.4f}")
```

## Literature Review

```python
# Literature review orchestrator
orchestrator = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Literature review coordinator and synthesizer"
)

papers = ["Attention is All You Need", "BERT", "GPT-3"]
focus = "transformer architecture evolution"

# Generate review plan
planning = builder.add_operation(
    "operate",
    branch=orchestrator,
    instruct=Instruct(
        instruction=f"Create analysis framework for: {focus}",
        context={"papers": papers, "focus": focus}
    ),
    field_models=[LIST_INSTRUCT_FIELD_MODEL]
)

result = await session.flow(builder.get_graph())
review_tasks = result["operation_results"][planning].instruct_models

# Parallel paper analysis
review_nodes = []
for task in review_tasks:
    reviewer = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="Academic paper analysis specialist"
    )
    
    node = builder.add_operation(
        "communicate",
        depends_on=[planning],
        branch=reviewer,
        **task.to_dict()
    )
    review_nodes.append(node)

await session.flow(builder.get_graph())

# Synthesis
synthesis = builder.add_operation(
    "communicate",
    depends_on=review_nodes,
    branch=orchestrator,
    instruction="Create comprehensive literature review synthesis"
)

final_result = await session.flow(builder.get_graph())
```

## Cost-Efficient Research

```python
# Budget-aware research
total_cost = 0
max_cost = 1.0

coordinator = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="Efficient research coordinator"
)

# Budget check before synthesis
def track_costs(node_id):
    global total_cost
    graph = builder.get_graph()
    node = graph.internal_nodes[node_id]
    branch = session.get_branch(node.branch_id, None)
    if (branch and len(branch.messages) > 0 and 
        isinstance(msg := branch.messages[-1], AssistantResponse)):
        cost = msg.model_response.get("total_cost_usd", 0)
        total_cost += cost
        return cost > 0
    return False

await session.flow(builder.get_graph())

if total_cost < max_cost:
    synthesis = builder.add_operation(
        "communicate",
        depends_on=research_nodes,
        branch=coordinator,
        instruction="Synthesize research findings efficiently"
    )
    final_result = await session.flow(builder.get_graph())
    print(f"Research completed. Cost: ${total_cost:.4f}")
else:
    print(f"Budget exceeded: ${total_cost:.4f}")
```

## Production Error Handling

```python
try:
    # Research execution with error handling
    final_result = await session.flow(builder.get_graph())
    print(f"Research complete. Total cost: ${costs:.4f}")
    
except Exception as e:
    print(f"Research failed: {e}")
    import traceback
    traceback.print_exc()
```

## When to Use

**Perfect for:** Complex topics, literature reviews, market research, technical
analysis requiring domain expertise

**Execution Flow:** Task Decomposition → Parallel Researchers → Context
Extraction → Synthesis

**Key Benefits:**

- 3-4x faster than sequential research
- Higher quality through diverse perspectives
- Systematic evidence gathering
- Cost-efficient parallel execution

Research synthesis leverages the fan-out/fan-in pattern for comprehensive
analysis through specialized parallel research with intelligent synthesis.
