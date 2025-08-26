# Claude Code Integration

Direct LionAGI integration with Claude Code workspaces and tooling.

## Setup

```python
from lionagi import Branch, iModel

# Basic Claude Code configuration
BASE_CONFIG = {
    "provider": "claude_code",
    "endpoint": "query_cli", 
    "model": "sonnet",
    "api_key": "dummy_api_key",
    "allowed_tools": ["Read"],
    "permission_mode": "bypassPermissions",
    "verbose_output": True,
}

# Create Claude Code branch
cc_branch = Branch(
    chat_model=iModel(cwd=".", **BASE_CONFIG),
    name="claude_code_agent"
)

response = await cc_branch.communicate("Analyze the project structure")
```

## Workspace Management

```python
CC_WORKSPACE = ".khive/workspace"

def create_cc_model(subdir: str, model: str = "sonnet"):
    """Create Claude Code model with workspace isolation"""
    return iModel(
        provider="claude_code",
        endpoint="query_cli",
        model=model,
        ws=f"{CC_WORKSPACE}/{subdir}",
        verbose_output=True,
        add_dir="../../../",
        permission_mode="default",
        cli_display_theme="light",
    )

# Usage with isolated workspaces
orchestrator_model = create_cc_model("orchestrator")
researcher_model = create_cc_model("researcher")
```

## Multi-Agent with Claude Code

```python
from lionagi import Session, Builder
from lionagi.fields import LIST_INSTRUCT_FIELD_MODEL, Instruct

async def claude_code_multi_agent():
    """Multi-agent workflow with Claude Code"""
    
    # Create agents with separate workspaces
    orchestrator = Branch(
        chat_model=create_cc_model("orchestrator"),
        parse_model=create_cc_model("orchestrator"),
        name="orchestrator"
    )
    
    session = Session(default_branch=orchestrator)
    builder = Builder("CCAnalysis")
    
    # Task decomposition
    root = builder.add_operation(
        "operate",
        instruct=Instruct(
            instruction="Analyze codebase and create research tasks",
            context="project_analysis"
        ),
        reason=True,
        field_models=[LIST_INSTRUCT_FIELD_MODEL]
    )
    
    result = await session.flow(builder.get_graph())
    tasks = result["operation_results"][root].instruct_models
    
    # Create researcher agents
    research_nodes = []
    for i, task in enumerate(tasks):
        node = builder.add_operation(
            "communicate",
            depends_on=[root],
            chat_model=create_cc_model(f"researcher_{i}"),
            **task.to_dict()
        )
        research_nodes.append(node)
    
    # Execute research
    await session.flow(builder.get_graph())
    
    # Synthesis
    synthesis = builder.add_operation(
        "communicate", 
        depends_on=research_nodes,
        branch=orchestrator,
        instruction="Synthesize research findings"
    )
    
    final_result = await session.flow(builder.get_graph())
    return final_result["operation_results"][synthesis]

# Usage
result = await claude_code_multi_agent()
```

## Cost Tracking

```python
def get_cc_costs(node_id, builder, session):
    """Extract costs from Claude Code responses"""
    graph = builder.get_graph()
    node = graph.internal_nodes[node_id]
    branch = session.get_branch(node.branch_id, None)
    
    if (branch and len(branch.messages) > 0):
        msg = branch.messages[-1]
        if hasattr(msg, 'model_response'):
            return msg.model_response.get("total_cost_usd", 0)
    return 0

# Track costs across workflow
total_cost = sum(get_cc_costs(node, builder, session) for node in research_nodes)
print(f"Total workflow cost: ${total_cost:.4f}")
```

## Best Practices

**Workspace Isolation:**

- Use separate workspace subdirectories per agent
- Isolate orchestrator from researchers
- Clean workspace management for complex flows

**Permission Management:**

- Use `bypassPermissions` for development
- Configure `allowed_tools` for production
- Manage CLI themes and verbosity per workspace

**Integration Patterns:**

- Leverage Claude Code's file access capabilities
- Use workspace directories for agent coordination
- Cost tracking with `total_cost_usd` extraction
