# Using LionAGI from Claude Code

Native integration guide for Claude Code users.

## Basic Setup

```python
from lionagi import Branch, iModel

BASE_CONFIG = {
    "provider": "claude_code",
    "endpoint": "query_cli",
    "model": "sonnet",
    "api_key": "dummy_api_key",
    "allowed_tools": ["Read"],
    "permission_mode": "bypassPermissions",
    "verbose_output": True,
    "cli_display_theme": "dark",
}

# Create Claude Code branch
branch = Branch(
    chat_model=iModel(cwd="lionagi", **BASE_CONFIG),
    name="cc_agent"
)

response = await branch.communicate("Analyze the codebase structure")
```

## Workspace Management

```python
CC_WORKSPACE = ".khive/workspace"

def create_cc(
    subdir: str,
    model: str = "sonnet",
    verbose_output: bool = True,
    permission_mode="default",
    auto_finish: bool = False,
):
    return iModel(
        provider="claude_code",
        endpoint="query_cli",
        model=model,
        ws=f"{CC_WORKSPACE}/{subdir}",
        verbose_output=verbose_output,
        add_dir="../../../",
        permission_mode=permission_mode,
        cli_display_theme="light",
        auto_finish=auto_finish,
    )
```

## Multi-Agent with Claude Code

```python
from lionagi import Session, Builder
from lionagi.fields import LIST_INSTRUCT_FIELD_MODEL, Instruct

async def claude_code_orchestration():
    try:
        orc_cc = create_cc("orchestrator")
        orc_branch = Branch(
            chat_model=orc_cc,
            parse_model=orc_cc,
            use_lion_system_message=True,
            system_datetime=True,
            name="orchestrator",
        )
        session = Session(default_branch=orc_branch)

        builder = Builder("CodeAnalysis")
        root = builder.add_operation(
            "operate",
            instruct=Instruct(
                instruction="Analyze codebase and create research tasks",
                context="project_root",
            ),
            reason=True,
            field_models=[LIST_INSTRUCT_FIELD_MODEL],
        )

        result = await session.flow(builder.get_graph())
        instruct_models = result["operation_results"][root].instruct_models
        
        # Fan-out to researchers
        research_nodes = []
        for i in instruct_models:
            node = builder.add_operation(
                "communicate",
                depends_on=[root],
                chat_model=create_cc("researcher"),
                **i.to_dict(),
            )
            research_nodes.append(node)

        # Execute research
        await session.flow(builder.get_graph())
        
        # Synthesis
        synthesis = builder.add_operation(
            "communicate",
            depends_on=research_nodes,
            branch=orc_branch,
            instruction="Synthesize researcher findings",
        )

        final_result = await session.flow(builder.get_graph())
        return final_result["operation_results"][synthesis]

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# Usage
result = await claude_code_orchestration()
```

## Cost Tracking

```python
def get_context_with_costs(node_id, builder, session):
    nonlocal costs
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

costs = 0
# After execution
print(f"Total cost: ${costs:.4f}")
```

## Integration Patterns

**Single Agent Analysis:**

```python
cc_model = iModel(**BASE_CONFIG)
investigator = Branch(
    name="investigator", 
    chat_model=cc_model,
    parse_model=cc_model,
)

response = await investigator.communicate(
    "Read into the architecture and explain key components"
)
```

**Multi-Workspace Setup:**

```python
# Orchestrator workspace
orc_model = create_cc("orchestrator")

# Researcher workspaces
researchers = [
    create_cc(f"researcher_{i}") 
    for i in range(3)
]
```

## Common Patterns

**File Analysis:**

```python
prompt = """
Read the specified directory structure.
Focus on architecture and design patterns.
Provide structured analysis of key components.
"""

response = await branch.communicate(prompt)
```

**Error Handling:**

```python
try:
    result = await session.flow(builder.get_graph())
    return result["operation_results"][synthesis]
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    return None
```

## Best Practices

- Use workspace isolation for multi-agent scenarios
- Track costs with `total_cost_usd` extraction
- Handle errors with traceback for debugging
- Keep configurations clean and reusable
- Leverage Claude Code's file access capabilities

## Troubleshooting

**Permission Issues:**

```python
# Use bypass for development
"permission_mode": "bypassPermissions"
```

**Workspace Conflicts:**

```python
# Separate workspaces per agent
ws=f"{CC_WORKSPACE}/{unique_subdir}"
```

**Cost Monitoring:**

```python
# Extract from model response
total_cost = msg.model_response.get("total_cost_usd", 0)
```
