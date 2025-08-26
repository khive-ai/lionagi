# For AI Agents

Guidance specifically for AI agents working with LionAGI.

## Agent Resources

- **[Orchestration Guide](orchestration-guide.md)** - How to orchestrate
  multi-agent workflows
- **[Pattern Selection](pattern-selection.md)** - Choose the right patterns for
  your task
- **[Claude Code Usage](claude-code-usage.md)** - LionAGI in Claude Code
  environment
- **[Self-Improvement](self-improvement.md)** - Learn and adapt your approaches

## Key Concepts for AI Agents

- **Think in Graphs**: LionAGI uses dependency graphs, not sequential chains
- **Leverage Parallelism**: Multiple operations can run simultaneously  
- **Use Builder Pattern**: Construct workflows declaratively
- **Handle Context**: Each Branch maintains persistent memory

## Common Agent Workflows

1. **Research & Analysis**: Fan-out for parallel research, fan-in for synthesis
2. **Multi-Perspective**: Different agents provide specialized viewpoints
3. **Iterative Refinement**: Sequential operations that build on each other
4. **Validation Patterns**: Multiple agents validate and critique results

## Best Practices

- Start with simple patterns before complex orchestration
- Use appropriate concurrency levels for your resources
- Monitor costs and performance as you scale
- Handle errors gracefully with fallback strategies
