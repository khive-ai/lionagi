# Orchestration Guide for AI Agents

> **Quick Decision:** When and how to orchestrate multi-agent workflows

## 🎯 Pattern Selection

Choose your orchestration pattern based on task characteristics:

```python
def select_orchestration_pattern(task):
    """Quick pattern selector for AI agents"""
    
    # Multiple viewpoints needed?
    if task.needs_multiple_perspectives:
        return "fan_out_fan_in"
    
    # Steps must happen in order?
    if task.has_sequential_dependencies:
        return "sequential_pipeline"
    
    # Tasks can run simultaneously?
    if task.has_independent_subtasks:
        return "parallel_execution"
    
    # Conditional branching required?
    if task.requires_conditional_logic:
        return "conditional_flows"
    
    # Simple single task
    return "single_branch"
```

## 📐 Orchestration Rules

Simple rules for deciding when to orchestrate:

---

### ✅ Rule 1: Single Task → Direct Execution

**When:** Task is self-contained and straightforward  
**Examples:** Simple questions, single analysis, direct requests

```python
from lionagi import Branch, iModel
import asyncio

async def single_task():
    """One task = one branch, no orchestration needed"""
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    return await branch.communicate("Analyze this code")
```

---

### ⚡ Rule 2: Independent Tasks → Parallel Execution

**When:** Multiple tasks that don't depend on each other  
**Examples:** Code review (security + performance + style), multi-aspect analysis

```python
async def parallel_tasks():
    """Independent tasks = run simultaneously"""
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    
    tasks = [
        "Review security",
        "Check performance", 
        "Validate style"
    ]
    
    return await asyncio.gather(*[
        branch.communicate(task) for task in tasks
    ])
```

---

### 🔗 Rule 3: Dependencies → Builder Graph

**When:** Tasks depend on results from previous tasks  
**Examples:** Analysis → Recommendations → Implementation

```python
async def dependent_tasks():
    """Sequential dependencies = Builder with graph"""
    session = Session()
    builder = Builder("workflow")
    branch = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
    
    # Chain of dependencies
    step1 = builder.add_operation("communicate", branch=branch,
        instruction="Analyze architecture")
    
    step2 = builder.add_operation("communicate", branch=branch,
        instruction="Provide recommendations",
        depends_on=[step1])
    
    step3 = builder.add_operation("communicate", branch=branch,
        instruction="Create implementation plan",
        depends_on=[step2])
    
    return await session.flow(builder.get_graph())
```

---

### 🌟 Rule 4: Multiple Perspectives → Fan-Out/In

**When:** Need different expert viewpoints synthesized  
**Examples:** Security + Performance + Maintainability review

```python
async def multiple_perspectives():
    """Different viewpoints = specialized branches + synthesis"""
    session = Session()
    builder = Builder("perspectives")
    
    # Create specialized experts
    experts = {
        "security": Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"),
                          system="Security expert"),
        "performance": Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"),
                             system="Performance expert"),
        "quality": Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"),
                         system="Code quality expert")
    }
    
    # Fan-out: Parallel expert analysis
    analyses = []
    for name, expert in experts.items():
        analyses.append(
            builder.add_operation("communicate", branch=expert,
                                 instruction=f"{name} analysis")
        )
    
    # Fan-in: Synthesize all perspectives
    synthesis = builder.add_aggregation(
        "communicate", branch=experts["security"],
        source_node_ids=analyses,
        instruction="Synthesize all analyses"
    )
    
    return await session.flow(builder.get_graph(), max_concurrent=3)
```

---

## 📋 Ready-to-Use Templates

Copy and adapt these patterns:

### Parallel Tasks
```python
# Run multiple tasks simultaneously
tasks = ["Review security", "Check performance", "Validate style"]
results = await asyncio.gather(*[
    branch.communicate(task) for task in tasks
])
```

### Sequential Pipeline
```python
# Each step depends on the previous
session = Session()
builder = Builder("pipeline")
branch = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
    session.include_branches([branch])
    
    previous_step = None
    for i, step in enumerate(pipeline_steps):
        step_op = builder.add_operation(
            "communicate", branch=branch,
            instruction=step,
            depends_on=[previous_step] if previous_step else None
        )
        previous_step = step_op
    
    return await session.flow(builder.get_graph())
```

### Template 3: Multi-Expert Analysis

```python
async def multi_expert_template(task: str, expert_roles: list[str]):
    """Template for multiple expert perspectives"""
    session = Session()
    builder = Builder("multi_expert")
    
    # Create expert branches
    experts = []
    analyses = []
    
    for role in expert_roles:
        expert = Branch(
            chat_model=iModel(provider="openai", model="gpt-4o-mini"),
            system=f"You are a {role}"
        )
        experts.append(expert)
        
        analysis = builder.add_operation(
            "communicate", branch=expert,
            instruction=f"{role} analysis: {task}"
        )
        analyses.append(analysis)
    
    session.include_branches(experts)
    
    # Synthesize all expert analyses
    synthesis = builder.add_aggregation(
        "communicate", branch=experts[0],
        source_node_ids=analyses,
        instruction="Combine all expert analyses"
    )
    
    return await session.flow(builder.get_graph(), max_concurrent=len(experts))
```

### Template 4: Research → Analysis → Report

```python
async def research_analysis_report_template(topic: str):
    """Template for research-analysis-report workflow"""
    session = Session()
    builder = Builder("research_workflow")
    
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    session.include_branches([branch])
    
    # Phase 1: Research
    research = builder.add_operation(
        "communicate", branch=branch,
        instruction=f"Research comprehensive information about {topic}"
    )
    
    # Phase 2: Analysis
    analysis = builder.add_operation(
        "communicate", branch=branch,
        instruction="Analyze the research findings for key insights",
        depends_on=[research]
    )
    
    # Phase 3: Report
    report = builder.add_operation(
        "communicate", branch=branch,
        instruction="Create executive summary report",
        depends_on=[analysis]
    )
    
    return await session.flow(builder.get_graph())
```

---

## ✅ Success Indicators

Know when you've made the right choice:

| Pattern Used | Good Sign | Bad Sign |
|-------------|-----------|----------|
| **Single Task** | Fast, direct answer | Incomplete or needs multiple tries |
| **Parallel** | All tasks complete quickly | Tasks waiting on each other |
| **Sequential** | Each step builds properly | Later steps lack context |
| **Fan-Out/In** | Rich synthesis from experts | Conflicting or redundant views |

### Quick Decision Checklist

- ✅ **Correct Pattern:** Task completes efficiently with quality results
- ⚠️ **Maybe Wrong:** Taking longer than expected but still produces results
- ❌ **Wrong Pattern:** Failed operations, timeout, or poor quality output

---

## 📊 Learn and Adapt

Track what works:

1. **Record Pattern Performance**
   - Which patterns work best for which tasks
   - Average completion time
   - Success rate

2. **Adapt Based on Results**
   - If parallel is slow → try sequential
   - If single task incomplete → try multi-perspective
   - If synthesis poor → add more experts

---

## 🚀 Quick Reference

### ✅ DO Orchestrate When

- **Multiple viewpoints needed** → Fan-out/in pattern
- **Tasks can run in parallel** → Asyncio.gather
- **Steps depend on each other** → Builder with dependencies
- **Complex workflow** → Full orchestration graph

### ❌ DON'T Orchestrate When

- **Single simple question** → Direct execution
- **No parallelism benefit** → Keep it simple
- **Quick conversation** → Direct branch chat

### 💡 Remember

> **Orchestration is a tool, not a requirement.**  
> Start simple, add complexity only when needed.
