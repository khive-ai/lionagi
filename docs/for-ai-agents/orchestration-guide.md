# Orchestration Guide for AI Agents

When and how to orchestrate multi-agent workflows.

## Pattern Recognition

Decision framework for orchestration patterns:

```python
# Pattern selection logic
def select_orchestration_pattern(task_requirements):
    if task_requirements.get("multiple_perspectives"):
        return "fan_out_fan_in"
    elif task_requirements.get("sequential_steps"):
        return "sequential_pipeline" 
    elif task_requirements.get("independent_parallel"):
        return "parallel_execution"
    elif task_requirements.get("conditional_logic"):
        return "conditional_flows"
    else:
        return "single_branch"
```

## Orchestration Rules

Clear rules for AI agents to follow:

### Rule 1: Single Task → Direct Execution
```python
from lionagi import Branch, iModel
import asyncio

async def single_task_rule():
    """Rule: One simple task = direct branch execution"""
    
    # Good for: Questions, single analysis, simple requests
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    
    result = await branch.communicate("Analyze this code for security issues")
    return result

# When to use: Task is self-contained and straightforward
```

### Rule 2: Independent Tasks → Parallel Execution
```python
async def parallel_rule():
    """Rule: Multiple independent tasks = asyncio.gather"""
    
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    
    independent_tasks = [
        "Review code security",
        "Check performance issues", 
        "Validate style compliance",
        "Test error handling"
    ]
    
    # All tasks can run simultaneously
    results = await asyncio.gather(*[
        branch.communicate(task) for task in independent_tasks
    ])
    
    return results

# When to use: Tasks don't depend on each other's results
```

### Rule 3: Dependencies → Builder Graph
```python
from lionagi import Session, Builder

async def dependency_rule():
    """Rule: Task dependencies = Builder with workflow graph"""
    
    session = Session()
    builder = Builder("dependent_workflow")
    
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    session.include_branches([branch])
    
    # Step 1: Initial analysis
    analysis = builder.add_operation(
        "communicate", branch=branch,
        instruction="Analyze codebase architecture"
    )
    
    # Step 2: Depends on analysis
    recommendations = builder.add_operation(
        "communicate", branch=branch,
        instruction="Provide improvement recommendations",
        depends_on=[analysis]
    )
    
    # Step 3: Implementation plan (depends on recommendations)
    plan = builder.add_operation(
        "communicate", branch=branch,
        instruction="Create implementation plan",
        depends_on=[recommendations]
    )
    
    result = await session.flow(builder.get_graph())
    return result

# When to use: Later tasks need results from earlier tasks
```

### Rule 4: Multiple Perspectives → Fan-Out/Fan-In
```python
async def multiple_perspectives_rule():
    """Rule: Need different viewpoints = specialized branches + aggregation"""
    
    session = Session()
    builder = Builder("multi_perspective")
    
    # Specialized branches for different perspectives
    security_expert = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="Security expert focused on vulnerabilities"
    )
    
    performance_expert = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="Performance expert focused on optimization"
    )
    
    maintainability_expert = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="Code quality expert focused on maintainability"
    )
    
    session.include_branches([security_expert, performance_expert, maintainability_expert])
    
    # Fan-out: Each expert analyzes independently
    security_analysis = builder.add_operation(
        "communicate", branch=security_expert,
        instruction="Security analysis of codebase"
    )
    
    performance_analysis = builder.add_operation(
        "communicate", branch=performance_expert,
        instruction="Performance analysis of codebase"
    )
    
    maintainability_analysis = builder.add_operation(
        "communicate", branch=maintainability_expert,
        instruction="Maintainability analysis of codebase"
    )
    
    # Fan-in: Combine all perspectives
    final_report = builder.add_aggregation(
        "communicate", branch=security_expert,
        source_node_ids=[security_analysis, performance_analysis, maintainability_analysis],
        instruction="Synthesize all expert analyses into comprehensive report"
    )
    
    result = await session.flow(builder.get_graph(), max_concurrent=3)
    return result

# When to use: Need different expert viewpoints on same problem
```

## Code Templates

Copy-paste templates for common orchestration patterns:

### Template 1: Simple Parallel Processing
```python
async def parallel_template(tasks: list[str]):
    """Template for independent parallel tasks"""
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    
    results = await asyncio.gather(*[
        branch.communicate(task) for task in tasks
    ])
    
    return results
```

### Template 2: Sequential Pipeline
```python
async def sequential_template(pipeline_steps: list[str]):
    """Template for sequential dependent tasks"""
    session = Session()
    builder = Builder("sequential")
    
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
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

## Success Indicators

How to verify orchestration decisions were correct:

### Execution Success Metrics
```python
def evaluate_orchestration_success(result: dict) -> dict:
    """Evaluate if orchestration was successful"""
    
    completed = len(result.get("completed_operations", []))
    failed = len(result.get("skipped_operations", []))
    total = completed + failed
    
    success_metrics = {
        "completion_rate": completed / total if total > 0 else 0,
        "all_completed": failed == 0,
        "partial_success": completed > 0,
        "total_operations": total
    }
    
    # Success thresholds
    if success_metrics["completion_rate"] >= 0.8:
        success_metrics["orchestration_quality"] = "excellent"
    elif success_metrics["completion_rate"] >= 0.6:
        success_metrics["orchestration_quality"] = "good"
    else:
        success_metrics["orchestration_quality"] = "needs_improvement"
    
    return success_metrics
```

### Pattern Effectiveness Check
```python
def check_pattern_effectiveness(task_type: str, pattern_used: str, execution_time: float) -> bool:
    """Check if chosen pattern was effective"""
    
    # Pattern efficiency expectations
    efficiency_standards = {
        ("single_task", "direct"): 2.0,  # Should complete in < 2 seconds
        ("independent_parallel", "gather"): 3.0,  # Parallel should be fast
        ("sequential_steps", "builder"): 10.0,  # Sequential can take longer
        ("multi_perspective", "builder"): 8.0,  # Multiple experts need time
    }
    
    expected_time = efficiency_standards.get((task_type, pattern_used), 5.0)
    return execution_time <= expected_time
```

## Learning Loop

Improve orchestration decisions over time:

```python
class OrchestrationLearner:
    """Learn from orchestration results to improve future decisions"""
    
    def __init__(self):
        self.execution_history = []
        self.pattern_success_rates = {}
    
    def record_execution(self, task_type: str, pattern: str, success_metrics: dict):
        """Record execution for learning"""
        execution = {
            "task_type": task_type,
            "pattern": pattern,
            "completion_rate": success_metrics["completion_rate"],
            "execution_time": success_metrics.get("execution_time", 0),
            "quality": success_metrics.get("orchestration_quality", "unknown")
        }
        
        self.execution_history.append(execution)
        
        # Update pattern success rates
        key = (task_type, pattern)
        if key not in self.pattern_success_rates:
            self.pattern_success_rates[key] = []
        self.pattern_success_rates[key].append(success_metrics["completion_rate"])
    
    def get_best_pattern_for_task(self, task_type: str) -> str:
        """Recommend best pattern based on historical success"""
        pattern_scores = {}
        
        for (stored_task, pattern), rates in self.pattern_success_rates.items():
            if stored_task == task_type:
                avg_success = sum(rates) / len(rates)
                pattern_scores[pattern] = avg_success
        
        if pattern_scores:
            best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
            return best_pattern[0]
        
        return "direct"  # Default fallback
    
    def should_adjust_pattern(self, task_type: str, current_pattern: str) -> bool:
        """Check if pattern should be changed based on recent performance"""
        key = (task_type, current_pattern)
        if key in self.pattern_success_rates:
            recent_rates = self.pattern_success_rates[key][-3:]  # Last 3 executions
            avg_recent_success = sum(recent_rates) / len(recent_rates)
            return avg_recent_success < 0.7  # Below 70% success
        return False

async def learning_example():
    """Example of using orchestration learning"""
    learner = OrchestrationLearner()
    
    # Simulate learning from executions
    learner.record_execution("code_review", "gather", {"completion_rate": 0.9})
    learner.record_execution("code_review", "builder", {"completion_rate": 0.95})
    learner.record_execution("research", "builder", {"completion_rate": 1.0})
    
    # Get recommendations
    best_for_review = learner.get_best_pattern_for_task("code_review")
    best_for_research = learner.get_best_pattern_for_task("research")
    
    print(f"Best pattern for code_review: {best_for_review}")
    print(f"Best pattern for research: {best_for_research}")
    
    return learner

asyncio.run(learning_example())
```

## Quick Reference

**When to orchestrate:**
- Multiple independent tasks → Use asyncio.gather
- Task dependencies → Use Builder with depends_on
- Multiple perspectives needed → Use specialized branches
- Complex multi-phase workflow → Use Builder graph

**When NOT to orchestrate:**
- Single simple question → Use direct branch.communicate()
- Quick conversational exchange → Use direct execution
- No parallelism or dependencies → Keep it simple

**Success indicators:**
- Completion rate > 80%
- Execution time within expected bounds
- All required perspectives captured
- Results quality meets requirements

Orchestration in LionAGI should be used when it adds clear value through parallelism, specialization, or dependency management - not for simple tasks that can be handled directly.