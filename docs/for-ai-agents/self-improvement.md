# Self-Improvement Guide

Learning from execution results to improve orchestration.

## Execution Metrics to Track

Monitor key metrics from LionAGI workflow executions.

```python
from lionagi import Session, Builder, Branch, iModel
import time
import asyncio

async def track_execution_metrics():
    """Track key metrics for improvement"""
    session = Session()
    builder = Builder("tracked_workflow")
    
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    session.include_branches([branch])
    
    # Add operations
    op1 = builder.add_operation("communicate", branch=branch, instruction="Task 1")
    op2 = builder.add_operation("communicate", branch=branch, instruction="Task 2")
    synthesis = builder.add_aggregation(
        "communicate", branch=branch,
        source_node_ids=[op1, op2],
        instruction="Synthesize results"
    )
    
    # Execute and track metrics
    start_time = time.time()
    result = await session.flow(builder.get_graph(), max_concurrent=2, verbose=True)
    execution_time = time.time() - start_time
    
    # Extract metrics
    metrics = {
        "execution_time": execution_time,
        "completed_operations": len(result["completed_operations"]),
        "failed_operations": len(result["skipped_operations"]),
        "success_rate": len(result["completed_operations"]) / (len(result["completed_operations"]) + len(result["skipped_operations"])),
        "parallel_efficiency": len(result["completed_operations"]) / execution_time,
        "pattern_used": "builder_graph"
    }
    
    print(f"Metrics: {metrics}")
    return metrics

asyncio.run(track_execution_metrics())
```

## Pattern Performance Analysis

Compare different orchestration patterns for similar tasks.

```python
async def pattern_comparison():
    """Compare patterns for the same task set"""
    
    tasks = ["Analyze market", "Research competitors", "Assess risks"]
    results = {}
    
    # Pattern 1: Sequential execution
    start_time = time.time()
    branch = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
    sequential_results = []
    for task in tasks:
        result = await branch.communicate(task)
        sequential_results.append(result)
    
    results["sequential"] = {
        "time": time.time() - start_time,
        "pattern": "sequential",
        "results": len(sequential_results)
    }
    
    # Pattern 2: Parallel execution
    start_time = time.time()
    parallel_results = await asyncio.gather(*[
        branch.communicate(task) for task in tasks
    ])
    
    results["parallel"] = {
        "time": time.time() - start_time,
        "pattern": "asyncio_gather", 
        "results": len(parallel_results)
    }
    
    # Pattern 3: Builder graph
    start_time = time.time()
    session = Session()
    builder = Builder("comparison")
    session.include_branches([branch])
    
    ops = [builder.add_operation("communicate", branch=branch, instruction=task) for task in tasks]
    synthesis = builder.add_aggregation("communicate", branch=branch, source_node_ids=ops, instruction="Combine")
    
    graph_result = await session.flow(builder.get_graph(), max_concurrent=3)
    
    results["builder"] = {
        "time": time.time() - start_time,
        "pattern": "builder_graph",
        "results": len(graph_result["completed_operations"])
    }
    
    # Analysis
    print("Pattern Performance Comparison:")
    for pattern, metrics in results.items():
        print(f"{pattern}: {metrics['time']:.2f}s, {metrics['results']} results")
    
    # Best pattern for this task type
    fastest = min(results.items(), key=lambda x: x[1]["time"])
    print(f"Best pattern: {fastest[0]} ({fastest[1]['time']:.2f}s)")
    
    return results

asyncio.run(pattern_comparison())
```

## Failure Analysis and Learning

Learn from execution failures to improve future orchestration.

```python
async def failure_analysis():
    """Analyze failures to improve future executions"""
    
    failure_log = []
    
    # Simulate different failure scenarios
    test_scenarios = [
        {"pattern": "direct", "task": "Complex multi-step analysis"},
        {"pattern": "gather", "task": "Sequential dependent tasks"},
        {"pattern": "builder", "task": "Simple single question"}
    ]
    
    for scenario in test_scenarios:
        try:
            if scenario["pattern"] == "direct":
                branch = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
                # This might fail for complex tasks
                result = await branch.communicate(scenario["task"])
                
            elif scenario["pattern"] == "gather":
                # This might fail for dependent tasks
                branch = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
                results = await asyncio.gather(*[
                    branch.communicate("Step 1 of analysis"),
                    branch.communicate("Step 2 that depends on Step 1")  # Problem: no dependency
                ])
                
            elif scenario["pattern"] == "builder":
                # This might be overkill for simple tasks
                session = Session()
                builder = Builder("simple")
                branch = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
                session.include_branches([branch])
                
                op = builder.add_operation("communicate", branch=branch, instruction="What is 2+2?")
                result = await session.flow(builder.get_graph())
            
            print(f"✓ {scenario['pattern']} pattern worked for: {scenario['task']}")
            
        except Exception as e:
            failure_info = {
                "pattern": scenario["pattern"],
                "task": scenario["task"], 
                "error": str(e),
                "lesson": self.analyze_failure(scenario["pattern"], scenario["task"], str(e))
            }
            failure_log.append(failure_info)
            print(f"✗ {scenario['pattern']} failed: {e}")
    
    return failure_log

def analyze_failure(pattern, task, error):
    """Extract lessons from failures"""
    lessons = {
        ("direct", "multi-step"): "Use Builder for complex workflows with multiple steps",
        ("gather", "dependent"): "Use Builder with dependencies for sequential tasks",
        ("builder", "simple"): "Use direct execution for simple single tasks"
    }
    
    # Simple pattern matching for lessons
    for key, lesson in lessons.items():
        if key[0] in pattern.lower() and any(word in task.lower() for word in key[1].split("-")):
            return lesson
    
    return "Review pattern selection guide"

asyncio.run(failure_analysis())
```

## Optimization Loop

Implement continuous improvement based on execution history.

```python
class ExecutionOptimizer:
    """Learn from execution patterns and optimize future choices"""
    
    def __init__(self):
        self.execution_history = []
        self.pattern_performance = {}
    
    def record_execution(self, task_type: str, pattern: str, metrics: dict):
        """Record execution results for learning"""
        execution = {
            "task_type": task_type,
            "pattern": pattern,
            "execution_time": metrics.get("execution_time", 0),
            "success_rate": metrics.get("success_rate", 0),
            "efficiency": metrics.get("parallel_efficiency", 0)
        }
        
        self.execution_history.append(execution)
        
        # Update pattern performance
        key = (task_type, pattern)
        if key not in self.pattern_performance:
            self.pattern_performance[key] = []
        self.pattern_performance[key].append(execution)
    
    def recommend_pattern(self, task_type: str) -> str:
        """Recommend best pattern based on historical performance"""
        
        # Find all patterns used for this task type
        relevant_patterns = {}
        for (stored_task, pattern), executions in self.pattern_performance.items():
            if stored_task == task_type:
                avg_time = sum(e["execution_time"] for e in executions) / len(executions)
                avg_success = sum(e["success_rate"] for e in executions) / len(executions)
                
                # Score: balance speed and success rate
                score = avg_success * 0.7 + (1/avg_time) * 0.3
                relevant_patterns[pattern] = score
        
        if relevant_patterns:
            best_pattern = max(relevant_patterns.items(), key=lambda x: x[1])
            return best_pattern[0]
        
        # Default fallback
        return "direct"
    
    def get_insights(self) -> dict:
        """Generate insights from execution history"""
        if not self.execution_history:
            return {"insight": "No execution history available"}
        
        # Pattern usage frequency
        pattern_usage = {}
        for execution in self.execution_history:
            pattern = execution["pattern"]
            pattern_usage[pattern] = pattern_usage.get(pattern, 0) + 1
        
        # Average performance by pattern
        pattern_performance = {}
        for execution in self.execution_history:
            pattern = execution["pattern"]
            if pattern not in pattern_performance:
                pattern_performance[pattern] = []
            pattern_performance[pattern].append(execution["execution_time"])
        
        avg_performance = {
            pattern: sum(times) / len(times) 
            for pattern, times in pattern_performance.items()
        }
        
        return {
            "most_used_pattern": max(pattern_usage.items(), key=lambda x: x[1])[0],
            "fastest_pattern": min(avg_performance.items(), key=lambda x: x[1])[0],
            "total_executions": len(self.execution_history),
            "pattern_usage": pattern_usage,
            "avg_performance": avg_performance
        }

async def optimization_example():
    """Example of using the optimization loop"""
    optimizer = ExecutionOptimizer()
    
    # Simulate some executions
    optimizer.record_execution("analysis", "direct", {"execution_time": 2.0, "success_rate": 1.0})
    optimizer.record_execution("analysis", "gather", {"execution_time": 1.5, "success_rate": 0.9})
    optimizer.record_execution("multi_step", "builder", {"execution_time": 3.0, "success_rate": 1.0})
    optimizer.record_execution("multi_step", "gather", {"execution_time": 2.0, "success_rate": 0.6})
    
    # Get recommendations
    print(f"Recommended pattern for 'analysis': {optimizer.recommend_pattern('analysis')}")
    print(f"Recommended pattern for 'multi_step': {optimizer.recommend_pattern('multi_step')}")
    
    # Get insights
    insights = optimizer.get_insights()
    print(f"Insights: {insights}")
    
    return optimizer

asyncio.run(optimization_example())
```

## Knowledge Persistence

Save and retrieve learned orchestration patterns.

```python
import json
import os

class PatternKnowledge:
    """Persist learned orchestration knowledge"""
    
    def __init__(self, knowledge_file: str = "orchestration_knowledge.json"):
        self.knowledge_file = knowledge_file
        self.knowledge = self.load_knowledge()
    
    def save_pattern_success(self, task_pattern: str, orchestration_pattern: str, metrics: dict):
        """Save successful pattern combination"""
        if "successful_patterns" not in self.knowledge:
            self.knowledge["successful_patterns"] = {}
        
        key = f"{task_pattern}:{orchestration_pattern}"
        if key not in self.knowledge["successful_patterns"]:
            self.knowledge["successful_patterns"][key] = []
        
        self.knowledge["successful_patterns"][key].append(metrics)
        self.save_knowledge()
    
    def save_pattern_failure(self, task_pattern: str, orchestration_pattern: str, error: str):
        """Save failed pattern combination to avoid repeating"""
        if "failed_patterns" not in self.knowledge:
            self.knowledge["failed_patterns"] = {}
        
        key = f"{task_pattern}:{orchestration_pattern}"
        if key not in self.knowledge["failed_patterns"]:
            self.knowledge["failed_patterns"][key] = []
        
        self.knowledge["failed_patterns"][key].append(error)
        self.save_knowledge()
    
    def get_best_pattern(self, task_pattern: str) -> str:
        """Get best orchestration pattern for task type"""
        successful = self.knowledge.get("successful_patterns", {})
        
        # Find all successful patterns for this task type
        candidates = {}
        for key, results in successful.items():
            stored_task, pattern = key.split(":", 1)
            if stored_task == task_pattern:
                # Average success metrics
                avg_time = sum(r.get("execution_time", 10) for r in results) / len(results)
                avg_success = sum(r.get("success_rate", 0) for r in results) / len(results)
                
                score = avg_success * 0.8 + (1/avg_time) * 0.2
                candidates[pattern] = score
        
        if candidates:
            return max(candidates.items(), key=lambda x: x[1])[0]
        
        return "direct"  # Default
    
    def should_avoid_pattern(self, task_pattern: str, orchestration_pattern: str) -> bool:
        """Check if pattern combination should be avoided"""
        failed = self.knowledge.get("failed_patterns", {})
        key = f"{task_pattern}:{orchestration_pattern}"
        
        # Avoid if it has failed multiple times
        return len(failed.get(key, [])) >= 3
    
    def load_knowledge(self) -> dict:
        """Load knowledge from file"""
        if os.path.exists(self.knowledge_file):
            with open(self.knowledge_file, "r") as f:
                return json.load(f)
        return {}
    
    def save_knowledge(self):
        """Save knowledge to file"""
        with open(self.knowledge_file, "w") as f:
            json.dump(self.knowledge, f, indent=2)

async def knowledge_example():
    """Example of using persistent knowledge"""
    knowledge = PatternKnowledge()
    
    # Save some successful patterns
    knowledge.save_pattern_success(
        "code_review", "gather", 
        {"execution_time": 1.5, "success_rate": 0.95}
    )
    
    knowledge.save_pattern_success(
        "research_workflow", "builder",
        {"execution_time": 3.0, "success_rate": 1.0}
    )
    
    # Save a failure
    knowledge.save_pattern_failure(
        "simple_question", "builder",
        "Overkill for simple task"
    )
    
    # Get recommendations
    print(f"Best for code_review: {knowledge.get_best_pattern('code_review')}")
    print(f"Best for research_workflow: {knowledge.get_best_pattern('research_workflow')}")
    print(f"Should avoid simple_question+builder: {knowledge.should_avoid_pattern('simple_question', 'builder')}")
    
    return knowledge

asyncio.run(knowledge_example())
```

## Best Practices for Self-Improvement

### 1. Track Key Metrics

- Execution time
- Success/failure rates
- Pattern effectiveness
- Resource usage

### 2. Analyze Patterns

- Compare different approaches for similar tasks
- Identify what makes patterns successful
- Learn from failures and mistakes

### 3. Build Feedback Loops

- Record execution results
- Update recommendations based on performance
- Continuously refine pattern selection

### 4. Persist Learning

- Save successful pattern combinations
- Remember failed approaches to avoid repeating
- Build knowledge base over time

Self-improvement in LionAGI orchestration comes from systematically tracking
execution results, analyzing pattern effectiveness, and building persistent
knowledge to make better orchestration decisions over time.
