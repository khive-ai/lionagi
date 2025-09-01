"""Orchestrator: Multi-agent coordination service.

High-level service that coordinates multiple branches/sessions.
"""

from typing import Any, List, Callable, TypeVar
from uuid import UUID, uuid4
import asyncio

from ..kernel.execution.branch import Branch
from ..domain.generic.pile import Pile

T = TypeVar('T')


class Orchestrator:
    """Coordinates multiple execution contexts.
    
    This is a SERVICE that uses branches/sessions, not a space itself.
    Provides patterns like map-reduce, pipeline, fan-out/fan-in.
    """
    
    def __init__(self):
        self.id = uuid4()
        self.branches: Pile[Branch] = Pile()
        self.results: dict[UUID, Any] = {}
    
    async def map(self,
                  func: Callable[[Branch, T], Any],
                  items: List[T],
                  branch_template: Branch = None) -> List[Any]:
        """Map function across items in parallel branches"""
        branches = []
        for item in items:
            branch = branch_template.fork() if branch_template else Branch()
            self.branches.include(branch)
            branches.append(branch)
        
        # Execute in parallel
        tasks = [func(branch, item) for branch, item in zip(branches, items)]
        results = await asyncio.gather(*tasks)
        
        # Store results
        for branch, result in zip(branches, results):
            self.results[branch.id] = result
        
        return results
    
    async def reduce(self,
                    func: Callable[[List[Any]], Any],
                    results: List[Any] = None) -> Any:
        """Reduce results from multiple branches"""
        if results is None:
            results = list(self.results.values())
        
        return await func(results) if asyncio.iscoroutinefunction(func) else func(results)
    
    async def map_reduce(self,
                        map_func: Callable[[Branch, T], Any],
                        reduce_func: Callable[[List[Any]], Any],
                        items: List[T],
                        branch_template: Branch = None) -> Any:
        """Map-reduce pattern"""
        map_results = await self.map(map_func, items, branch_template)
        return await self.reduce(reduce_func, map_results)
    
    async def pipeline(self,
                      stages: List[Callable[[Branch, Any], Any]],
                      initial: Any,
                      branch_template: Branch = None) -> Any:
        """Pipeline pattern - sequential processing"""
        result = initial
        for stage in stages:
            branch = branch_template.fork() if branch_template else Branch()
            self.branches.include(branch)
            result = await stage(branch, result) if asyncio.iscoroutinefunction(stage) else stage(branch, result)
            self.results[branch.id] = result
        
        return result
    
    async def fan_out_fan_in(self,
                            splitter: Callable[[Any], List[Any]],
                            processor: Callable[[Branch, Any], Any],
                            combiner: Callable[[List[Any]], Any],
                            input_data: Any,
                            branch_template: Branch = None) -> Any:
        """Fan-out/fan-in pattern"""
        # Split
        parts = splitter(input_data)
        
        # Process in parallel
        results = await self.map(processor, parts, branch_template)
        
        # Combine
        return await self.reduce(combiner, results)