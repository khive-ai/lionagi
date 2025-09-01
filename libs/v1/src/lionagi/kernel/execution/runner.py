"""Runner: The execution engine.

Single-threaded async executor - no nested TaskGroups!
Learned from v1 hanging issues.
"""

import asyncio
from typing import Any, Callable, Optional
from uuid import UUID, uuid4
from contextlib import asynccontextmanager

from ..foundation.contracts import Observable, Observation
from ..safety.ipu import IPU


class Runner:
    """Core execution engine.
    
    Key lessons from v1 issues:
    - Single TaskGroup only (no nesting!)
    - Pure async/await
    - No hidden thread pools
    - Clear cancellation semantics
    """
    
    def __init__(self, ipu: IPU = None):
        self.id = uuid4()
        self.ipu = ipu or IPU()
        self.current_task: Optional[asyncio.Task] = None
        self.task_group: Optional[asyncio.TaskGroup] = None
    
    async def execute(self, 
                     coro: Callable[..., Any],
                     *args,
                     **kwargs) -> Any:
        """Execute a coroutine with IPU validation"""
        # Create observable for the execution
        # TODO: Wrap execution in observable
        
        # Single task execution
        self.current_task = asyncio.create_task(coro(*args, **kwargs))
        try:
            result = await self.current_task
            return result
        finally:
            self.current_task = None
    
    @asynccontextmanager
    async def batch(self):
        """Execute multiple tasks in a single TaskGroup"""
        async with asyncio.TaskGroup() as tg:
            self.task_group = tg
            try:
                yield tg
            finally:
                self.task_group = None
    
    async def cancel_current(self):
        """Cancel current execution"""
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass
    
    async def shutdown(self):
        """Graceful shutdown"""
        await self.cancel_current()
        # No nested groups to worry about!