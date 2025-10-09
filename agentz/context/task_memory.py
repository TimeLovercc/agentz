"""
Task-specific memory management for individual agent tasks.
Handles task state, progress tracking, and intermediate results.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Task execution result."""
    status: TaskStatus
    output: Any = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[float] = None


@dataclass
class Task:
    """Task definition and state."""
    task_id: str
    name: str
    description: str
    agent_id: str
    created_at: datetime
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[TaskResult] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "agent_id": self.agent_id,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "result": {
                "status": self.result.status.value,
                "output": self.result.output,
                "error": self.result.error,
                "metrics": self.result.metrics,
                "execution_time": self.result.execution_time
            } if self.result else None,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
            "progress": self.progress
        }


class TaskMemory:
    """Memory store for task management and tracking."""

    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._agent_tasks: Dict[str, List[str]] = {}
        self._task_dependencies: Dict[str, List[str]] = {}

    def create_task(
        self,
        name: str,
        description: str,
        agent_id: str,
        dependencies: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Create a new task and return its ID."""
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            name=name,
            description=description,
            agent_id=agent_id,
            created_at=datetime.now(),
            dependencies=dependencies or [],
            metadata=metadata or {}
        )

        self._tasks[task_id] = task

        if agent_id not in self._agent_tasks:
            self._agent_tasks[agent_id] = []
        self._agent_tasks[agent_id].append(task_id)

        if dependencies:
            self._task_dependencies[task_id] = dependencies

        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """Update task status."""
        if task_id in self._tasks:
            self._tasks[task_id].status = status
            return True
        return False

    def update_task_progress(self, task_id: str, progress: float) -> bool:
        """Update task progress (0.0 to 1.0)."""
        if task_id in self._tasks:
            self._tasks[task_id].progress = max(0.0, min(1.0, progress))
            return True
        return False

    def set_task_result(self, task_id: str, result: TaskResult) -> bool:
        """Set task result."""
        if task_id in self._tasks:
            self._tasks[task_id].result = result
            self._tasks[task_id].status = result.status
            return True
        return False

    def get_agent_tasks(self, agent_id: str, status: Optional[TaskStatus] = None) -> List[Task]:
        """Get all tasks for an agent, optionally filtered by status."""
        task_ids = self._agent_tasks.get(agent_id, [])
        tasks = [self._tasks[task_id] for task_id in task_ids if task_id in self._tasks]

        if status:
            tasks = [task for task in tasks if task.status == status]

        return tasks

    def get_ready_tasks(self, agent_id: Optional[str] = None) -> List[Task]:
        """Get tasks that are ready to execute (dependencies satisfied)."""
        ready_tasks = []

        for task in self._tasks.values():
            if agent_id and task.agent_id != agent_id:
                continue

            if task.status != TaskStatus.PENDING:
                continue

            dependencies_satisfied = all(
                self._tasks.get(dep_id, {}).status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )

            if dependencies_satisfied:
                ready_tasks.append(task)

        return ready_tasks

    def get_task_chain(self, task_id: str) -> List[str]:
        """Get the dependency chain for a task."""
        visited = set()
        chain = []

        def _build_chain(tid: str):
            if tid in visited:
                return
            visited.add(tid)

            task = self._tasks.get(tid)
            if task:
                for dep_id in task.dependencies:
                    _build_chain(dep_id)
                chain.append(tid)

        _build_chain(task_id)
        return chain

    def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        if task_id in self._tasks:
            task = self._tasks[task_id]

            # Remove from agent tasks
            if task.agent_id in self._agent_tasks:
                self._agent_tasks[task.agent_id].remove(task_id)

            # Remove dependencies
            if task_id in self._task_dependencies:
                del self._task_dependencies[task_id]

            del self._tasks[task_id]
            return True
        return False

    def clear_completed_tasks(self) -> int:
        """Clear all completed tasks and return count."""
        completed_ids = [
            task_id for task_id, task in self._tasks.items()
            if task.status == TaskStatus.COMPLETED
        ]

        for task_id in completed_ids:
            self.delete_task(task_id)

        return len(completed_ids)

    def get_stats(self) -> Dict[str, Any]:
        """Get task statistics."""
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = sum(
                1 for task in self._tasks.values() if task.status == status
            )

        return {
            "total_tasks": len(self._tasks),
            "status_counts": status_counts,
            "agents": list(self._agent_tasks.keys()),
            "average_progress": sum(task.progress for task in self._tasks.values()) / len(self._tasks) if self._tasks else 0
        }


# Global instance
task_memory = TaskMemory()