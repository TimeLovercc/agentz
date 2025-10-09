"""
Persistence and recovery processes for memory management.
Handles saving/loading memory state to/from disk and recovery mechanisms.
"""

import json
import pickle
import threading
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import logging

from .global_memory import global_memory, MemoryEntry
from .task_memory import task_memory, Task, TaskResult, TaskStatus


class PersistenceManager:
    """Manages persistence of memory state to disk."""

    def __init__(self, persist_dir: str = "data/persistence"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    def save_global_memory(self, filename: Optional[str] = None) -> str:
        """Save global memory to disk."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"global_memory_{timestamp}.json"

        filepath = self.persist_dir / filename

        with self._lock:
            try:
                # Convert memory entries to serializable format
                memory_data = {}
                for key, entry in global_memory._memory.items():
                    memory_data[key] = {
                        "key": entry.key,
                        "value": entry.value,
                        "timestamp": entry.timestamp.isoformat(),
                        "agent_id": entry.agent_id,
                        "tags": entry.tags
                    }

                with open(filepath, 'w') as f:
                    json.dump(memory_data, f, indent=2, default=str)

                self.logger.info(f"Global memory saved to {filepath}")
                return str(filepath)

            except Exception as e:
                self.logger.error(f"Failed to save global memory: {e}")
                raise

    def load_global_memory(self, filename: str) -> bool:
        """Load global memory from disk."""
        filepath = self.persist_dir / filename

        if not filepath.exists():
            self.logger.error(f"Memory file not found: {filepath}")
            return False

        with self._lock:
            try:
                with open(filepath, 'r') as f:
                    memory_data = json.load(f)

                global_memory.clear()

                for key, entry_data in memory_data.items():
                    entry = MemoryEntry(
                        key=entry_data["key"],
                        value=entry_data["value"],
                        timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                        agent_id=entry_data.get("agent_id"),
                        tags=entry_data.get("tags", [])
                    )
                    global_memory._memory[key] = entry

                self.logger.info(f"Global memory loaded from {filepath}")
                return True

            except Exception as e:
                self.logger.error(f"Failed to load global memory: {e}")
                return False

    def save_task_memory(self, filename: Optional[str] = None) -> str:
        """Save task memory to disk."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"task_memory_{timestamp}.pkl"

        filepath = self.persist_dir / filename

        with self._lock:
            try:
                task_data = {
                    "tasks": {task_id: task.to_dict() for task_id, task in task_memory._tasks.items()},
                    "agent_tasks": task_memory._agent_tasks,
                    "task_dependencies": task_memory._task_dependencies
                }

                with open(filepath, 'wb') as f:
                    pickle.dump(task_data, f)

                self.logger.info(f"Task memory saved to {filepath}")
                return str(filepath)

            except Exception as e:
                self.logger.error(f"Failed to save task memory: {e}")
                raise

    def load_task_memory(self, filename: str) -> bool:
        """Load task memory from disk."""
        filepath = self.persist_dir / filename

        if not filepath.exists():
            self.logger.error(f"Task memory file not found: {filepath}")
            return False

        with self._lock:
            try:
                with open(filepath, 'rb') as f:
                    task_data = pickle.load(f)

                # Clear existing task memory
                task_memory._tasks.clear()
                task_memory._agent_tasks.clear()
                task_memory._task_dependencies.clear()

                # Restore tasks
                for task_id, task_dict in task_data["tasks"].items():
                    task = Task(
                        task_id=task_dict["task_id"],
                        name=task_dict["name"],
                        description=task_dict["description"],
                        agent_id=task_dict["agent_id"],
                        created_at=datetime.fromisoformat(task_dict["created_at"]),
                        status=TaskStatus(task_dict["status"]),
                        dependencies=task_dict["dependencies"],
                        metadata=task_dict["metadata"],
                        progress=task_dict["progress"]
                    )

                    if task_dict["result"]:
                        result_data = task_dict["result"]
                        task.result = TaskResult(
                            status=TaskStatus(result_data["status"]),
                            output=result_data["output"],
                            error=result_data["error"],
                            metrics=result_data["metrics"],
                            execution_time=result_data["execution_time"]
                        )

                    task_memory._tasks[task_id] = task

                # Restore agent tasks and dependencies
                task_memory._agent_tasks = task_data["agent_tasks"]
                task_memory._task_dependencies = task_data["task_dependencies"]

                self.logger.info(f"Task memory loaded from {filepath}")
                return True

            except Exception as e:
                self.logger.error(f"Failed to load task memory: {e}")
                return False

    def create_checkpoint(self, checkpoint_name: Optional[str] = None) -> str:
        """Create a full checkpoint of all memory state."""
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_{timestamp}"

        checkpoint_dir = self.persist_dir / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)

        global_file = self.save_global_memory(f"{checkpoint_name}_global.json")
        task_file = self.save_task_memory(f"{checkpoint_name}_tasks.pkl")

        # Move files to checkpoint directory
        Path(global_file).rename(checkpoint_dir / f"global_memory.json")
        Path(task_file).rename(checkpoint_dir / f"task_memory.pkl")

        self.logger.info(f"Checkpoint created: {checkpoint_dir}")
        return str(checkpoint_dir)

    def restore_checkpoint(self, checkpoint_name: str) -> bool:
        """Restore from a checkpoint."""
        checkpoint_dir = self.persist_dir / checkpoint_name

        if not checkpoint_dir.exists():
            self.logger.error(f"Checkpoint not found: {checkpoint_dir}")
            return False

        try:
            global_file = checkpoint_dir / "global_memory.json"
            task_file = checkpoint_dir / "task_memory.pkl"

            global_success = self.load_global_memory(str(global_file))
            task_success = self.load_task_memory(str(task_file))

            if global_success and task_success:
                self.logger.info(f"Successfully restored from checkpoint: {checkpoint_name}")
                return True
            else:
                self.logger.error(f"Failed to restore checkpoint: {checkpoint_name}")
                return False

        except Exception as e:
            self.logger.error(f"Error restoring checkpoint {checkpoint_name}: {e}")
            return False

    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        checkpoints = []
        for item in self.persist_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint_"):
                checkpoints.append(item.name)
        return sorted(checkpoints, reverse=True)

    def cleanup_old_files(self, keep_latest: int = 10) -> int:
        """Clean up old persistence files, keeping only the latest N files."""
        files = []
        for pattern in ["global_memory_*.json", "task_memory_*.pkl"]:
            files.extend(self.persist_dir.glob(pattern))

        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        deleted_count = 0
        for file_to_delete in files[keep_latest:]:
            try:
                file_to_delete.unlink()
                deleted_count += 1
            except Exception as e:
                self.logger.error(f"Failed to delete {file_to_delete}: {e}")

        self.logger.info(f"Cleaned up {deleted_count} old persistence files")
        return deleted_count


# Global instance
persistence_manager = PersistenceManager()