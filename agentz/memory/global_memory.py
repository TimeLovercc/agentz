"""
Global memory management for the multi-agent system.
Handles shared state and data across all agents.
"""

from typing import Dict, Any, List, Optional
import threading
import json
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class MemoryEntry:
    """Single memory entry with metadata."""
    key: str
    value: Any
    timestamp: datetime
    agent_id: Optional[str] = None
    tags: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class GlobalMemory:
    """Thread-safe global memory store for multi-agent system."""

    def __init__(self):
        self._memory: Dict[str, MemoryEntry] = {}
        self._lock = threading.RLock()
        self._subscribers: Dict[str, List[callable]] = {}

    def store(self, key: str, value: Any, agent_id: Optional[str] = None, tags: List[str] = None) -> None:
        """Store a value in global memory."""
        with self._lock:
            entry = MemoryEntry(
                key=key,
                value=value,
                timestamp=datetime.now(),
                agent_id=agent_id,
                tags=tags or []
            )
            self._memory[key] = entry
            self._notify_subscribers(key, entry)

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from global memory."""
        with self._lock:
            entry = self._memory.get(key)
            return entry.value if entry else None

    def retrieve_entry(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve full memory entry with metadata."""
        with self._lock:
            return self._memory.get(key)

    def delete(self, key: str) -> bool:
        """Delete a key from global memory."""
        with self._lock:
            if key in self._memory:
                del self._memory[key]
                return True
            return False

    def list_keys(self, agent_id: Optional[str] = None, tags: List[str] = None) -> List[str]:
        """List all keys, optionally filtered by agent_id or tags."""
        with self._lock:
            keys = []
            for key, entry in self._memory.items():
                if agent_id and entry.agent_id != agent_id:
                    continue
                if tags and not any(tag in entry.tags for tag in tags):
                    continue
                keys.append(key)
            return keys

    def clear(self) -> None:
        """Clear all memory."""
        with self._lock:
            self._memory.clear()

    def subscribe(self, key_pattern: str, callback: callable) -> None:
        """Subscribe to memory changes for a key pattern."""
        with self._lock:
            if key_pattern not in self._subscribers:
                self._subscribers[key_pattern] = []
            self._subscribers[key_pattern].append(callback)

    def _notify_subscribers(self, key: str, entry: MemoryEntry) -> None:
        """Notify subscribers of memory changes."""
        for pattern, callbacks in self._subscribers.items():
            if pattern == key or pattern == "*":
                for callback in callbacks:
                    try:
                        callback(key, entry)
                    except Exception:
                        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            return {
                "total_entries": len(self._memory),
                "agents": list(set(entry.agent_id for entry in self._memory.values() if entry.agent_id)),
                "tags": list(set(tag for entry in self._memory.values() for tag in entry.tags))
            }


# Global instance
global_memory = GlobalMemory()