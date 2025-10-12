"""
Pipeline-scoped data store for sharing in-memory objects between agents.
Enables efficient data sharing without redundant file I/O.
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass
import threading
from loguru import logger


@dataclass
class DataStoreEntry:
    """Single entry in the pipeline data store with metadata."""
    key: str
    value: Any
    timestamp: datetime
    data_type: str
    metadata: Dict[str, Any]

    def size_mb(self) -> float:
        """Estimate size in MB (rough approximation)."""
        import sys
        return sys.getsizeof(self.value) / 1024 / 1024


class DataStore:
    """Thread-safe data store for sharing objects within a pipeline run.

    Designed for storing:
    - DataFrames loaded from files
    - Trained models
    - Preprocessed data
    - Intermediate computation results

    Key naming conventions:
    - DataFrames: f"dataframe:{file_path}"
    - Models: f"model:{model_name}"
    - Preprocessed data: f"preprocessed:{file_path}"
    - Custom: any string key
    """

    def __init__(self, experiment_id: Optional[str] = None):
        """Initialize the data store.

        Args:
            experiment_id: Optional experiment ID for tracking
        """
        self._store: Dict[str, DataStoreEntry] = {}
        self._lock = threading.RLock()
        self.experiment_id = experiment_id
        logger.debug(f"Initialized DataStore for experiment: {experiment_id}")

    def set(
        self,
        key: str,
        value: Any,
        data_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a value in the data store.

        Args:
            key: Unique key for the value
            value: The object to store
            data_type: Optional type descriptor (e.g., 'dataframe', 'model')
            metadata: Optional metadata dict
        """
        with self._lock:
            # Infer data type if not provided
            if data_type is None:
                data_type = type(value).__name__

            entry = DataStoreEntry(
                key=key,
                value=value,
                timestamp=datetime.now(),
                data_type=data_type,
                metadata=metadata or {}
            )

            self._store[key] = entry
            logger.debug(
                f"Stored {data_type} at key '{key}' "
                f"(size: {entry.size_mb():.2f} MB)"
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the data store.

        Args:
            key: Key to retrieve
            default: Default value if key not found

        Returns:
            The stored value or default
        """
        with self._lock:
            entry = self._store.get(key)
            if entry:
                logger.debug(f"Retrieved {entry.data_type} from key '{key}'")
                return entry.value
            return default

    def get_entry(self, key: str) -> Optional[DataStoreEntry]:
        """Retrieve full entry with metadata.

        Args:
            key: Key to retrieve

        Returns:
            DataStoreEntry or None
        """
        with self._lock:
            return self._store.get(key)

    def has(self, key: str) -> bool:
        """Check if a key exists in the store.

        Args:
            key: Key to check

        Returns:
            True if key exists
        """
        with self._lock:
            return key in self._store

    def delete(self, key: str) -> bool:
        """Delete a key from the store.

        Args:
            key: Key to delete

        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            if key in self._store:
                del self._store[key]
                logger.debug(f"Deleted key '{key}' from data store")
                return True
            return False

    def clear(self) -> None:
        """Clear all data from the store."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            logger.debug(f"Cleared {count} entries from data store")

    def list_keys(self, data_type: Optional[str] = None) -> List[str]:
        """List all keys, optionally filtered by data type.

        Args:
            data_type: Optional filter by data type

        Returns:
            List of keys
        """
        with self._lock:
            if data_type is None:
                return list(self._store.keys())
            return [
                key for key, entry in self._store.items()
                if entry.data_type == data_type
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the data store.

        Returns:
            Dictionary with stats
        """
        with self._lock:
            total_size_mb = sum(entry.size_mb() for entry in self._store.values())
            data_types = {}
            for entry in self._store.values():
                data_types[entry.data_type] = data_types.get(entry.data_type, 0) + 1

            return {
                "total_entries": len(self._store),
                "total_size_mb": round(total_size_mb, 2),
                "data_types": data_types,
                "experiment_id": self.experiment_id,
                "keys": list(self._store.keys())
            }

    def __len__(self) -> int:
        """Return number of entries."""
        with self._lock:
            return len(self._store)

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        return self.has(key)

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access."""
        value = self.get(key)
        if value is None and key not in self._store:
            raise KeyError(f"Key '{key}' not found in data store")
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """Support dictionary-style assignment."""
        self.set(key, value)
