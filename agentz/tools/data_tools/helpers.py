"""Helper utilities for data tools to access cached DataFrames."""

from typing import Optional
from pathlib import Path
import pandas as pd
from agentz.flow import get_current_data_store
from loguru import logger


def get_dataframe(file_path: str, prefer_preprocessed: bool = False) -> Optional[pd.DataFrame]:
    """Get a DataFrame from cache or load from file.

    Args:
        file_path: Path to the dataset file
        prefer_preprocessed: If True, try to get preprocessed version first

    Returns:
        DataFrame or None if not found
    """
    file_path = Path(file_path)
    data_store = get_current_data_store()

    if not data_store:
        return None

    # Try preprocessed first if requested
    if prefer_preprocessed:
        preprocessed_key = f"preprocessed:{file_path.resolve()}"
        if data_store.has(preprocessed_key):
            logger.info(f"Using cached preprocessed DataFrame for {file_path}")
            return data_store.get(preprocessed_key)

    # Try regular DataFrame
    cache_key = f"dataframe:{file_path.resolve()}"
    if data_store.has(cache_key):
        logger.info(f"Using cached DataFrame for {file_path}")
        return data_store.get(cache_key)

    return None


def load_or_get_dataframe(file_path: str, prefer_preprocessed: bool = False) -> pd.DataFrame:
    """Get DataFrame from cache or load from file, with fallback loading.

    Args:
        file_path: Path to the dataset file
        prefer_preprocessed: If True, try to get preprocessed version first

    Returns:
        DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist and not in cache
        ValueError: If file format is not supported
    """
    # Try cache first
    df = get_dataframe(file_path, prefer_preprocessed)
    if df is not None:
        return df

    # Load from file
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif file_path.suffix.lower() == '.json':
        df = pd.read_json(file_path)
    elif file_path.suffix.lower() == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Cache it for future use
    data_store = get_current_data_store()
    if data_store:
        cache_key = f"dataframe:{file_path.resolve()}"
        data_store.set(
            cache_key,
            df,
            data_type="dataframe",
            metadata={"file_path": str(file_path), "shape": df.shape}
        )
        logger.info(f"Cached DataFrame from {file_path}")

    return df


def cache_object(key: str, obj: any, data_type: str = None, metadata: dict = None) -> None:
    """Cache an object in the pipeline data store.

    Args:
        key: Cache key
        obj: Object to cache
        data_type: Type descriptor (e.g., 'model', 'scaler')
        metadata: Optional metadata
    """
    data_store = get_current_data_store()
    if data_store:
        data_store.set(key, obj, data_type=data_type, metadata=metadata)
        logger.info(f"Cached {data_type or 'object'} with key: {key}")


def get_cached_object(key: str) -> Optional[any]:
    """Get a cached object from the pipeline data store.

    Args:
        key: Cache key

    Returns:
        Cached object or None
    """
    data_store = get_current_data_store()
    if data_store and data_store.has(key):
        logger.info(f"Retrieved cached object with key: {key}")
        return data_store.get(key)
    return None


def get_current_dataset() -> Optional[pd.DataFrame]:
    """Get the current active dataset from the pipeline context.

    This is the main dataset that agents are working with. It's automatically
    updated when data is loaded or preprocessed.

    Returns:
        The current DataFrame or None if no dataset is loaded
    """
    data_store = get_current_data_store()
    if data_store and data_store.has("current_dataset"):
        logger.info("Using current dataset from pipeline context")
        return data_store.get("current_dataset")
    return None


def set_current_dataset(df: pd.DataFrame, metadata: dict = None) -> None:
    """Set the current active dataset in the pipeline context.

    Args:
        df: DataFrame to set as current
        metadata: Optional metadata about the dataset
    """
    data_store = get_current_data_store()
    if data_store:
        data_store.set(
            "current_dataset",
            df,
            data_type="dataframe",
            metadata=metadata or {}
        )
        logger.info("Set current dataset in pipeline context")
