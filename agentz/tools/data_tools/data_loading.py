"""Data loading tool for loading and inspecting datasets."""

from typing import Union, Dict, Any
from pathlib import Path
import pandas as pd
from agents import function_tool
from agents.run_context import RunContextWrapper
from agentz.context.data_store import DataStore
from loguru import logger


@function_tool
async def load_dataset(ctx: RunContextWrapper[DataStore], file_path: str) -> Union[Dict[str, Any], str]:
    """Loads a dataset and provides comprehensive inspection information.

    This tool caches the loaded DataFrame in the pipeline data store so other
    tools can reuse it without reloading from disk.

    Args:
        ctx: Pipeline context wrapper for accessing the data store
        file_path: Path to the dataset file (CSV, JSON, Excel, etc.)

    Returns:
        Dictionary containing:
            - shape: Tuple of (rows, columns)
            - columns: List of column names
            - dtypes: Dictionary of column data types
            - missing_values: Dictionary of missing value counts per column
            - sample_data: First 5 rows as dictionary
            - summary_stats: Statistical summary for numerical columns
            - memory_usage: Memory usage information
        Or error message string if loading fails
    """
    try:
        file_path = Path(file_path)

        if not file_path.exists():
            return f"File not found: {file_path}"

        # Load based on file extension
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.json':
            df = pd.read_json(file_path)
        elif file_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            return f"Unsupported file format: {file_path.suffix}"

        # Store DataFrame in data store for reuse by other tools
        data_store = ctx.context
        if data_store is not None:
            # Store with file path key for backward compatibility
            cache_key = f"dataframe:{file_path.resolve()}"
            data_store.set(
                cache_key,
                df,
                data_type="dataframe",
                metadata={"file_path": str(file_path), "shape": df.shape}
            )
            logger.info(f"Cached DataFrame from {file_path} with key: {cache_key}")

            # Also set as the current active dataset
            data_store.set(
                "current_dataset",
                df,
                data_type="dataframe",
                metadata={"file_path": str(file_path), "shape": df.shape, "source": "loaded"}
            )
            logger.info(f"Set as current dataset for pipeline")

        # Gather comprehensive information
        result = {
            "file_path": str(file_path),
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "sample_data": df.head(5).to_dict(orient='records'),
            "summary_stats": df.describe().to_dict() if not df.select_dtypes(include=['number']).empty else {},
            "memory_usage": df.memory_usage(deep=True).to_dict(),
            "total_memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "duplicate_rows": int(df.duplicated().sum()),
        }

        return result

    except Exception as e:
        return f"Error loading dataset: {str(e)}"
