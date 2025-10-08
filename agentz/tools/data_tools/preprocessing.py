"""Data preprocessing tool for cleaning and transforming datasets."""

from typing import Union, Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from agents import function_tool
from .helpers import get_current_dataset, load_or_get_dataframe
from agentz.flow import get_current_data_store
from loguru import logger


@function_tool
async def preprocess_data(
    operations: List[str],
    file_path: Optional[str] = None,
    target_column: Optional[str] = None,
    output_path: Optional[str] = None
) -> Union[Dict[str, Any], str]:
    """Performs data preprocessing operations on a dataset.

    This tool automatically uses the current dataset from the pipeline context.
    A file_path can optionally be provided to preprocess a different dataset.

    Args:
        operations: List of preprocessing operations to perform. Options:
            - "handle_missing": Handle missing values (mean/median/mode imputation)
            - "remove_duplicates": Remove duplicate rows
            - "encode_categorical": Encode categorical variables
            - "scale_standard": Standardize numerical features (z-score)
            - "scale_minmax": Min-max scaling for numerical features
            - "remove_outliers": Remove outliers using IQR method
            - "feature_engineering": Create interaction features
        file_path: Optional path to dataset file. If not provided, uses current dataset.
        target_column: Optional target column to preserve
        output_path: Optional path to save preprocessed dataset

    Returns:
        Dictionary containing:
            - operations_applied: List of operations performed
            - original_shape: Original dataset shape
            - preprocessed_shape: Preprocessed dataset shape
            - changes_summary: Summary of changes made
            - output_path: Path where preprocessed data was saved (if output_path provided)
        Or error message string if preprocessing fails
    """
    try:
        # Get DataFrame - either from file_path or current dataset
        if file_path is None:
            df = get_current_dataset()
            if df is None:
                return "Error: No dataset loaded. Please load a dataset first using the load_dataset tool."
            logger.info("Preprocessing current dataset from pipeline context")
        else:
            df = load_or_get_dataframe(file_path, prefer_preprocessed=False)
            logger.info(f"Preprocessing dataset from: {file_path}")

        original_shape = df.shape
        changes_summary = []
        operations_applied = []

        # Handle missing values
        if "handle_missing" in operations:
            missing_before = df.isnull().sum().sum()

            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # Use median for numerical columns
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        # Use mode for categorical columns
                        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)

            missing_after = df.isnull().sum().sum()
            changes_summary.append(f"Filled {missing_before - missing_after} missing values")
            operations_applied.append("handle_missing")

        # Remove duplicates
        if "remove_duplicates" in operations:
            duplicates_before = df.duplicated().sum()
            df.drop_duplicates(inplace=True)
            changes_summary.append(f"Removed {duplicates_before} duplicate rows")
            operations_applied.append("remove_duplicates")

        # Encode categorical variables
        if "encode_categorical" in operations:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

            # Exclude target column from encoding if specified
            if target_column and target_column in categorical_cols:
                categorical_cols.remove(target_column)

            encoded_cols = []
            for col in categorical_cols:
                if df[col].nunique() <= 10:  # Only encode if reasonable number of categories
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    encoded_cols.append(col)

            changes_summary.append(f"Encoded {len(encoded_cols)} categorical columns: {', '.join(encoded_cols)}")
            operations_applied.append("encode_categorical")

        # Standardize numerical features
        if "scale_standard" in operations:
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

            # Exclude target column from scaling if specified
            if target_column and target_column in numerical_cols:
                numerical_cols.remove(target_column)

            if numerical_cols:
                scaler = StandardScaler()
                df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
                changes_summary.append(f"Standardized {len(numerical_cols)} numerical columns")
                operations_applied.append("scale_standard")

        # Min-max scaling
        if "scale_minmax" in operations:
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

            # Exclude target column from scaling if specified
            if target_column and target_column in numerical_cols:
                numerical_cols.remove(target_column)

            if numerical_cols:
                scaler = MinMaxScaler()
                df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
                changes_summary.append(f"Min-max scaled {len(numerical_cols)} numerical columns")
                operations_applied.append("scale_minmax")

        # Remove outliers
        if "remove_outliers" in operations:
            rows_before = len(df)
            numerical_cols = df.select_dtypes(include=['number']).columns

            for col in numerical_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

            rows_removed = rows_before - len(df)
            changes_summary.append(f"Removed {rows_removed} outlier rows")
            operations_applied.append("remove_outliers")

        # Feature engineering
        if "feature_engineering" in operations:
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

            # Exclude target column
            if target_column and target_column in numerical_cols:
                numerical_cols.remove(target_column)

            # Create polynomial features for first 3 numerical columns (to avoid explosion)
            if len(numerical_cols) >= 2:
                for i, col1 in enumerate(numerical_cols[:3]):
                    for col2 in numerical_cols[i+1:3]:
                        new_col_name = f"{col1}_x_{col2}"
                        df[new_col_name] = df[col1] * df[col2]

                changes_summary.append("Created interaction features")
                operations_applied.append("feature_engineering")

        # Save preprocessed dataset if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix.lower() == '.csv':
                df.to_csv(output_path, index=False)
            elif output_path.suffix.lower() in ['.xlsx', '.xls']:
                df.to_excel(output_path, index=False)
            elif output_path.suffix.lower() == '.json':
                df.to_json(output_path, orient='records')
            else:
                return f"Unsupported output format: {output_path.suffix}"

        # Update the current dataset with preprocessed version
        data_store = get_current_data_store()
        if data_store:
            # Always update current_dataset with the preprocessed version
            data_store.set(
                "current_dataset",
                df,
                data_type="dataframe",
                metadata={
                    "shape": df.shape,
                    "operations": operations_applied,
                    "source": "preprocessed"
                }
            )
            logger.info(f"Updated current dataset with preprocessed version")

            # Also cache with file path key if provided
            if file_path:
                file_path_obj = Path(file_path)
                preprocessed_key = f"preprocessed:{file_path_obj.resolve()}"
                data_store.set(
                    preprocessed_key,
                    df,
                    data_type="dataframe",
                    metadata={
                        "file_path": str(file_path),
                        "shape": df.shape,
                        "operations": operations_applied
                    }
                )
                logger.info(f"Cached preprocessed DataFrame with key: {preprocessed_key}")

        result = {
            "operations_applied": operations_applied,
            "original_shape": original_shape,
            "preprocessed_shape": df.shape,
            "changes_summary": changes_summary,
            "output_path": str(output_path) if output_path else None,
        }

        return result

    except Exception as e:
        return f"Error preprocessing dataset: {str(e)}"
