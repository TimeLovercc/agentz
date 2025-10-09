"""Data analysis tool for exploratory data analysis and statistical analysis."""

from typing import Union, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from agents import function_tool
from agents.run_context import RunContextWrapper
from agentz.memory.pipeline_context import PipelineDataStore
from .helpers import load_or_get_dataframe
from loguru import logger


@function_tool
async def analyze_data(ctx: RunContextWrapper[PipelineDataStore], file_path: Optional[str] = None, target_column: str = None) -> Union[Dict[str, Any], str]:
    """Performs comprehensive exploratory data analysis on a dataset.

    This tool automatically uses the current dataset from the pipeline context.
    A file_path can optionally be provided to analyze a different dataset.

    Args:
        ctx: Pipeline context wrapper for accessing the data store
        file_path: Optional path to dataset file. If not provided, uses current dataset.
        target_column: Optional target column for correlation analysis

    Returns:
        Dictionary containing:
            - distributions: Distribution statistics for each column
            - correlations: Correlation matrix for numerical columns
            - outliers: Outlier detection results using IQR method
            - patterns: Identified patterns and insights
            - recommendations: Data quality and preprocessing recommendations
        Or error message string if analysis fails
    """
    try:
        # Get DataFrame - either from file_path or current dataset
        data_store = ctx.context
        if file_path is None:
            if data_store and data_store.has("current_dataset"):
                df = data_store.get("current_dataset")
                logger.info("Analyzing current dataset from pipeline context")
            else:
                return "Error: No dataset loaded. Please load a dataset first using the load_dataset tool."
        else:
            df = load_or_get_dataframe(file_path, prefer_preprocessed=False, data_store=data_store)
            logger.info(f"Analyzing dataset from: {file_path}")

        result = {}

        # Distribution analysis
        distributions = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                distributions[col] = {
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "q25": float(df[col].quantile(0.25)),
                    "q75": float(df[col].quantile(0.75)),
                    "skewness": float(df[col].skew()),
                    "kurtosis": float(df[col].kurtosis()),
                }
            else:
                distributions[col] = {
                    "unique_values": int(df[col].nunique()),
                    "top_value": str(df[col].mode()[0]) if not df[col].mode().empty else None,
                    "top_frequency": int(df[col].value_counts().iloc[0]) if len(df[col].value_counts()) > 0 else 0,
                }
        result["distributions"] = distributions

        # Correlation analysis
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            result["correlations"] = corr_matrix.to_dict()

            if target_column and target_column in corr_matrix.columns:
                target_corr = corr_matrix[target_column].drop(target_column).sort_values(ascending=False)
                result["target_correlations"] = target_corr.to_dict()

        # Outlier detection using IQR method
        outliers = {}
        for col in numeric_df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outliers[col] = {
                "count": int(outlier_count),
                "percentage": float(outlier_count / len(df) * 100),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }
        result["outliers"] = outliers

        # Pattern identification
        patterns = []

        # High correlation patterns
        if "correlations" in result:
            for col1 in corr_matrix.columns:
                for col2 in corr_matrix.columns:
                    if col1 < col2:  # Avoid duplicates
                        corr_val = corr_matrix.loc[col1, col2]
                        if abs(corr_val) > 0.7:
                            patterns.append(f"Strong correlation ({corr_val:.2f}) between {col1} and {col2}")

        # Missing data patterns
        missing_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
        if missing_cols:
            patterns.append(f"Missing data detected in {len(missing_cols)} columns: {', '.join(missing_cols[:5])}")

        # Outlier patterns
        high_outlier_cols = [col for col, info in outliers.items() if info['percentage'] > 5]
        if high_outlier_cols:
            patterns.append(f"High outlier percentage (>5%) in columns: {', '.join(high_outlier_cols)}")

        result["patterns"] = patterns

        # Recommendations
        recommendations = []

        if missing_cols:
            recommendations.append("Consider imputation strategies for missing values")

        if high_outlier_cols:
            recommendations.append("Review and handle outliers before modeling")

        # Check for imbalanced categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            value_counts = df[col].value_counts()
            if len(value_counts) > 1:
                imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
                if imbalance_ratio > 10:
                    recommendations.append(f"Column '{col}' shows class imbalance (ratio: {imbalance_ratio:.1f})")

        # Check for constant or near-constant columns
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.01 and df[col].nunique() > 1:
                recommendations.append(f"Column '{col}' has very low variance, consider removing")

        result["recommendations"] = recommendations

        return result

    except Exception as e:
        return f"Error analyzing dataset: {str(e)}"
