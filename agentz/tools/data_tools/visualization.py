"""Data visualization tool for creating charts and plots."""

from typing import Union, Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from agents import function_tool
from agents.run_context import RunContextWrapper
from agentz.context.pipeline_context import PipelineDataStore
from .helpers import load_or_get_dataframe
from loguru import logger


@function_tool
async def create_visualization(
    ctx: RunContextWrapper[PipelineDataStore],
    plot_type: str,
    file_path: Optional[str] = None,
    columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    output_path: Optional[str] = None
) -> Union[Dict[str, Any], str]:
    """Creates data visualizations from a dataset.

    This tool automatically uses the current dataset from the pipeline context.
    A file_path can optionally be provided to visualize a different dataset.

    Args:
        ctx: Pipeline context wrapper for accessing the data store
        plot_type: Type of visualization to create. Options:
            - "distribution": Histogram/distribution plots for numerical columns
            - "correlation": Correlation heatmap
            - "scatter": Scatter plot (requires 2 columns)
            - "box": Box plot for outlier detection
            - "bar": Bar chart for categorical data
            - "pairplot": Pairwise relationships plot
        file_path: Optional path to dataset file. If not provided, uses current dataset.
        columns: List of columns to visualize (optional, uses all if not specified)
        target_column: Target column for colored scatter/pair plots
        output_path: Path to save the visualization (PNG format)

    Returns:
        Dictionary containing:
            - plot_type: Type of plot created
            - columns_plotted: Columns included in the plot
            - output_path: Path where plot was saved
            - plot_base64: Base64-encoded PNG image (if no output_path)
            - insights: Visual insights extracted from the plot
        Or error message string if visualization fails
    """
    try:
        # Get DataFrame - either from file_path or current dataset
        data_store = ctx.context
        if file_path is None:
            if data_store and data_store.has("current_dataset"):
                df = data_store.get("current_dataset")
                logger.info("Creating visualization from current dataset in pipeline context")
            else:
                return "Error: No dataset loaded. Please load a dataset first using the load_dataset tool."
        else:
            df = load_or_get_dataframe(file_path, prefer_preprocessed=False, data_store=data_store)
            logger.info(f"Creating visualization from dataset: {file_path}")

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        insights = []
        columns_plotted = []

        if plot_type == "distribution":
            # Distribution plots for numerical columns
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

            if columns:
                numerical_cols = [col for col in columns if col in numerical_cols]

            if not numerical_cols:
                return "No numerical columns found for distribution plot"

            n_cols = min(len(numerical_cols), 6)  # Limit to 6 subplots
            n_rows = (n_cols + 2) // 3
            fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_cols > 1 else [axes]

            for i, col in enumerate(numerical_cols[:n_cols]):
                df[col].hist(bins=30, ax=axes[i], edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')

                # Generate insight
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    insights.append(f"{col} shows {'right' if skewness > 0 else 'left'} skewness ({skewness:.2f})")

            # Hide empty subplots
            for i in range(n_cols, len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            columns_plotted = numerical_cols[:n_cols]

        elif plot_type == "correlation":
            # Correlation heatmap
            numerical_cols = df.select_dtypes(include=['number']).columns

            if columns:
                numerical_cols = [col for col in columns if col in numerical_cols]

            if len(numerical_cols) < 2:
                return "Need at least 2 numerical columns for correlation plot"

            corr_matrix = df[numerical_cols].corr()

            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, square=True, linewidths=1)
            plt.title('Correlation Heatmap')
            plt.tight_layout()

            # Generate insights
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr.append(f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]}: {corr_val:.2f}")

            if high_corr:
                insights.append(f"Strong correlations found: {', '.join(high_corr[:3])}")

            columns_plotted = list(numerical_cols)

        elif plot_type == "scatter":
            # Scatter plot
            if not columns or len(columns) < 2:
                numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
                if len(numerical_cols) < 2:
                    return "Need at least 2 numerical columns for scatter plot"
                columns = numerical_cols[:2]

            x_col, y_col = columns[0], columns[1]

            plt.figure(figsize=(10, 8))
            if target_column and target_column in df.columns:
                scatter = plt.scatter(df[x_col], df[y_col], c=df[target_column],
                                    cmap='viridis', alpha=0.6)
                plt.colorbar(scatter, label=target_column)
            else:
                plt.scatter(df[x_col], df[y_col], alpha=0.6)

            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f'Scatter Plot: {x_col} vs {y_col}')
            plt.tight_layout()

            # Generate insight
            correlation = df[x_col].corr(df[y_col])
            insights.append(f"Correlation between {x_col} and {y_col}: {correlation:.2f}")

            columns_plotted = [x_col, y_col]

        elif plot_type == "box":
            # Box plot for outlier detection
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

            if columns:
                numerical_cols = [col for col in columns if col in numerical_cols]

            if not numerical_cols:
                return "No numerical columns found for box plot"

            n_cols = min(len(numerical_cols), 6)
            plt.figure(figsize=(15, 8))
            df[numerical_cols[:n_cols]].boxplot()
            plt.xticks(rotation=45)
            plt.title('Box Plot - Outlier Detection')
            plt.ylabel('Value')
            plt.tight_layout()

            # Generate insights
            for col in numerical_cols[:n_cols]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
                if outliers > 0:
                    insights.append(f"{col}: {outliers} outliers detected ({outliers/len(df)*100:.1f}%)")

            columns_plotted = numerical_cols[:n_cols]

        elif plot_type == "bar":
            # Bar chart for categorical data
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

            if columns:
                categorical_cols = [col for col in columns if col in categorical_cols]

            if not categorical_cols:
                return "No categorical columns found for bar chart"

            col = categorical_cols[0]
            value_counts = df[col].value_counts().head(10)  # Top 10

            plt.figure(figsize=(12, 8))
            value_counts.plot(kind='bar')
            plt.title(f'Bar Chart: {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Generate insight
            total = value_counts.sum()
            top_percent = value_counts.iloc[0] / total * 100
            insights.append(f"Most common value in {col}: '{value_counts.index[0]}' ({top_percent:.1f}%)")

            columns_plotted = [col]

        elif plot_type == "pairplot":
            # Pairwise relationships
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

            if columns:
                numerical_cols = [col for col in columns if col in numerical_cols]

            # Limit to 5 columns to avoid too large plots
            numerical_cols = numerical_cols[:5]

            if len(numerical_cols) < 2:
                return "Need at least 2 numerical columns for pair plot"

            if target_column and target_column in df.columns:
                pairplot_df = df[numerical_cols + [target_column]]
                sns.pairplot(pairplot_df, hue=target_column)
            else:
                pairplot_df = df[numerical_cols]
                sns.pairplot(pairplot_df)

            plt.suptitle('Pairwise Relationships', y=1.02)

            insights.append(f"Pairplot created for {len(numerical_cols)} numerical columns")
            columns_plotted = numerical_cols

        else:
            return f"Unknown plot type: {plot_type}"

        # Save or encode plot
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plot_base64 = None
        else:
            # Encode to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.read()).decode()
            buffer.close()

        plt.close('all')

        result = {
            "plot_type": plot_type,
            "columns_plotted": columns_plotted,
            "output_path": str(output_path) if output_path else None,
            "plot_base64": plot_base64,
            "insights": insights,
        }

        return result

    except Exception as e:
        plt.close('all')
        return f"Error creating visualization: {str(e)}"
