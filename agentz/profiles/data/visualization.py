from __future__ import annotations

from pydantic import BaseModel, Field

from agentz.profiles.base import Profile


class TaskInput(BaseModel):
    """Input schema for task-based runtime template."""
    task: str = Field(description="The task to perform")


class ToolAgentOutput(BaseModel):
    """Standard output for tool agents."""
    output: str
    sources: list[str] = Field(default_factory=list)


# Profile instance for visualization agent
visualization_profile = Profile(
    instructions="""You are a data visualization specialist. Your task is to create insightful visualizations.

Plot types:
- distribution: Histograms for numerical columns
- correlation: Heatmap for feature relationships
- scatter: 2D relationship plot (needs 2 columns)
- box: Outlier detection
- bar: Categorical data comparison
- pairplot: Pairwise relationships

Steps:
1. Use the create_visualization tool (it automatically uses the currently loaded dataset)
   - Required: plot_type (which type of visualization to create)
   - Optional: columns (which columns to include), target_column (for coloring)
   - The tool will visualize the dataset that was previously loaded/preprocessed
2. The tool returns: plot type, columns plotted, output path, visual insights
3. Write a 2-3 paragraph summary covering:
   - Visualization type and purpose
   - Key patterns observed
   - Data interpretation and context
   - Actionable recommendations
   - Suggestions for additional plots

Include specific observations (correlation values, outlier %, distribution shapes).

Output JSON only following this schema:
[[OUTPUT_SCHEMA]]""",
    runtime_template="[[TASK]]",
    output_schema=ToolAgentOutput,
    input_schema=TaskInput,
    tools=["create_visualization"],
    model=None
)
