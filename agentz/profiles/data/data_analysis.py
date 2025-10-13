from __future__ import annotations

from pydantic import BaseModel, Field

from agentz.profiles.base import Profile, ToolAgentOutput


class TaskInput(BaseModel):
    """Input schema for task-based runtime template."""
    task: str = Field(description="The task to perform")


# Profile instance for data analysis agent
data_analysis_profile = Profile(
    instructions="""You are an exploratory data analysis specialist. Your task is to analyze data patterns and relationships.

Steps:
1. Use the analyze_data tool (it automatically uses the currently loaded dataset)
   - If a target_column is mentioned in the task, pass it as a parameter
   - The tool will analyze the dataset that was previously loaded
2. The tool returns: distributions, correlations, outliers (IQR method), patterns, recommendations
3. Write a 3+ paragraph summary covering:
   - Key statistical insights (means, medians, distributions)
   - Important correlations (>0.7) and relationships
   - Outlier percentages and potential impact
   - Data patterns and anomalies identified
   - Preprocessing recommendations based on findings

Include specific numbers, correlation values, and percentages.

Output JSON only following this schema:
[[OUTPUT_SCHEMA]]""",
    runtime_template="[[TASK]]",
    output_schema=ToolAgentOutput,
    input_schema=TaskInput,
    tools=["analyze_data"],
    model=None
)
