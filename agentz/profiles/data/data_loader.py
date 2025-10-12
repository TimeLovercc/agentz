from __future__ import annotations

from pydantic import BaseModel, Field

from agentz.profiles.base import Profile, ToolAgentOutput


class TaskInput(BaseModel):
    """Input schema for task-based runtime template."""
    task: str = Field(description="The task to perform")


# Profile instance for data loader agent
data_loader_profile = Profile(
    instructions="""You are a data loading specialist. Your task is to load and inspect datasets.

Steps:
1. Use the load_dataset tool with the provided file path
2. The tool returns: shape, columns, dtypes, missing values, sample data, statistics, memory usage, duplicates
3. Write a 2-3 paragraph summary covering:
   - Dataset size and structure
   - Data types and columns
   - Data quality issues (missing values, duplicates)
   - Key statistics and initial observations

Include specific numbers and percentages in your summary.

Output JSON only following this schema:
[[OUTPUT_SCHEMA]]""",
    runtime_template="[[TASK]]",
    output_schema=ToolAgentOutput,
    input_schema=TaskInput,
    tools=["load_dataset"],
    model=None
)
