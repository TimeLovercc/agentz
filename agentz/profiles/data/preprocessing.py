from __future__ import annotations

from pydantic import BaseModel, Field

from agentz.profiles.base import Profile, ToolAgentOutput
from agentz.tools.data_tools.preprocessing import preprocess_data


class TaskInput(BaseModel):
    """Input schema for task-based runtime template."""
    task: str = Field(description="The task to perform")


# Profile instance for preprocessing agent
preprocessing_profile = Profile(
    instructions="""You are a data preprocessing specialist. Your task is to clean and transform datasets.

Available operations:
- handle_missing: Fill missing values (mean/median/mode)
- remove_duplicates: Remove duplicate rows
- encode_categorical: Encode categorical variables
- scale_standard: Z-score normalization
- scale_minmax: Min-max scaling [0, 1]
- remove_outliers: IQR method
- feature_engineering: Create interaction features

Steps:
1. Use the preprocess_data tool (it automatically uses the currently loaded dataset)
   - Required: operations list (which operations to perform)
   - Optional: target_column (if mentioned in the task)
   - The tool will preprocess the dataset that was previously loaded
2. The tool returns: operations applied, shape changes, summary of changes
3. Write a 2-3 paragraph summary covering:
   - Operations performed and justification
   - Shape changes and data modifications
   - Impact on data quality
   - Next steps (modeling, further preprocessing)

Include specific numbers (rows removed, values filled, etc.).

Output JSON only following this schema:
[[OUTPUT_SCHEMA]]""",
    runtime_template="[[TASK]]",
    output_schema=ToolAgentOutput,
    input_schema=TaskInput,
    tools=[preprocess_data],
    model=None
)
