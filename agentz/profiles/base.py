from __future__ import annotations
import re
from typing import Optional, List, Type, Set
from pydantic import BaseModel, Field, model_validator


class Profile(BaseModel):
    instructions: str = Field(description="The agent's system prompt/instructions that define its behavior")
    runtime_template: str = Field(description="The runtime template for the agent's behavior")
    model: Optional[str] = Field(default=None, description="Model override for this profile (e.g., 'gpt-4', 'claude-3-5-sonnet')")
    output_schema: Optional[Type[BaseModel]] = Field(default=None, description="Pydantic model class for structured output validation")
    input_schema: Optional[Type[BaseModel]] = Field(default=None, description="Pydantic model class for input validation")
    tools: Optional[List[str]] = Field(default=None, description="List of tools to use for this profile")

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def validate_runtime_template_placeholders(self) -> 'Profile':
        """Validate that all placeholders in runtime_template match fields in input_schema."""
        if not self.runtime_template or not self.input_schema:
            return self

        # Extract placeholders from runtime_template (format: [[FIELD_NAME]])
        placeholder_pattern = r'\[\[([A-Z_]+)\]\]'
        placeholders: Set[str] = set(re.findall(placeholder_pattern, self.runtime_template))

        # Get field names from input_schema and convert to uppercase
        schema_fields: Set[str] = {field_name.upper() for field_name in self.input_schema.model_fields.keys()}

        # Check for mismatches
        missing_in_schema = placeholders - schema_fields

        if missing_in_schema:
            raise ValueError(
                f"Runtime template contains placeholders that don't match input_schema fields: "
                f"{missing_in_schema}. Available fields in input_schema (uppercase): {schema_fields}"
            )

        return self
    

