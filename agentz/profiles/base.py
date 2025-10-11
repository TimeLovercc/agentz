from __future__ import annotations
import re
from typing import Optional, List, Type, Set
from pydantic import BaseModel, Field, model_validator


class ToolAgentOutput(BaseModel):
    """Standard output for all tool agents"""
    output: str
    sources: list[str] = Field(default_factory=list)


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


def load_all_profiles():
    """Load all Profile instances from the profiles package.

    Returns:
        Dict with shortened keys (e.g., "observe" instead of "observe_profile")
        Each profile has a _key attribute added for automatic name derivation
    """
    import importlib
    import inspect
    from pathlib import Path

    profiles = {}
    package_path = Path(__file__).parent

    # Recursively find all .py files in the profiles directory
    for py_file in package_path.rglob('*.py'):
        if py_file.name == 'base.py' or py_file.name.startswith('_'):
            continue

        # Convert file path to module name (need to find 'agentz' root)
        # Go up from current file: profiles/base.py -> profiles -> agentz
        agentz_root = package_path.parent
        relative_path = py_file.relative_to(agentz_root)
        module_name = 'agentz.' + str(relative_path.with_suffix('')).replace('/', '.')

        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if isinstance(obj, Profile) and not name.startswith('_'):
                    # Strip "_profile" suffix from key for cleaner access
                    key = name.replace('_profile', '') if name.endswith('_profile') else name
                    # Add _key attribute to profile for automatic name derivation
                    obj._key = key
                    profiles[key] = obj
        except Exception:
            pass

    return profiles


