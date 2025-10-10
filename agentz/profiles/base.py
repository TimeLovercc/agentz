from __future__ import annotations
from typing import Optional, List, Type
from pydantic import BaseModel, Field


class Profile(BaseModel):
    instructions: str = Field(description="The agent's system prompt/instructions that define its behavior")
    runtime_template: str = Field(description="The runtime template for the agent's behavior")
    model: Optional[str] = Field(default=None, description="Model override for this profile (e.g., 'gpt-4', 'claude-3-5-sonnet')")
    output_schema: Optional[Type[BaseModel]] = Field(default=None, description="Pydantic model class for structured output validation")
    input_schema: Optional[Type[BaseModel]] = Field(default=None, description="Pydantic model class for input validation")
    tools: Optional[List[str]] = Field(default=None, description="List of tools to use for this profile")

    class Config:
        arbitrary_types_allowed = True
    

