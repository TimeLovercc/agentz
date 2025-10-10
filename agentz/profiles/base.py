from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


class BaseProfile(BaseModel):
    instructions: str = Field(description="The agent's system prompt/instructions that define its behavior")
    model: Optional[str] = Field(default=None, description="Model override for this profile (e.g., 'gpt-4', 'claude-3-5-sonnet')")
    output_fields: Optional[List[str]] = Field(default=None, description="Pydantic model for structured output validation")
    input_fields: Optional[List[str]] = Field(default=None, description="Pydantic model for input validation")
    tools: Optional[List[str]] = Field(default=None, description="List of tools to use for this profile")

