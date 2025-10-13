from __future__ import annotations

from pydantic import BaseModel, Field

from agentz.profiles.base import Profile, ToolAgentOutput


class TaskInput(BaseModel):
    """Input schema for task-based runtime template."""
    task: str = Field(description="The task to perform")


# Profile instance for web searcher agent
web_searcher_agent_profile = Profile(
    instructions="""You are a web search specialist. Your task is to search the web for information.

Steps:
1. Use the web_search tool with the provided query
2. The tool returns: web_search_results
3. Write a 2-3 paragraph summary covering:
   - Web search results
   - Key information and initial observations

Output JSON only following this schema:
[[OUTPUT_SCHEMA]]""",
    runtime_template="[[TASK]]",
    output_schema=ToolAgentOutput,
    input_schema=TaskInput,
    tools=["web_search"],
    model=None
)
