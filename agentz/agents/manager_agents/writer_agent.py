from __future__ import annotations

from typing import Optional, Dict, Any
from loguru import logger

from agents import Agent
from agentz.llm.llm_setup import LLMConfig


def create_writer_agent(config: LLMConfig, full_config: Optional[Dict[str, Any]] = None) -> Agent:
    """Create a writer agent using OpenAI Agents SDK.

    Args:
        config: LLM configuration
        full_config: Optional full config dictionary with agent prompts

    Returns:
        Agent instance configured for technical writing
    """

    default_instructions = """You are a technical writing agent specialized in creating comprehensive data science reports.

Your responsibilities:
1. Synthesize findings from multiple research iterations
2. Create clear, well-structured reports with proper formatting
3. Include executive summaries when appropriate
4. Present technical information in an accessible manner
5. Follow specific formatting guidelines when provided
6. Ensure all key insights and recommendations are highlighted

Report Structure Guidelines:
- Start with a clear summary of the task/objective
- Present methodology and approach
- Include key findings and insights
- Provide actionable recommendations
- Use proper markdown formatting when appropriate
- Include code examples when relevant
- Ensure technical accuracy while maintaining readability

Focus on creating professional, comprehensive reports that effectively communicate the research findings and their practical implications."""

    instructions = default_instructions
    if full_config:
        instructions = full_config.get('agents', {}).get('writer_agent', {}).get('instructions', default_instructions)

    agent = Agent(
        name="Technical Writer",
        instructions=instructions,
        model=config.main_model
    )

    logger.info("Created WriterAgent")
    return agent