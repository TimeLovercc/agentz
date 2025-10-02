from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from loguru import logger

from agents import Agent
from agentz.llm.llm_setup import LLMConfig


class AgentTask(BaseModel):
    """Task definition for routing to specific agents."""
    agent: str = Field(description="Name of the agent to use")
    query: str = Field(description="Query/task for the agent")
    gap: str = Field(description="The knowledge gap this task addresses")
    entity_website: Optional[str] = Field(description="Optional entity or website context", default=None)


class AgentSelectionPlan(BaseModel):
    """Plan containing multiple agent tasks to address knowledge gaps."""
    tasks: List[AgentTask] = Field(description="List of tasks for different agents", default_factory=list)
    reasoning: str = Field(description="Reasoning for the agent selection", default="")


def create_routing_agent(config: LLMConfig, full_config: Optional[Dict[str, Any]] = None) -> Agent:
    """Create a routing agent using OpenAI Agents SDK.

    Args:
        config: LLM configuration
        full_config: Optional full config dictionary with agent prompts

    Returns:
        Agent instance configured for task routing
    """

    available_agents = [
        "data_loader_agent",
        "data_analysis_agent",
        "preprocessing_agent",
        "model_training_agent",
        "evaluation_agent",
        "visualization_agent",
        "code_generation_agent",
        "research_agent"
    ]

    available_agents_str = ", ".join(available_agents)

    default_instructions = f"""You are a task routing agent. Your role is to analyze knowledge gaps and route appropriate tasks to specialized agents.

Available agents: {available_agents_str}

Agent capabilities:
- data_loader_agent: Load and inspect datasets, understand data structure
- data_analysis_agent: Perform exploratory data analysis, statistical analysis
- preprocessing_agent: Clean data, handle missing values, feature engineering
- model_training_agent: Train machine learning models, hyperparameter tuning
- evaluation_agent: Evaluate model performance, generate metrics
- visualization_agent: Create charts, plots, and visualizations
- code_generation_agent: Generate code snippets and complete implementations
- research_agent: Research methodologies, best practices, domain knowledge

Your task:
1. Analyze the knowledge gap that needs to be addressed
2. Select the most appropriate agent(s) to handle the gap
3. Create specific, actionable tasks for each selected agent
4. Ensure tasks are clear and focused

Create a routing plan with appropriate agents and tasks to address the knowledge gap."""

    instructions = default_instructions
    if full_config:
        instructions = full_config.get('agents', {}).get('routing_agent', {}).get('instructions', default_instructions)

    agent = Agent(
        name="Task Router",
        instructions=instructions,
        output_type=AgentSelectionPlan,
        model=config.main_model
    )

    logger.info("Created RoutingAgent")
    return agent