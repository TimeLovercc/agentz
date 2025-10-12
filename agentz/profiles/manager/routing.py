from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

from agentz.profiles.base import Profile


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


class RoutingInput(BaseModel):
    """Input schema for routing agent runtime template."""
    query: str = Field(description="Original user query")
    gap: str = Field(description="Knowledge gap to address")
    history: str = Field(description="History of actions, findings and thoughts")


# Profile instance for routing agent
routing_profile = Profile(
    instructions="""You are a task routing agent. Your role is to analyze knowledge gaps and route appropriate tasks to specialized agents.

Available agents: data_loader_agent, data_analysis_agent, preprocessing_agent, model_training_agent, evaluation_agent, visualization_agent, code_generation_agent, research_agent

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

Create a routing plan with appropriate agents and tasks to address the knowledge gap.""",
    runtime_template="""ORIGINAL QUERY:
[[QUERY]]

KNOWLEDGE GAP TO ADDRESS:
[[GAP]]


HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
[[HISTORY]]""",
    output_schema=AgentSelectionPlan,
    input_schema=RoutingInput,
    tools=None,
    model=None
)
