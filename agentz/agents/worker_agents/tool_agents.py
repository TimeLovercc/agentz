from __future__ import annotations

import json
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from loguru import logger

from agents import Agent, Runner
from agentz.llm.llm_setup import LLMConfig


class ToolAgentOutput(BaseModel):
    """Standard output format for tool agent results."""
    output: str = Field(description="The main output/result from the tool agent")
    sources: List[str] = Field(description="List of sources or references used", default_factory=list)
    metadata: Dict[str, Any] = Field(description="Additional metadata about the execution", default_factory=dict)


# Create specialized agents using OpenAI Agents SDK
class DataLoaderAgent:
    """Agent for loading and inspecting datasets."""

    def __init__(self, config: LLMConfig):
        self.config = config

        if not config.full_config:
            raise ValueError("Agent instructions for 'data_loader_agent' not found in config. Please provide config_file with agent instructions.")

        instructions = config.full_config.get('agents', {}).get('tool_agents', {}).get('data_loader_agent', {}).get('instructions')
        if not instructions:
            raise ValueError("Agent instructions for 'data_loader_agent' not found in config. Please provide config_file with agent instructions.")

        self.agent = Agent(
            name="Data Loader",
            instructions=instructions,
            model=config.main_model
        )

    async def run(self, task_json: str) -> ToolAgentOutput:
        """Execute data loading task."""
        try:
            task = json.loads(task_json)
            query = task.get("query", "Load and inspect dataset")

            result = await Runner.run(self.agent, f"Task: {query}")

            return ToolAgentOutput(
                output=result.final_output,
                sources=["pandas", "numpy"],
                metadata={"agent": "data_loader_agent", "task": query}
            )

        except Exception as e:
            return ToolAgentOutput(
                output=f"Error in data loading: {str(e)}",
                sources=[],
                metadata={"error": str(e)}
            )


class DataAnalysisAgent:
    """Agent for exploratory data analysis."""

    def __init__(self, config: LLMConfig):
        self.config = config

        if not config.full_config:
            raise ValueError("Agent instructions for 'data_analysis_agent' not found in config. Please provide config_file with agent instructions.")

        instructions = config.full_config.get('agents', {}).get('tool_agents', {}).get('data_analysis_agent', {}).get('instructions')
        if not instructions:
            raise ValueError("Agent instructions for 'data_analysis_agent' not found in config. Please provide config_file with agent instructions.")

        self.agent = Agent(
            name="Data Analyst",
            instructions=instructions,
            model=config.main_model
        )

    async def run(self, task_json: str) -> ToolAgentOutput:
        """Execute data analysis task."""
        try:
            task = json.loads(task_json)
            query = task.get("query", "Perform exploratory data analysis")

            result = await Runner.run(self.agent, f"Task: {query}")

            return ToolAgentOutput(
                output=result.final_output,
                sources=["pandas", "matplotlib", "seaborn", "numpy"],
                metadata={"agent": "data_analysis_agent", "task": query}
            )

        except Exception as e:
            return ToolAgentOutput(
                output=f"Error in data analysis: {str(e)}",
                sources=[],
                metadata={"error": str(e)}
            )


class ModelTrainingAgent:
    """Agent for machine learning model training."""

    def __init__(self, config: LLMConfig):
        self.config = config

        if not config.full_config:
            raise ValueError("Agent instructions for 'model_training_agent' not found in config. Please provide config_file with agent instructions.")

        instructions = config.full_config.get('agents', {}).get('tool_agents', {}).get('model_training_agent', {}).get('instructions')
        if not instructions:
            raise ValueError("Agent instructions for 'model_training_agent' not found in config. Please provide config_file with agent instructions.")

        self.agent = Agent(
            name="ML Trainer",
            instructions=instructions,
            model=config.main_model
        )

    async def run(self, task_json: str) -> ToolAgentOutput:
        """Execute model training task."""
        try:
            task = json.loads(task_json)
            query = task.get("query", "Train machine learning models")

            result = await Runner.run(self.agent, f"Task: {query}")

            return ToolAgentOutput(
                output=result.final_output,
                sources=["scikit-learn", "pandas", "numpy"],
                metadata={"agent": "model_training_agent", "task": query}
            )

        except Exception as e:
            return ToolAgentOutput(
                output=f"Error in model training: {str(e)}",
                sources=[],
                metadata={"error": str(e)}
            )


class CodeGenerationAgent:
    """Agent for generating complete code solutions."""

    def __init__(self, config: LLMConfig):
        self.config = config

        if not config.full_config:
            raise ValueError("Agent instructions for 'code_generation_agent' not found in config. Please provide config_file with agent instructions.")

        instructions = config.full_config.get('agents', {}).get('tool_agents', {}).get('code_generation_agent', {}).get('instructions')
        if not instructions:
            raise ValueError("Agent instructions for 'code_generation_agent' not found in config. Please provide config_file with agent instructions.")

        self.agent = Agent(
            name="Code Generator",
            instructions=instructions,
            model=config.main_model
        )

    async def run(self, task_json: str) -> ToolAgentOutput:
        """Execute code generation task."""
        try:
            task = json.loads(task_json)
            query = task.get("query", "Generate complete data science solution")

            result = await Runner.run(self.agent, f"Task: {query}")

            return ToolAgentOutput(
                output=result.final_output,
                sources=["pandas", "scikit-learn", "matplotlib", "seaborn", "numpy"],
                metadata={"agent": "code_generation_agent", "task": query}
            )

        except Exception as e:
            return ToolAgentOutput(
                output=f"Error in code generation: {str(e)}",
                sources=[],
                metadata={"error": str(e)}
            )


def init_tool_agents(config: LLMConfig) -> Dict[str, Any]:
    """
    Initialize all available tool agents.

    Args:
        config: LLM configuration with full_config containing agent prompts

    Returns:
        Dictionary mapping agent names to agent instances
    """
    agents = {
        "data_loader_agent": DataLoaderAgent(config),
        "data_analysis_agent": DataAnalysisAgent(config),
        "preprocessing_agent": DataAnalysisAgent(config),  # Reuse for preprocessing
        "model_training_agent": ModelTrainingAgent(config),
        "evaluation_agent": ModelTrainingAgent(config),  # Reuse for evaluation
        "visualization_agent": DataAnalysisAgent(config),  # Reuse for visualization
        "code_generation_agent": CodeGenerationAgent(config),
        "research_agent": CodeGenerationAgent(config),  # Reuse for research
    }

    logger.info(f"Initialized {len(agents)} tool agents")
    return agents