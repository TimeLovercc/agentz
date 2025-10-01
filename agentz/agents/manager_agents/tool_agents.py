from __future__ import annotations

import json
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from loguru import logger

from agents import Agent, Runner
from ds1.src.llm.llm_setup import LLMConfig


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

        self.agent = Agent(
            name="Data Loader",
            instructions="""You are a data loading specialist. Generate Python code and analysis for loading and inspecting datasets.

Provide:
1. Code to load the dataset (pandas, numpy, etc.)
2. Basic dataset inspection (shape, columns, data types, missing values)
3. Sample data preview
4. Summary statistics
5. Any data quality observations

Focus on comprehensive dataset understanding.""",
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
        self.agent = Agent(
            name="Data Analyst",
            instructions="""You are an exploratory data analysis specialist. Generate comprehensive Python code and analysis.

Provide:
1. Statistical summaries and distributions
2. Correlation analysis
3. Data visualization code (matplotlib, seaborn)
4. Outlier detection
5. Pattern identification
6. Insights and observations
7. Recommendations for preprocessing

Focus on uncovering insights and data characteristics.""",
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
        self.agent = Agent(
            name="ML Trainer",
            instructions="""You are a machine learning specialist. Generate comprehensive model training code and analysis.

Provide:
1. Data preprocessing and feature engineering
2. Train/validation/test splits
3. Model selection (multiple algorithms)
4. Hyperparameter tuning approaches
5. Model training code
6. Performance evaluation metrics
7. Model comparison and recommendations

Focus on best practices and comprehensive evaluation.""",
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
        self.agent = Agent(
            name="Code Generator",
            instructions="""You are a senior data scientist and software engineer. Generate complete, production-ready Python code solutions.

Provide:
1. Complete end-to-end pipeline
2. Data loading and preprocessing
3. Exploratory data analysis
4. Feature engineering
5. Model training and evaluation
6. Visualization and reporting
7. Error handling and logging
8. Clear documentation and comments

Focus on creating comprehensive, executable solutions.""",
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
        config: LLM configuration

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