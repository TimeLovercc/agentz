from __future__ import annotations

from typing import Dict

from agents import Agent
from agentz.llm.llm_setup import LLMConfig
from agentz.agents.worker_agents.data_loader_agent import create_data_loader_agent
from agentz.agents.worker_agents.data_analysis_agent import create_data_analysis_agent
from agentz.agents.worker_agents.preprocessing_agent import create_preprocessing_agent
from agentz.agents.worker_agents.model_training_agent import create_model_training_agent
from agentz.agents.worker_agents.evaluation_agent import create_evaluation_agent
from agentz.agents.worker_agents.visualization_agent import create_visualization_agent
from agentz.agents.worker_agents.code_generation_agent import create_code_generation_agent


# Map worker agent names to their factory functions
AGENT_FACTORIES: Dict[str, callable] = {
    "data_loader_agent": create_data_loader_agent,
    "data_analysis_agent": create_data_analysis_agent,
    "preprocessing_agent": create_preprocessing_agent,
    "model_training_agent": create_model_training_agent,
    "evaluation_agent": create_evaluation_agent,
    "visualization_agent": create_visualization_agent,
    "code_generation_agent": create_code_generation_agent,
}
