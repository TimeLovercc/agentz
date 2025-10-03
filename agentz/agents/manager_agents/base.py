from __future__ import annotations

from typing import Dict

from agents import Agent
from agentz.llm.llm_setup import LLMConfig
from agentz.agents.manager_agents.observe_agent import create_observe_agent
from agentz.agents.manager_agents.evaluate_agent import create_evaluate_agent
from agentz.agents.manager_agents.routing_agent import create_routing_agent
from agentz.agents.manager_agents.writer_agent import create_writer_agent


# Map manager agent names to their factory functions
MANAGER_AGENT_FACTORIES: Dict[str, callable] = {
    "observe_agent": create_observe_agent,
    "evaluate_agent": create_evaluate_agent,
    "routing_agent": create_routing_agent,
    "writer_agent": create_writer_agent,
}
