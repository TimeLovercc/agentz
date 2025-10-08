from __future__ import annotations

import asyncio
import time
from typing import Any, List
from loguru import logger

from agentz.agents.manager_agents.routing_agent import AgentTask
from agentz.agents.registry import create_agents
from agentz.flow import auto_trace
from agentz.memory.global_memory import global_memory
from agentz.memory.conversation import Conversation
from pipelines.base import BasePipeline

class DebuggingPipeline(BasePipeline):
    """Main pipeline orchestrator for debugging a single agent."""

    def __init__(self, config):
        super().__init__(config)

        self.agent = create_agents("chrome_agent", config)
        self.conversation = Conversation()

    @auto_trace
    async def run(self):
        logger.info(f"Debugging agent: {self.agent.name}")
        logger.info(f"User prompt: {self.config.prompt}")
        self.iteration = 0
        query = self.prepare_query(
            content=f"Task: {self.config.prompt}\n"
        )

        await self.agent.run(query = query)