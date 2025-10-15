"""Vanilla Chat Pipeline - Simple conversational pipeline with a single chat agent."""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel

from agentz.agent.base import ContextAgent
from agentz.context.context import Context
from pipelines.base import BasePipeline


class ChatQuery(BaseModel):
    """Query model for vanilla chat pipeline."""
    prompt: str

    def format(self) -> str:
        """Return the prompt as-is."""
        return self.prompt


class VanillaChatPipeline(BasePipeline):
    """Simple single-pass pipeline with just a chat agent."""

    def __init__(self, config):
        super().__init__(config)

        # Initialize shared context
        self.context = Context(["profiles", "states"])
        llm = self.config.llm.main_model

        # Bind vanilla chat agent from registered profile
        self.chat_agent = ContextAgent.from_profile(self, "vanilla_chat", llm)

    async def initialize_pipeline(self, query: Any) -> None:
        """Ensure a ChatQuery is available when the pipeline starts."""
        if query is None:
            prompt = self.config.prompt or "Hello!"
            query = ChatQuery(prompt=prompt)
        elif not isinstance(query, ChatQuery):
            # Coerce arbitrary input into ChatQuery
            prompt = getattr(query, "prompt", None) or str(query)
            query = ChatQuery(prompt=prompt)

        await super().initialize_pipeline(query)

    async def execute(self) -> Any:
        """Execute the vanilla chat agent directly."""
        logger.info(f"User message: {self.config.prompt}")

        # Start single iteration for structured logging
        _, group_id = self.begin_iteration(title="Chat")
        try:
            # Prepare input for chat agent
            chat_input = self.chat_agent.input_model(
                message=self.context.state.query or ""
            )

            # Execute chat agent
            result = await self.chat_agent(chat_input, group_id=group_id)

            # Store result
            if self.state:
                self.state.final_report = result.response
                self.state.mark_research_complete()

            logger.info("Vanilla chat pipeline completed")
            return result
        finally:
            self.end_iteration(group_id)

    async def finalize(self, result: Any) -> Any:
        """Return the chat output directly."""
        return result
