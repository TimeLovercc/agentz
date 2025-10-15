"""Vanilla Chat Pipeline - Persistent conversational pipeline for multi-turn chat."""

from __future__ import annotations

import asyncio
import queue
from typing import Any, Optional

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


class VanillaChatPersistentPipeline(BasePipeline):
    """Persistent conversational pipeline that handles multiple user messages in one run."""

    def __init__(self, config):
        super().__init__(config)

        # Initialize shared context
        self.context = Context(["profiles", "states"])
        llm = self.config.llm.main_model

        # Bind vanilla chat agent from registered profile
        self.chat_agent = ContextAgent.from_profile(self, "vanilla_chat", llm)

        # Input queue for receiving user messages
        self.input_queue: queue.Queue = queue.Queue()
        self.running = True

    def add_user_message(self, message: str) -> None:
        """Add a user message to the input queue.

        Args:
            message: User's message
        """
        self.input_queue.put(message)
        logger.info(f"Message added to queue: {message}")

    def stop(self) -> None:
        """Signal the pipeline to stop after processing current message."""
        self.running = False
        self.input_queue.put(None)  # Sentinel value to unblock queue

    async def initialize_pipeline(self, query: Any) -> None:
        """Initialize with the first message if provided."""
        if query is None:
            prompt = self.config.prompt or "Hello!"
            query = ChatQuery(prompt=prompt)
        elif not isinstance(query, ChatQuery):
            prompt = getattr(query, "prompt", None) or str(query)
            query = ChatQuery(prompt=prompt)

        # Add first message to queue
        self.add_user_message(query.prompt)

        # Don't call super() with query since we're handling it differently
        self.update_printer("initialization", "Pipeline initialized", is_done=True)

    async def execute(self) -> Any:
        """Execute conversation loop - wait for user messages and respond."""
        logger.info("Starting persistent conversation loop")

        conversation_count = 0
        last_response = None

        # Override max_iterations for persistent chat
        self.max_iterations = 100  # Allow many turns

        while self.running and conversation_count < self.max_iterations:
            conversation_count += 1

            # Start iteration for this conversation turn
            _, group_id = self.begin_iteration(title=f"Turn {conversation_count}")

            try:
                # Wait for user message from queue
                logger.info("Waiting for user message...")
                self.update_printer("waiting", "Waiting for user input...", group_id=group_id)

                # Block until we get a message (or timeout)
                try:
                    user_message = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.input_queue.get,
                        True,  # block
                        60.0   # timeout after 60 seconds
                    )
                except queue.Empty:
                    logger.info("No message received within timeout")
                    break

                # Check for stop signal
                if user_message is None or not self.running:
                    logger.info("Stop signal received")
                    break

                logger.info(f"Processing message: {user_message}")
                self.update_printer("processing", f"Processing: {user_message[:50]}...", group_id=group_id, is_done=True)

                # Prepare input for chat agent
                chat_input = self.chat_agent.input_model(message=user_message)

                # Execute chat agent
                result = await self.chat_agent(chat_input, group_id=group_id)

                # Store result
                last_response = result
                if self.state:
                    self.state.final_report = result.response

                logger.info(f"Generated response: {result.response}")
                self.update_printer("response", "Response generated", group_id=group_id, is_done=True)

            finally:
                self.end_iteration(group_id)

        logger.info(f"Conversation loop ended after {conversation_count} turns")
        return last_response

    async def finalize(self, result: Any) -> Any:
        """Return the last chat output."""
        return result
