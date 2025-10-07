from typing import List, Optional

from pydantic import BaseModel, Field


class IterationData(BaseModel):
    """Data for a single iteration of the research loop."""
    gap: str = Field(description="The gap addressed in the iteration", default_factory=list)
    tool_calls: List[str] = Field(description="The tool calls made", default_factory=list)
    findings: List[str] = Field(description="The findings collected from tool calls", default_factory=list)
    thought: List[str] = Field(description="The thinking done to reflect on the success of the iteration and next steps", default_factory=list)
    summarized: bool = Field(description="Whether the iteration has been summarized", default=False)


class Conversation(BaseModel):
    """A conversation between the user and the iterative researcher."""
    history: List[IterationData] = Field(description="The data for each iteration of the research loop", default_factory=list)
    summary: str = Field(description="The summary of the conversation", default_factory=list)

    def add_iteration(self, iteration_data: Optional[IterationData] = None):
        if iteration_data is None:
            iteration_data = IterationData()
        self.history.append(iteration_data)

    def set_latest_gap(self, gap: str):
        self.history[-1].gap = gap

    def set_latest_tool_calls(self, tool_calls: List[str]):
        self.history[-1].tool_calls = tool_calls

    def set_latest_findings(self, findings: List[str]):
        self.history[-1].findings = findings

    def set_latest_thought(self, thought: str):
        self.history[-1].thought = thought

    def get_latest_gap(self) -> str:
        return self.history[-1].gap

    def get_latest_tool_calls(self) -> List[str]:
        return self.history[-1].tool_calls

    def get_latest_findings(self) -> List[str]:
        return self.history[-1].findings

    def get_latest_thought(self) -> str:
        return self.history[-1].thought

    def get_all_findings(self) -> List[str]:
        return [finding for iteration_data in self.history for finding in iteration_data.findings]

    def compile_conversation_history(self) -> str:
        """Compile the conversation history into a string."""
        conversation = ""
        for iteration_num, iteration_data in enumerate(self.history):
            conversation += f"[ITERATION {iteration_num + 1}]\n\n"
            if iteration_data.thought:
                conversation += f"{self.get_thought_string(iteration_num)}\n\n"
            if iteration_data.gap:
                conversation += f"{self.get_task_string(iteration_num)}\n\n"
            if iteration_data.tool_calls:
                conversation += f"{self.get_action_string(iteration_num)}\n\n"
            if iteration_data.findings:
                conversation += f"{self.get_findings_string(iteration_num)}\n\n"

        return conversation

    def compile_unsummarized_conversation_history(self) -> str:
        """Compile the conversation history into a string."""
        conversation = ""
        for iteration_num, iteration_data in enumerate(self.history):
            if not iteration_data.summarized:
                conversation += f"[ITERATION {iteration_num + 1}]\n\n"
                if iteration_data.thought:
                    conversation += f"{self.get_thought_string(iteration_num)}\n\n"
                if iteration_data.gap:
                    conversation += f"{self.get_task_string(iteration_num)}\n\n"
                if iteration_data.tool_calls:
                    conversation += f"{self.get_action_string(iteration_num)}\n\n"
                if iteration_data.findings:
                    conversation += f"{self.get_findings_string(iteration_num)}\n\n"
        return conversation
    
    def compile_conversation_history_with_summary(self) -> str:
        """Compile the conversation history into a string."""
        summary = self.get_latest_summary()
        conversation = f"[SUMMARY BEFORE NEW ITERATION]\n\n{summary}\n\n"
        conversation += self.compile_unsummarized_conversation_history()
        return conversation

    def get_latest_summary(self) -> str:
        """Get the latest summary."""
        return self.summary
    
    def set_latest_summary(self, summary: str):
        """Set the latest summary."""
        self.summary = summary
        for iteration_data in self.history:
            iteration_data.summarized = True

    def get_task_string(self, iteration_num: int) -> str:
        """Get the task for the current iteration."""
        if self.history[iteration_num].gap:
            return f"<task>\nAddress this knowledge gap: {self.history[iteration_num].gap}\n</task>"
        return ""

    def get_action_string(self, iteration_num: int) -> str:
        """Get the action for the current iteration."""
        if self.history[iteration_num].tool_calls:
            joined_calls = '\n'.join(self.history[iteration_num].tool_calls)
            return (
                "<action>\nCalling the following tools to address the knowledge gap:\n"
                f"{joined_calls}\n</action>"
            )
        return ""

    def get_findings_string(self, iteration_num: int) -> str:
        """Get the findings for the current iteration."""
        if self.history[iteration_num].findings:
            joined_findings = '\n\n'.join(self.history[iteration_num].findings)
            return f"<findings>\n{joined_findings}\n</findings>"
        return ""

    def get_thought_string(self, iteration_num: int) -> str:
        """Get the thought for the current iteration."""
        if self.history[iteration_num].thought:
            return f"<thought>\n{self.history[iteration_num].thought}\n</thought>"
        return ""

    def latest_task_string(self) -> str:
        """Get the latest task."""
        return self.get_task_string(len(self.history) - 1)

    def latest_action_string(self) -> str:
        """Get the latest action."""
        return self.get_action_string(len(self.history) - 1)

    def latest_findings_string(self) -> str:
        """Get the latest findings."""
        return self.get_findings_string(len(self.history) - 1)

    def latest_thought_string(self) -> str:
        """Get the latest thought."""
        return self.get_thought_string(len(self.history) - 1)
