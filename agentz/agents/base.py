from typing import Any, Callable, Optional
from agents import Agent, Runner, RunResult
from agents.run_context import TContext


class ResearchAgent(Agent[TContext]):
    """
    This is a custom implementation of the OpenAI Agent class that supports output parsing
    for models that don't support structured output types. The user can specify an output_parser
    function that will be called with the raw output from the agent. This can run custom logic 
    such as cleaning up the output and converting it to a structured JSON object.

    Needs to be run with the ResearchRunner to work.
    """
    
    def __init__(
        self,
        *args,
        output_parser: Optional[Callable[[str], Any]] = None,
        **kwargs
    ):
        # The output_parser is a function that only takes effect if output_type is not specified
        self.output_parser = output_parser

        # If both are specified, we raise an error - they can't be used together
        if self.output_parser and kwargs.get('output_type'):
            raise ValueError("Cannot specify both output_parser and output_type")
            
        super().__init__(*args, **kwargs)
    

    async def parse_output(self, run_result: RunResult) -> RunResult:
        """
        Process the RunResult by applying the output_parser to its final_output if specified.
        This preserves the RunResult structure while modifying its content.
        """
        if self.output_parser:
            raw_output = run_result.final_output            
            parsed_output = self.output_parser(raw_output)
            run_result.final_output = parsed_output            
        return run_result
    

class ResearchRunner(Runner):
    """
    Custom implementation of the OpenAI Runner class that supports output parsing
    for models that don't support structured output types with tools. 
    
    Needs to be run with the ResearchAgent class.
    """
    
    @classmethod
    async def run(cls, *args, **kwargs) -> RunResult:
        """
        Run the agent and process its output with the custom parser if applicable.
        """
        # Call the original run method
        result = await Runner.run(*args, **kwargs)
        
        # Get the starting agent
        starting_agent = kwargs.get('starting_agent') or args[0]
        
        # If the starting agent is of type ResearchAgent, parse the output
        if isinstance(starting_agent, ResearchAgent):
            return await starting_agent.parse_output(result)
        
        return result