"""Simple AIOps Agent for AIOpsLab.

A streamlined agent that focuses on:
- Clear problem analysis
- Systematic troubleshooting
- Effective solution implementation
"""

import asyncio
import json
from typing import Dict, Any

from aiopslab.orchestrator import Orchestrator
from clients.utils.llm import GPT4Turbo
from clients.utils.templates import DOCS


class SimpleAgent:
    """A simple, effective AIOps agent."""
    
    def __init__(self):
        self.history = []
        self.llm = GPT4Turbo()
        self.step_count = 0
        
    def init_context(self, problem_desc: str, instructions: str, apis: str):
        """Initialize the context for the agent."""
        
        # Categorize APIs
        self.shell_api = self._filter_dict(apis, lambda k, _: "exec_shell" in k)
        self.submit_api = self._filter_dict(apis, lambda k, _: "submit" in k)
        self.telemetry_apis = self._filter_dict(
            apis, lambda k, _: "exec_shell" not in k and "submit" not in k
        )
        
        # Create system message using the standard template
        stringify_apis = lambda apis: "\n\n".join([f"{k}\n{v}" for k, v in apis.items()])
        
        self.system_message = DOCS.format(
            prob_desc=problem_desc,
            telemetry_apis=stringify_apis(self.telemetry_apis),
            shell_api=stringify_apis(self.shell_api),
            submit_api=stringify_apis(self.submit_api),
        )
        
        # Add enhanced instructions
        enhanced_instructions = f"""
{instructions}

PROBLEM-SOLVING APPROACH:
1. First, understand the problem by analyzing the description and any immediate symptoms
2. Gather telemetry data to understand the current system state
3. Identify potential root causes based on the data
4. Implement targeted solutions step by step
5. Validate that your solution addresses the root cause
6. Submit your solution when confident it resolves the issue

IMPORTANT GUIDELINES:
- Be systematic in your approach
- Always explain your reasoning
- Don't rush to solutions without understanding the problem
- Use telemetry data to guide your decisions
- Test your solutions before submitting
"""
        
        # Initialize conversation
        self.history.append({"role": "system", "content": self.system_message})
        self.history.append({"role": "user", "content": enhanced_instructions})
        
    async def get_action(self, input_data: str) -> str:
        """
        Main interface method for the agent.
        
        Args:
            input_data (str): Input from the orchestrator/environment
            
        Returns:
            str: The agent's response/action
        """
        
        self.step_count += 1
        
        # Add step context to input
        contextual_input = f"""
Step {self.step_count}:
{input_data}

Remember to:
- Think through the problem step by step
- Use "Thought:" to explain your reasoning
- Use "Action:" to specify your next action
- Be specific about what you're trying to achieve
"""
        
        # Add to history
        self.history.append({"role": "user", "content": contextual_input})
        
        # Get LLM response
        response = self.llm.run(self.history)
        result = response[0] if isinstance(response, list) else response
        
        # Add response to history
        self.history.append({"role": "assistant", "content": result})
        
        return result
    
    def _filter_dict(self, dictionary: Dict, filter_func) -> Dict:
        """Filter dictionary based on a function."""
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}


# Example usage
if __name__ == "__main__":
    async def test_simple_agent():
        """Test the simple agent."""
        
        agent = SimpleAgent()
        orchestrator = Orchestrator()
        orchestrator.register_agent(agent, name="simple-aiops-agent")
        
        try:
            pid = "misconfig_app_hotel_res-mitigation-1"
            problem_desc, instructions, apis = orchestrator.init_problem(pid)
            agent.init_context(problem_desc, instructions, apis)
            
            print(f"Simple agent initialized with problem: {pid}")
            
            # Start the problem solving process
            await orchestrator.start_problem(max_steps=10)
            
        except Exception as e:
            print(f"Error during testing: {e}")
            print("This is expected if the full AIOpsLab environment is not set up.")
    
    # Run the test
    asyncio.run(test_simple_agent())
