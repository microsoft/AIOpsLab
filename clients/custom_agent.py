"""Custom AIOps Agent for AIOpsLab.

This agent implements a comprehensive approach to AIOps problem solving with:
- Systematic problem analysis
- Multi-step reasoning 
- Telemetry data analysis
- Iterative solution refinement
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional

from aiopslab.orchestrator import Orchestrator
from clients.utils.llm import GPT4Turbo
from clients.utils.templates import DOCS


class CustomAgent:
    """A comprehensive AIOps agent with advanced reasoning capabilities."""
    
    def __init__(self):
        self.history = []
        self.llm = GPT4Turbo()
        self.problem_context = {}
        self.analysis_steps = []
        self.solution_attempts = []
        self.current_step = 0
        self.max_analysis_steps = 10
        
    def init_context(self, problem_desc: str, instructions: str, apis: str):
        """Initialize the context for the agent."""
        
        # Store problem context
        self.problem_context = {
            'description': problem_desc,
            'instructions': instructions,
            'apis': apis
        }
        
        # Categorize APIs
        self.shell_api = self._filter_dict(apis, lambda k, _: "exec_shell" in k)
        self.submit_api = self._filter_dict(apis, lambda k, _: "submit" in k)
        self.telemetry_apis = self._filter_dict(
            apis, lambda k, _: "exec_shell" not in k and "submit" not in k
        )
        
        # Create enhanced system message
        self.system_message = self._create_enhanced_system_message(
            problem_desc, instructions, apis
        )
        
        # Initialize conversation
        self.history.append({"role": "system", "content": self.system_message})
        self.history.append({"role": "user", "content": self._create_initial_task_message()})
        
    def _create_enhanced_system_message(self, problem_desc: str, instructions: str, apis: str) -> str:
        """Create an enhanced system message with structured approach."""
        
        stringify_apis = lambda apis: "\n\n".join([f"{k}\n{v}" for k, v in apis.items()])
        
        return f"""
{problem_desc}

You are an advanced AIOps agent with the following capabilities:
1. Systematic problem analysis and diagnosis
2. Telemetry data interpretation
3. Root cause analysis
4. Solution implementation and validation
5. Iterative refinement

ANALYSIS FRAMEWORK:
1. **Problem Understanding**: Thoroughly analyze the problem description
2. **Data Collection**: Gather relevant telemetry and system data
3. **Pattern Recognition**: Identify anomalies and patterns
4. **Hypothesis Formation**: Develop potential root causes
5. **Solution Design**: Create targeted mitigation strategies
6. **Implementation**: Execute solutions systematically
7. **Validation**: Verify effectiveness and monitor results

AVAILABLE APIS:

Telemetry APIs:
{stringify_apis(self.telemetry_apis)}

Shell API:
{stringify_apis(self.shell_api)}

Submit API:
{stringify_apis(self.submit_api)}

RESPONSE FORMAT:
Always respond with structured reasoning:

Analysis: <your systematic analysis>
Hypothesis: <your current hypothesis about the root cause>
Action: <your specific action>
Rationale: <why this action addresses the hypothesis>
Expected_Outcome: <what you expect to happen>

IMPORTANT GUIDELINES:
- Take a methodical approach, don't jump to conclusions
- Always validate your hypotheses with data
- Consider multiple potential root causes
- Monitor the impact of your actions
- Be prepared to adapt your approach based on results
"""

    def _create_initial_task_message(self) -> str:
        """Create the initial task message with structured approach."""
        return f"""
{self.problem_context['instructions']}

Begin by conducting a systematic analysis of the problem:

1. Start with understanding the problem scope and impact
2. Gather baseline telemetry data
3. Identify key metrics and potential anomalies
4. Form initial hypotheses about root causes
5. Design and implement targeted solutions
6. Validate results and iterate if needed

Remember to follow the structured response format and provide clear reasoning for each step.
"""

    async def get_action(self, input_data: str) -> str:
        """
        Main interface method for the agent.
        
        Args:
            input_data (str): Input from the orchestrator/environment
            
        Returns:
            str: The agent's response/action
        """
        
        # Increment step counter
        self.current_step += 1
        
        # Add input to history
        self.history.append({"role": "user", "content": input_data})
        
        # Enhanced input processing
        processed_input = self._process_input(input_data)
        
        # Generate response with context awareness
        response = await self._generate_contextual_response(processed_input)
        
        # Add response to history
        self.history.append({"role": "assistant", "content": response})
        
        # Store analysis step
        self.analysis_steps.append({
            'step': self.current_step,
            'input': input_data,
            'response': response,
            'timestamp': asyncio.get_event_loop().time()
        })
        
        return response
    
    def _process_input(self, input_data: str) -> str:
        """Process and enhance the input with context."""
        
        # Extract any error messages or important data
        error_patterns = [
            r'Error: (.+)',
            r'Failed: (.+)',
            r'Exception: (.+)',
            r'Warning: (.+)'
        ]
        
        extracted_info = []
        for pattern in error_patterns:
            matches = re.findall(pattern, input_data, re.IGNORECASE)
            extracted_info.extend(matches)
        
        # Add context about current step
        context = f"""
Current Step: {self.current_step}/{self.max_analysis_steps}
Previous Actions: {len(self.analysis_steps)} analysis steps completed

Input Data:
{input_data}
"""
        
        if extracted_info:
            context += f"""
Extracted Key Information:
{chr(10).join(f"- {info}" for info in extracted_info)}
"""
        
        return context
    
    async def _generate_contextual_response(self, processed_input: str) -> str:
        """Generate a response with enhanced context awareness."""
        
        # Add context about current analysis state
        context_prompt = f"""
{processed_input}

Analysis Progress:
- Current step: {self.current_step}
- Previous hypotheses: {len(self.solution_attempts)} attempts
- Available APIs: {len(self.telemetry_apis)} telemetry + shell + submit

Based on the current state and input, provide your structured response following the format:
Analysis: <your systematic analysis>
Hypothesis: <your current hypothesis>
Action: <your specific action>
Rationale: <why this action addresses the hypothesis>
Expected_Outcome: <what you expect to happen>
"""
        
        # Get LLM response
        temp_history = self.history.copy()
        temp_history.append({"role": "user", "content": context_prompt})
        
        response = self.llm.run(temp_history)
        
        return response[0] if isinstance(response, list) else response
    
    def _filter_dict(self, dictionary: Dict, filter_func) -> Dict:
        """Filter dictionary based on a function."""
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis performed."""
        return {
            'total_steps': self.current_step,
            'analysis_steps': self.analysis_steps,
            'solution_attempts': self.solution_attempts,
            'problem_context': self.problem_context
        }
    
    def reset_analysis(self):
        """Reset the analysis state for a new problem."""
        self.analysis_steps = []
        self.solution_attempts = []
        self.current_step = 0
        self.history = []
        self.problem_context = {}


# Example usage and testing
if __name__ == "__main__":
    async def test_agent():
        """Test the custom agent with a sample problem."""
        
        agent = CustomAgent()
        orchestrator = Orchestrator()
        orchestrator.register_agent(agent, name="custom-aiops-agent")
        
        # Test with a sample problem
        try:
            pid = "misconfig_app_hotel_res-mitigation-1"
            problem_desc, instructions, apis = orchestrator.init_problem(pid)
            agent.init_context(problem_desc, instructions, apis)
            
            print(f"Initialized agent with problem: {pid}")
            print(f"Problem Description: {problem_desc[:200]}...")
            print(f"Available APIs: {len(apis)}")
            
            # Start the problem solving process
            await orchestrator.start_problem(max_steps=15)
            
            # Get analysis summary
            summary = agent.get_analysis_summary()
            print(f"\nAnalysis Summary:")
            print(f"Total Steps: {summary['total_steps']}")
            print(f"Analysis Steps: {len(summary['analysis_steps'])}")
            
        except Exception as e:
            print(f"Error during testing: {e}")
            print("This is expected if the full AIOpsLab environment is not set up.")
    
    # Run the test
    asyncio.run(test_agent())
