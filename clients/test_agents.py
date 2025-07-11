#!/usr/bin/env python3
"""Test script for custom AIOps agents.

This script helps you test your custom agents with different problems
and configurations in the AIOpsLab environment.
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from aiopslab.orchestrator import Orchestrator
from clients.custom_agent import CustomAgent
from clients.simple_agent import SimpleAgent


class AgentTester:
    """Test runner for AIOps agents."""
    
    def __init__(self):
        self.orchestrator = Orchestrator()
        self.agents = {
            'custom': CustomAgent,
            'simple': SimpleAgent
        }
    
    async def test_agent(self, agent_type: str, problem_id: str, max_steps: int = 10):
        """Test a specific agent with a problem."""
        
        if agent_type not in self.agents:
            print(f"Unknown agent type: {agent_type}")
            print(f"Available agents: {list(self.agents.keys())}")
            return
        
        print(f"Testing {agent_type} agent with problem: {problem_id}")
        print("-" * 50)
        
        # Create agent instance
        agent = self.agents[agent_type]()
        
        # Register with orchestrator
        self.orchestrator.register_agent(agent, name=f"{agent_type}-test-agent")
        
        try:
            # Initialize problem
            problem_desc, instructions, apis = self.orchestrator.init_problem(problem_id)
            
            print(f"Problem Description: {problem_desc[:200]}...")
            print(f"Available APIs: {len(apis)}")
            print(f"Max Steps: {max_steps}")
            print("-" * 50)
            
            # Initialize agent context
            agent.init_context(problem_desc, instructions, apis)
            
            # Start problem solving
            await self.orchestrator.start_problem(max_steps=max_steps)
            
            # Print summary if available
            if hasattr(agent, 'get_analysis_summary'):
                summary = agent.get_analysis_summary()
                print(f"\nAnalysis Summary:")
                print(f"Total Steps: {summary['total_steps']}")
                print(f"Analysis Steps: {len(summary['analysis_steps'])}")
            
            print("\nAgent test completed successfully!")
            
        except Exception as e:
            print(f"Error during agent test: {e}")
            import traceback
            traceback.print_exc()
    
    def list_problems(self):
        """List available problems."""
        try:
            problems = self.orchestrator.probs.list_problems()
            print("Available problems:")
            for problem in problems:
                print(f"  - {problem}")
        except Exception as e:
            print(f"Error listing problems: {e}")
    
    def list_agents(self):
        """List available agents."""
        print("Available agents:")
        for agent_name, agent_class in self.agents.items():
            print(f"  - {agent_name}: {agent_class.__doc__ or 'No description'}")


async def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Test AIOps agents')
    parser.add_argument('--agent', '-a', choices=['custom', 'simple'], 
                       default='simple', help='Agent type to test')
    parser.add_argument('--problem', '-p', 
                       default='misconfig_app_hotel_res-mitigation-1',
                       help='Problem ID to test')
    parser.add_argument('--steps', '-s', type=int, default=10,
                       help='Maximum number of steps')
    parser.add_argument('--list-problems', action='store_true',
                       help='List available problems')
    parser.add_argument('--list-agents', action='store_true',
                       help='List available agents')
    
    args = parser.parse_args()
    
    tester = AgentTester()
    
    if args.list_problems:
        tester.list_problems()
        return
    
    if args.list_agents:
        tester.list_agents()
        return
    
    # Test the agent
    await tester.test_agent(args.agent, args.problem, args.steps)


if __name__ == "__main__":
    # Example usage without arguments
    if len(sys.argv) == 1:
        print("AIOps Agent Tester")
        print("=" * 30)
        print("\nUsage examples:")
        print("  python test_agents.py --agent simple --problem misconfig_app_hotel_res-mitigation-1")
        print("  python test_agents.py --agent custom --steps 15")
        print("  python test_agents.py --list-problems")
        print("  python test_agents.py --list-agents")
        print("\nRunning with default settings (simple agent)...")
        print("-" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
