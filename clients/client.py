"""Main runner for AIOpsLab agents."""

import os
import asyncio
import argparse

import wandb
from aiopslab.orchestrator import Orchestrator
from aiopslab.orchestrator.problems.registry import ProblemRegistry
from clients.registry import AgentRegistry

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run an AIOpsLab agent")
    parser.add_argument("--agent", type=str, required=True, 
                        choices=["gpt", "qwen", "deepseek", "vllm"],
                        help="The agent implementation to use")
    parser.add_argument("--problem-id", type=str, 
                        default="misconfig_app_hotel_res-mitigation-1",
                        help="The problem ID to solve")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Maximum number of interaction steps")
    
    # Add vLLM-specific arguments
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct",
                        help="Model name for vLLM")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty for vLLM")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling (vLLM only)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p for nucleus sampling (vLLM only)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum tokens to generate (vLLM only)")
    return parser.parse_args()

async def run_agent(agent_name, problem_id, max_steps, model, temperature, top_p, max_tokens, repetition_penalty, use_wandb=False):
    """Run an agent on a problem."""
    if use_wandb:
        # Initialize wandb running
        wandb.init(project="AIOpsLab", entity="AIOpsLab")

    # Get the agent class from registry and instantiate
    agent_registry = AgentRegistry()
    agent_cls = agent_registry.get_agent(agent_name)
    if agent_cls is None:
        raise ValueError(f"Unknown agent: {agent_name}")
    
    # If the agent is vLLM, pass the specific parameters
    if agent_name == "vllm":
        agent = agent_cls(
            model=model,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    else:
         # For other agents, just instantiate without parameters
        agent = agent_cls()


    orchestrator = Orchestrator()
    orchestrator.register_agent(agent, name=f"{agent_name}-agent")

    try:
        print(f"{'*'*30}")
        print(f"Starting problem {problem_id} with agent {agent_name}")
        print(f"{'*'*30}")
        
        problem_desc, instructs, apis = orchestrator.init_problem(problem_id)
        agent.init_context(problem_desc, instructs, apis)
        await orchestrator.start_problem(max_steps=max_steps)
        
        print(f"{'*'*30}")
        print(f"Successfully completed problem {problem_id}")
        print(f"{'*'*30}")
    except Exception as e:
        print(f"Failed to process problem {problem_id}. Error: {e}")
    finally:
        if use_wandb:
            # Finish the wandb run
            wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    
    # Override with environment variable if set
    use_wandb = os.getenv("USE_WANDB", "false").lower() == "true"
    
    asyncio.run(run_agent(
        agent_name=args.agent,
        problem_id=args.problem_id,
        max_steps=args.max_steps,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        use_wandb=use_wandb
    ))