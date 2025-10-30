"""Naive Qwen client (with shell access) for AIOpsLab.
"""

import os
import asyncio
import argparse
import json
from pathlib import Path

import wandb
from aiopslab.orchestrator import Orchestrator
from clients.utils.llm import QwenClient
from aiopslab.orchestrator.problems.registry import ProblemRegistry
from clients.utils.templates import DOCS_SHELL_ONLY
from dotenv import load_dotenv

load_dotenv()

class QwenAgent:
    def __init__(self):
        self.history = []
        self.llm = QwenClient()

    def init_context(self, problem_desc: str, instructions: str, apis: str):
        """Initialize the context for the agent."""

        self.shell_api = self._filter_dict(
            apis, lambda k, _: "exec_shell" in k)
        self.submit_api = self._filter_dict(apis, lambda k, _: "submit" in k)

        def stringify_apis(apis): return "\n\n".join(
            [f"{k}\n{v}" for k, v in apis.items()]
        )

        self.system_message = DOCS_SHELL_ONLY.format(
            prob_desc=problem_desc,
            shell_api=stringify_apis(self.shell_api),
            submit_api=stringify_apis(self.submit_api),
        )

        self.task_message = instructions

        self.history.append({"role": "system", "content": self.system_message})
        self.history.append({"role": "user", "content": self.task_message})

    async def get_action(self, input) -> str:
        """Wrapper to interface the agent with OpsBench.

        Args:
            input (str): The input from the orchestrator/environment.

        Returns:
            str: The response from the agent.
        """
        self.history.append({"role": "user", "content": input})
        response = self.llm.run(self.history)
        print(f"===== Agent (Qwen - {self.llm.model}) ====\n{response[0]}")
        self.history.append({"role": "assistant", "content": response[0]})
        return response[0]

    def _filter_dict(self, dictionary, filter_func):
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}

def get_completed_problems(results_dir: Path, agent_name: str, model: str) -> set:
    """Get set of completed problem IDs from existing result files."""
    completed = set()

    # Look in organized directory structure first
    organized_dir = results_dir / agent_name / model.replace("/", "_")
    if organized_dir.exists():
        for result_file in organized_dir.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    if 'problem_id' in data:
                        completed.add(data['problem_id'])
            except (json.JSONDecodeError, IOError):
                continue

    # Also check legacy flat structure
    for result_file in results_dir.glob("*.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                if ('problem_id' in data and
                    data.get('agent') == agent_name and
                    model.split('/')[-1] in str(result_file)):
                    completed.add(data['problem_id'])
        except (json.JSONDecodeError, IOError):
            continue

    return completed

def setup_results_directory(model: str, agent_name: str = "qwen") -> Path:
    """Setup organized results directory structure."""
    results_base = Path("aiopslab/data/results")

    # Create organized structure: results/{agent}/{model_safe}/
    model_safe = model.replace("/", "_")
    results_dir = results_base / agent_name / model_safe
    results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run qwen agent on AIOpsLab problems')
    parser.add_argument('--skip-completed', action='store_true',
                       help='Skip problems that have already been completed')
    parser.add_argument('--problem-ids', nargs='+',
                       help='Run only specific problem IDs')
    parser.add_argument('--max-steps', type=int, default=30,
                       help='Maximum steps per problem (default: 30)')
    parser.add_argument('--model', type=str,
                       default=os.getenv("QWEN_MODEL", "openai/gpt-4o-mini"),
                       help='qwen model to use')

    args = parser.parse_args()
    # Load use_wandb from environment variable with a default of False
    use_wandb = os.getenv("USE_WANDB", "false").lower() == "true"

    if use_wandb:
        # Initialize wandb running
        wandb.init(project="AIOpsLab", entity="AIOpsLab")

    model = args.model
    agent_name = "qwen"

    # Setup organized results directory
    results_dir = setup_results_directory(model, agent_name)
    print(f"Results will be saved to: {results_dir}")

    # Get all problems
    problems = ProblemRegistry().PROBLEM_REGISTRY

    # Filter problems if specific IDs requested
    if args.problem_ids:
        problems = {pid: problems[pid] for pid in args.problem_ids if pid in problems}
        if not problems:
            print("No valid problem IDs found")
            exit(1)

    # Skip completed problems if requested
    if args.skip_completed:
        completed_problems = get_completed_problems(
            Path("aiopslab/data/results"), agent_name, model
        )
        problems = {pid: prob for pid, prob in problems.items()
                   if pid not in completed_problems}

        print(f"Found {len(completed_problems)} completed problems")
        print(f"Running {len(problems)} remaining problems")

        if not problems:
            print("All problems have been completed!")
            exit(0)

    print(f"Running {len(problems)} problems with model: {model}")

    for pid in problems:
        print(f"\n=== Starting problem: {pid} ===")
        agent = QwenAgent()

        orchestrator = Orchestrator(results_dir=results_dir)
        orchestrator.register_agent(agent, name=agent_name)

        problem_desc, instructs, apis = orchestrator.init_problem(pid)
        agent.init_context(problem_desc, instructs, apis)
        asyncio.run(orchestrator.start_problem(max_steps=args.max_steps))

    if use_wandb:
        # Finish the wandb run
        wandb.finish()
