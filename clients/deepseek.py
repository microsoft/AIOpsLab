"""Naive DeepSeek-R1 client (with shell access) for AIOpsLab.

"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" arXiv preprint arXiv:2501.12948 (2025).

Paper: https://arxiv.org/abs/2501.12948
"""


import os
import asyncio

import wandb
from aiopslab.orchestrator import Orchestrator
from clients.utils.llm import DeepSeekClient
from clients.utils.templates import DOCS_SHELL_ONLY
from dotenv import load_dotenv

load_dotenv()

class DeepSeekAgent:
    def __init__(self):
        self.history = []
        self.llm = DeepSeekClient()

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
        self.history.append({"role": "assistant", "content": ""}) # Interleave the user/assistant messages in the message sequence.

    async def get_action(self, input) -> str:
        """Wrapper to interface the agent with OpsBench.

        Args:
            input (str): The input from the orchestrator/environment.

        Returns:
            str: The response from the agent.
        """
        self.history.append({"role": "user", "content": input})
        response = self.llm.run(self.history)
        self.history.append({"role": "assistant", "content": response[0]})
        return response[0]

    def _filter_dict(self, dictionary, filter_func):
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}


if __name__ == "__main__":
    # Load use_wandb from environment variable with a default of False
    use_wandb = os.getenv("USE_WANDB", "false").lower() == "true"
    
    if use_wandb:
        # Initialize wandb running
        wandb.init(project="AIOpsLab", entity="AIOpsLab")

    agent = DeepSeekAgent()

    orchestrator = Orchestrator()
    orchestrator.register_agent(agent, name="deepseek-r1")

    pid = "misconfig_app_hotel_res-mitigation-1"
    problem_desc, instructs, apis = orchestrator.init_problem(pid)
    agent.init_context(problem_desc, instructs, apis)
    asyncio.run(orchestrator.start_problem(max_steps=10))

    if use_wandb:
        # Finish the wandb run
        wandb.finish()
