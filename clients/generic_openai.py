"""Generic OpenAI-compatible chat client (with shell access) for AIOpsLab.

This agent works with any provider that implements the OpenAI Chat Completions
API endpoint (/v1/chat/completions), such as Poe
(https://creator.poe.com/docs/external-applications/openai-compatible-api),
standard OpenAI deployments, vLLM, LocalAI, or other compatible services.

Configure the endpoint and model via environment variables or constructor arguments:
    OPENAI_COMPATIBLE_API_KEY  — API key for the target endpoint
    OPENAI_COMPATIBLE_BASE_URL — Base URL of the target endpoint (e.g. https://api.poe.com/llm/v1)
    OPENAI_COMPATIBLE_MODEL    — Model name to use (e.g. MiniMax-Text-01)
"""

import os
import asyncio
import wandb
from aiopslab.orchestrator import Orchestrator
from aiopslab.orchestrator.problems.registry import ProblemRegistry
from clients.utils.llm import GenericOpenAIClient
from clients.utils.templates import DOCS_SHELL_ONLY
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


class GenericOpenAIAgent:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.history = []
        self.llm = GenericOpenAIClient(
            base_url=base_url,
            model=model,
            api_key=api_key,
        )

    def init_context(self, problem_desc: str, instructions: str, apis: str):
        """Initialize the context for the agent."""

        self.shell_api = self._filter_dict(apis, lambda k, _: "exec_shell" in k)
        self.submit_api = self._filter_dict(apis, lambda k, _: "submit" in k)
        stringify_apis = lambda apis: "\n\n".join(
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
        """Wrapper to interface the agent with AIOpsLab.

        Args:
            input (str): The input from the orchestrator/environment.

        Returns:
            str: The response from the agent.
        """
        self.history.append({"role": "user", "content": input})
        response = self.llm.run(self.history)
        model_name = self.llm.model
        print(f"===== Agent (GenericOpenAI - {model_name}) ====\n{response[0]}")
        self.history.append({"role": "assistant", "content": response[0]})
        return response[0]

    def _filter_dict(self, dictionary, filter_func):
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}


if __name__ == "__main__":
    # Load use_wandb from environment variable with a default of False
    use_wandb = os.getenv("USE_WANDB", "false").lower() == "true"

    if use_wandb:
        wandb.init(project="AIOpsLab", entity="AIOpsLab")

    problems = ProblemRegistry().PROBLEM_REGISTRY
    for pid in problems:
        agent = GenericOpenAIAgent()

        orchestrator = Orchestrator()
        orchestrator.register_agent(agent, name="generic-openai")

        problem_desc, instructs, apis = orchestrator.init_problem(pid)
        agent.init_context(problem_desc, instructs, apis)
        asyncio.run(orchestrator.start_problem(max_steps=30))

    if use_wandb:
        wandb.finish()
