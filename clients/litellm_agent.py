"""LiteLLM-backed agent (with shell access) for AIOpsLab.

LiteLLM (https://docs.litellm.ai/) gives one interface to 100+ model
providers, so a problem can be evaluated against models the registry has no
dedicated client for, including Anthropic and Gemini.

Two ways to use it, from the same model name:

1. Direct, with no extra infrastructure. LiteLLM routes each model to its own
   provider using that provider's key:

       LITELLM_MODEL=anthropic/claude-opus-4-7
       ANTHROPIC_API_KEY=...

2. Through a self-hosted LiteLLM gateway, which keeps every provider key
   server-side and adds centralized cost tracking, budgets, rate limiting and
   fallbacks across a long benchmark run:

       LITELLM_MODEL=claude-opus-4-7
       LITELLM_API_BASE=http://localhost:4000
       LITELLM_API_KEY=...

Configure via environment variables or constructor arguments:
    LITELLM_MODEL    — model to route (default: gpt-4o)
    LITELLM_API_BASE — optional gateway URL; unset means call the provider directly
    LITELLM_API_KEY  — gateway key when LITELLM_API_BASE is set, otherwise unset
                       so LiteLLM reads the provider's own variable
"""

import os
import asyncio
import wandb
from aiopslab.orchestrator import Orchestrator
from aiopslab.orchestrator.problems.registry import ProblemRegistry
from clients.utils.llm import LiteLLMClient
from clients.utils.templates import DOCS_SHELL_ONLY
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


class LiteLLMAgent:
    def __init__(
        self,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        self.history = []
        self.llm = LiteLLMClient(
            model=model,
            api_base=api_base,
            api_key=api_key,
        )

    def init_context(self, problem_desc: str, instructions: str, apis: dict[str, str]):
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
        print(f"===== Agent (LiteLLM - {model_name}) ====\n{response[0]}")
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
        agent = LiteLLMAgent()

        orchestrator = Orchestrator()
        orchestrator.register_agent(agent, name="litellm")

        problem_desc, instructs, apis = orchestrator.init_problem(pid)
        agent.init_context(problem_desc, instructs, apis)
        asyncio.run(orchestrator.start_problem(max_steps=30))

    if use_wandb:
        wandb.finish()
