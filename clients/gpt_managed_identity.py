"""Naive GPT4 client (with shell access) for AIOpsLab. Uses Azure Managed Identity for authentication.

Achiam, Josh, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida et al. 
"Gpt-4 technical report." arXiv preprint arXiv:2303.08774 (2023).

Code: https://openai.com/index/gpt-4-research/
Paper: https://arxiv.org/abs/2303.08774
"""

import sys
import asyncio

from aiopslab.orchestrator import Orchestrator
from clients.utils.llm import GPT4Turbo
from clients.utils.templates import DOCS_SHELL_ONLY


class Agent:
    def __init__(self, azure_config_file: str):
        self.history = []
        self.llm = GPT4Turbo(auth_type="managed", azure_config_file=azure_config_file)

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
    if len(sys.argv) < 2:
        raise Exception(
            "Please provide a filename as argument. Usage: python gpt_managed_identity.py <azure_config_file>"
        )

    agent = Agent(azure_config_file=sys.argv[1])

    orchestrator = Orchestrator()
    orchestrator.register_agent(agent, name="gpt-w-shell")

    pid = "misconfig_app_hotel_res-mitigation-1"
    problem_desc, instructs, apis = orchestrator.init_problem(pid)
    agent.init_context(problem_desc, instructs, apis)
    asyncio.run(orchestrator.start_problem(max_steps=10))
