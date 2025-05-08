"""Naive ReAct client for AIOpsLab.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). 
React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.

Code: https://github.com/ysymyth/ReAct
Paper: https://arxiv.org/abs/2210.03629
"""

import asyncio
import json
import os
from aiopslab.orchestrator import Orchestrator
from aiopslab.orchestrator.problems.registry import ProblemRegistry
from clients.utils.llm import GPTClient
from clients.utils.templates import DOCS

RESP_INSTR = """DO NOT REPEAT ACTIONS! Respond with:
Thought: <your thought on the previous output>
Action: <your action towards mitigating>
"""


class Agent:
    def __init__(self):
        self.history = []
        self.llm = GPTClient()

    def init_context(self, problem_desc: str, instructions: str, apis: str):
        """Initialize the context for the agent."""

        self.shell_api = self._filter_dict(apis, lambda k, _: "exec_shell" in k)
        self.submit_api = self._filter_dict(apis, lambda k, _: "submit" in k)
        self.telemetry_apis = self._filter_dict(
            apis, lambda k, _: "exec_shell" not in k and "submit" not in k
        )

        stringify_apis = lambda apis: "\n\n".join(
            [f"{k}\n{v}" for k, v in apis.items()]
        )

        self.system_message = DOCS.format(
            prob_desc=problem_desc,
            telemetry_apis=stringify_apis(self.telemetry_apis),
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
        self.history.append({"role": "user", "content": self._add_instr(input)})
        response = self.llm.run(self.history)
        self.history.append({"role": "assistant", "content": response[0]})
        return response[0]

    def _filter_dict(self, dictionary, filter_func):
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}

    def _add_instr(self, input):
        return input + "\n\n" + RESP_INSTR


if __name__ == "__main__":
    problems = ProblemRegistry().PROBLEM_REGISTRY
    os.makedirs("results", exist_ok=True)

    for pid in problems:
        agent = Agent()
        orchestrator = Orchestrator()
        orchestrator.register_agent(agent, name="react")

        try:
            problem_desc, instructs, apis = orchestrator.init_problem(pid)
            agent.init_context(problem_desc, instructs, apis)

            full_output = asyncio.run(orchestrator.start_problem(max_steps=30))
            results = full_output.get("react_results", {})

            filename = os.path.join("results", f"{pid}.json")
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"Error while running problem {pid}: {e}")