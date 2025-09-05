# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


"""AIOpsLab CLI client."""

import asyncio
import json
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit.completion import WordCompleter

from aiopslab.onboarding_evaluator import Evaluator


WELCOME = """
# AIOpsLab Onboarding Assessment
"""

TASK_MESSAGE = """\n\n\n\n
There's a problem in the kubernetes cluster in the test-hotel-reservation namespace.

The issue is that there are unmet PersistentVolumeClaims (PVCs) because of unbound persistent volumes.

You need to fix the issue and get all the pods into a ready state.

You have access to a shell, take whatever action you deem necessary to resolve the issue.

Once you believe the incident is resolved, run the `submit` command. If your solution is incorrect, it will tell you.

You can use any resources you want to complete the assessment except for another person. However, please run all shell commands inside of the interface.

Your results will be saved in a file called yourFistName_results.json, please email it to jclark58@illinois.edu

If you encounter a bug, send it to jclark58@illinois.edu
"""


class HumanAgent:
    def __init__(self, orchestrator):
        self.session = PromptSession()
        self.console = Console(force_terminal=True, color_system="auto")
        self.orchestrator = orchestrator
        self.pids = self.orchestrator.probs.get_problem_ids()
        self.completer = WordCompleter(self.pids, ignore_case=True, match_middle=True)

    def display_welcome_message(self):
        self.console.print(Markdown(WELCOME), justify="center")
        self.console.print()

    def display_context(self, problem_desc, apis):
        self.shell_api = self._filter_dict(apis, lambda k, _: "exec_shell" in k)
        self.submit_api = self._filter_dict(apis, lambda k, _: "submit" in k)
        self.telemetry_apis = self._filter_dict(
            apis, lambda k, _: "exec_shell" not in k and "submit" not in k
        )

        stringify_apis = lambda apis: "\n\n".join(
            [f"{k}\n{v}" for k, v in apis.items()]
        )

        self.task_message = TASK_MESSAGE.format(
            prob_desc=problem_desc,
            telemetry_apis=stringify_apis(self.telemetry_apis),
            shell_api=stringify_apis(self.shell_api),
            submit_api=stringify_apis(self.submit_api),
        )

        self.console.print(Markdown(self.task_message))

    def display_env_message(self, env_input):
        self.console.print(Panel(env_input, title="Environment", style="white on blue"))
        self.console.print()

    async def set_problem(self):
        self.init_problem("redeploy_without_PV-mitigation-1")

    async def get_action(self, env_input):
        user_input = await self.get_user_input()
        template = "Action:```\n{}\n```"
        return template.format(user_input)

    def init_problem(self, problem_id="misconfig-mitigation-1"):
        problem_desc, _, apis = self.orchestrator.init_problem(problem_id)
        self.display_context(problem_desc, apis)

    async def get_user_input(self, completer=None):
        loop = asyncio.get_running_loop()
        style = Style.from_dict({"prompt": "ansigreen bold"})
        prompt_text = [("class:prompt", "shell> ")]

        with patch_stdout():
            try:
                input = await loop.run_in_executor(
                    None,
                    lambda: self.session.prompt(
                        prompt_text, style=style, completer=completer
                    ),
                )

                if input.lower() == "exit":
                    raise SystemExit

                return input
            except (SystemExit, KeyboardInterrupt, EOFError):
                raise SystemExit from None

    def _filter_dict(self, dictionary, filter_func):
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}


async def main():
    orchestrator = Evaluator()
    agent = HumanAgent(orchestrator)
    orchestrator.register_agent(agent, name="human")

    first_name = input("What is your first name?: ")

    agent.display_welcome_message()
    await agent.set_problem()

    results = await orchestrator.start_problem()
    
    session_data = orchestrator.session.to_dict()
    
    with open(f"{first_name}_results.json", "w") as f:
        json.dump(session_data, f, indent=2)
    
    print(f"Results saved to {first_name}_results.json")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
