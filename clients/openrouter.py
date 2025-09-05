"""OpenRouter client (with shell access) for AIOpsLab.

OpenRouter provides access to multiple AI models through a unified API.
More info: https://openrouter.ai/
"""

import os
import asyncio
import tiktoken
import wandb
from aiopslab.orchestrator import Orchestrator
from aiopslab.orchestrator.problems.registry import ProblemRegistry
from clients.utils.llm import OpenRouterClient
from clients.utils.templates import DOCS_SHELL_ONLY
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

def count_message_tokens(message, enc):
    # Each message format adds ~4 tokens of overhead
    tokens = 4  # <|start|>role/name + content + <|end|>
    tokens += len(enc.encode(message.get("content", "")))
    return tokens

def trim_history_to_token_limit(history, max_tokens=120000, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)

    trimmed = []
    total_tokens = 0

    # Always include the last message
    last_msg = history[-1]
    last_msg_tokens = count_message_tokens(last_msg, enc)

    if last_msg_tokens > max_tokens:
        # If even the last message is too big, truncate its content
        truncated_content = enc.decode(enc.encode(last_msg["content"])[:max_tokens - 4])
        return [{"role": last_msg["role"], "content": truncated_content}]

    trimmed.insert(0, last_msg)
    total_tokens += last_msg_tokens

    # Add earlier messages in reverse until limit is reached
    for message in reversed(history[:-1]):
        message_tokens = count_message_tokens(message, enc)
        if total_tokens + message_tokens > max_tokens:
            break
        trimmed.insert(0, message)
        total_tokens += message_tokens

    return trimmed

class OpenRouterAgent:
    def __init__(self, model="anthropic/claude-3.5-sonnet"):
        self.history = []
        self.llm = OpenRouterClient(model=model)
        self.model = model

    def test(self):
        return self.llm.run([{"role": "system", "content": "hello"}])

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
        try:
            trimmed_history = trim_history_to_token_limit(self.history)
            response = self.llm.run(trimmed_history)
            print(f"===== Agent (OpenRouter - {self.model}) ====\n{response[0]}")
            self.history.append({"role": "assistant", "content": response[0]})
            return response[0]
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            # Return a fallback response
            fallback_response = f"Error occurred while calling OpenRouter API: {e}"
            self.history.append({"role": "assistant", "content": fallback_response})
            return fallback_response

    def _filter_dict(self, dictionary, filter_func):
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}


if __name__ == "__main__":
    # Load use_wandb from environment variable with a default of False
    use_wandb = os.getenv("USE_WANDB", "false").lower() == "true"

    if use_wandb:
        # Initialize wandb running
        wandb.init(project="AIOpsLab", entity="AIOpsLab")

    # You can specify different models supported by OpenRouter
    # Popular models:
    # - "anthropic/claude-3.5-sonnet"
    # - "openai/gpt-4-turbo"
    # - "meta-llama/llama-3.1-8b-instruct"
    # - "google/gemini-pro"
    # - "mistralai/mixtral-8x7b-instruct"
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    problems = ProblemRegistry().PROBLEM_REGISTRY
    for pid in problems:
        agent = OpenRouterAgent(model=model)

        orchestrator = Orchestrator()
        orchestrator.register_agent(agent, name="openrouter")

        problem_desc, instructs, apis = orchestrator.init_problem(pid)
        agent.init_context(problem_desc, instructs, apis)
        asyncio.run(orchestrator.start_problem(max_steps=30))

    if use_wandb:
        # Finish the wandb run
        wandb.finish()