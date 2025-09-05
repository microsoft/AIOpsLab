# This is a naive implementation of Flash without tool and TSG.

import asyncio
import json
import os
import logging
import tiktoken
from typing import List, Dict, Tuple, Any
from pydantic import BaseModel
from clients.utils.llm import GPTClient
from aiopslab.orchestrator import Orchestrator
from aiopslab.orchestrator.problems.registry import ProblemRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def count_message_tokens(message, enc):
    # Each message format adds ~4 tokens of overhead
    tokens = 4  # <|start|>role/name + content + <|end|>
    tokens += len(enc.encode(message.get("content", "")))
    return tokens

def trim_history_to_token_limit(history, max_tokens=90000, model="gpt-4"):
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

class FlashAgent:
    def __init__(self):
        self.history = []
        self.llm = GPTClient()
        self.hindsight_builder = HindsightBuilder()

    def init_context(self, problem_desc: str, instructions: str, apis: dict):
        self.shell_api = self._filter_dict(apis, lambda k, _: "exec_shell" in k)
        self.submit_api = self._filter_dict(apis, lambda k, _: "submit" in k)
        self.telemetry_apis = self._filter_dict(
            apis, lambda k, _: "exec_shell" not in k and "submit" not in k
        )

        self.system_message = f"""
        Problem Description: {problem_desc}

        Available Telemetry APIs:
        {self._stringify_apis(self.telemetry_apis)}

        Shell API:
        {self._stringify_apis(self.shell_api)}

        Submit API:
        {self._stringify_apis(self.submit_api)}
        """

        self.task_message = instructions
        self.history.append({"role": "system", "content": self.system_message})
        self.history.append({"role": "user", "content": self.task_message})

    def _filter_dict(self, dictionary, filter_func):
        """Helper function to filter the API dictionary."""
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}

    def _stringify_apis(self, apis):
        return "\n\n".join([f"{k}\n{v}" for k, v in apis.items()])

    async def get_action(self, input_text: str) -> str:
        """Determine the next action based on the input, hindsight, and reasoning."""
        trimmed_for_hindsight = trim_history_to_token_limit(self.history, max_tokens=50000)
        hindsight = await self.diagnose_with_hindsight(input_text, trimmed_for_hindsight)
        if hindsight:
            hightsight = hindsight[:1000]


        combined_input = (
            f"{input_text}\n\nHindsight from Flash agent:\n{hindsight}"
            if hindsight
            else input_text
        )
        trimmed_history = trim_history_to_token_limit(self.history + [{"role": "user", "content": combined_input}])
        response = self.llm.run(trimmed_history)
        self.history = trimmed_history + [{"role": "assistant", "content": response[0]}]

        return response[0]

    async def diagnose_with_hindsight(self, input: str, history: dict):
        """Diagnose the incident and integrate hindsight from the environment status."""
        logger.info("Starting diagnosis with hindsight integration...")
        hindsight = self.hindsight_builder.develop_hindsight(input, history)
        if hindsight:
            logger.info(f"Generated Hindsight: {hindsight}")
        else:
            logger.info("No hindsight generated, continuing with normal execution.")


class HindsightBuilder:
    """Agent hindsight generator."""

    llm = GPTClient()

    def summarize_history(self, history: List[Dict]) -> str:
        summary = []
        for msg in history[-5:]:  # Keep only last 5 messages
            content = msg['content']
            summary.append(f"{msg['role']}: {content[:300]}")  # Truncate long messages
        return "\n".join(summary)

    def generate_prompt(self, input: str, history: List[Dict]) -> str:
        summarized_history = self.summarize_history(history)
        prompt = f"""
    You are a helpful assistant determining the next best action based on the current situation.

    Given the history of the previous actions:
    {summarized_history}

    And the environment output from last action:
    {input[:1000]}

    1. Should the next action be a submit operation? 
    2. If not, please suggest additional diagnostic steps.

    Thought: Identify whether submitting is the right next step.
    Solution: Provide reasoning and next steps.
    """
        return prompt


    def develop_hindsight(self, input: str, history: dict) -> str:
        """
        Develop hindsight based on the input and provide guidance for the next action.
        """
        prompt = self.generate_prompt(input, history)
        response = self.llm.run([{"role": "user", "content": prompt}])
        return response[0]


if __name__ == "__main__":
    problems = ProblemRegistry().PROBLEM_REGISTRY
    os.makedirs("results", exist_ok=True)

    for pid in problems:
        flash_agent = FlashAgent()
        orchestrator = Orchestrator()

        orchestrator.register_agent(flash_agent, name="flash")

        try:
            problem_desc, instructs, apis = orchestrator.init_problem(pid)
            flash_agent.init_context(problem_desc, instructs, apis)

            full_output = asyncio.run(orchestrator.start_problem(max_steps=30))
            results = full_output.get("results", {})
            
            filename = os.path.join("results", f"flash_{pid}.json")
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"Error while running problem {pid}: {e}")        
