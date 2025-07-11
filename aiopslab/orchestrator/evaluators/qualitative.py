# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper functions for qualitative evaluation of solutions."""

import os
import re
import ast
from openai import OpenAI

from aiopslab.session import SessionItem
from aiopslab.utils.cache import LLMCache
from aiopslab.orchestrator.evaluators.prompts import SCORER_PROMPTS


class LLMJudge:
    """A LLM as a judge that evaluates the quality of a solution."""

    def __init__(self, trace: list[SessionItem]):
        self.trace = trace
        self.llm = GPT4Turbo()
        self.prompt = None

        self._format_trace()

    def reasoning_score(self) -> bool:
        """Generate a 1-10 score based on the agent's response to a task"""
        self.prompt = SCORER_PROMPTS
        self.prompt["user"] = self.prompt["user"].format(trace=self.trace)
        judgement = self.llm.inference(self._get_payload())[0]
        score = self._parse_score(judgement)
        return score, judgement

    # helper functions

    def _get_payload(self):
        """Prepare the payload for the LLM."""
        payload = []
        for role, content in self.prompt.items():
            payload.append({"role": role, "content": content})
        return payload

    def _format_trace(self):
        """Format the trace for the LLM."""
        item2str = lambda item: f"###{item.role}:\n{item.content}\n\n"
        self.trace = "".join([item2str(item) for item in self.trace])

    def _parse_score(self, judgement: str) -> int:
        """Parse the score from the judgement."""
        one_score_pattern = re.compile(r"\[\[(\d+\.?\d*)\]\]")
        one_score_pattern_backup = re.compile(r"\[(\d+\.?\d*)\]")

        match = re.search(one_score_pattern, judgement)
        if not match:
            match = re.search(one_score_pattern_backup, judgement)

        if match:
            score = ast.literal_eval(match.groups()[0])
        else:
            score = -1

        return score


class GPT4Turbo:
    """An abstraction of the GPT-4 Turbo model (default judge)."""

    def __init__(self):
        self.cache = LLMCache()

    def inference(self, payload: list[dict[str, str]]) -> list[str]:
        if self.cache is not None:
            cache_result = self.cache.get_from_cache(payload)
            if cache_result is not None:
                return cache_result

        # Check if using Azure OpenAI
        if os.getenv("OPENAI_API_TYPE") == "azure":
            from openai import AzureOpenAI
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not azure_endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required for Azure OpenAI")
            client = AzureOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                api_version=os.getenv("OPENAI_API_VERSION", "2023-12-01-preview"),
                azure_endpoint=azure_endpoint
            )
            model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        else:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model_name = "gpt-4-turbo-2024-04-09"
            
        try:
            response = client.chat.completions.create(
                messages=payload,  # type: ignore
                model=model_name,
                max_tokens=1024,
                temperature=0.0,
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                n=1,
                timeout=60,
                stop=[],
            )
        except Exception as e:
            print(f"Exception: {repr(e)}")
            raise e

        response = [c.message.content for c in response.choices]  # type: ignore

        if self.cache is not None:
            self.cache.add_to_cache(payload, response)
            self.cache.save_cache()

        return response
