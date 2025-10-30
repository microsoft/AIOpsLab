# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A common abstraction for a cached LLM inference setup. Currently supports OpenAI's gpt-4-turbo and other models."""


import os
import json
import yaml
from groq import Groq
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass

from groq import Groq
from openai import OpenAI, AzureOpenAI
from azure.identity import get_bearer_token_provider, AzureCliCredential, ManagedIdentityCredential

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
"""An common abstraction for a cached LLM inference setup. Currently supports OpenAI's gpt-4-turbo and other models."""


CACHE_DIR = Path("./cache_dir")
CACHE_PATH = CACHE_DIR / "cache.json"
GPT_MODEL = "gpt-4o"


@dataclass
class AzureConfig:
    azure_endpoint: str
    api_version: str


class Cache:
    """A simple cache implementation to store the results of the LLM inference."""

    def __init__(self) -> None:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH) as f:
                self.cache_dict = json.load(f)
        else:
            os.makedirs(CACHE_DIR, exist_ok=True)
            self.cache_dict = {}

    @staticmethod
    def process_payload(payload):
        if isinstance(payload, (list, dict)):
            return json.dumps(payload)
        return payload

    def get_from_cache(self, payload):
        payload_cache = self.process_payload(payload)
        if payload_cache in self.cache_dict:
            return self.cache_dict[payload_cache]
        return None

    def add_to_cache(self, payload, output):
        payload_cache = self.process_payload(payload)
        self.cache_dict[payload_cache] = output

    def save_cache(self):
        with open(CACHE_PATH, "w") as f:
            json.dump(self.cache_dict, f, indent=4)


class GPTClient:
    """Abstraction for OpenAI's GPT series model."""

    def __init__(self):
        self.cache = Cache()

    def inference(self, payload: list[dict[str, str]]) -> list[str]:
        if self.cache is not None:
            cache_result = self.cache.get_from_cache(payload)
            if cache_result is not None:
                return cache_result

        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("OPENROUTER_MODEL", "gpt-4-turbo-2024-04-09")

        client = OpenAI(api_key=api_key, base_url=base_url)

        print("===== prompt =====")
        print(payload)
        try:
            response = client.chat.completions.create(
                messages=payload,  # type: ignore
                model=model,
                max_tokens=1024,
                temperature=0.5,
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

        return [c.message.content for c in response.choices]  # type: ignore

    def run(self, payload: list[dict[str, str]]) -> list[str]:
        response = self.inference(payload)
        if self.cache is not None:
            self.cache.add_to_cache(payload, response)
            self.cache.save_cache()
        return response


class DeepSeekClient:
    """Abstraction for DeepSeek model."""

    def __init__(self):
        self.cache = Cache()

    def inference(self, payload: list[dict[str, str]]) -> list[str]:
        if self.cache is not None:
            cache_result = self.cache.get_from_cache(payload)
            if cache_result is not None:
                return cache_result

        client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                        base_url="https://api.deepseek.com")

        print("===== prompt =====")
        print(payload)
        try:
            response = client.chat.completions.create(
                messages=payload,  # type: ignore
                model="deepseek-reasoner",
                max_tokens=1024,
                stop=[],
            )

        except Exception as e:
            print(f"Exception: {repr(e)}")
            raise e

        return [c.message.content for c in response.choices]  # type: ignore

    def run(self, payload: list[dict[str, str]]) -> list[str]:
        response = self.inference(payload)
        if self.cache is not None:
            self.cache.add_to_cache(payload, response)
            self.cache.save_cache()
        return response


class QwenClient:
    """Abstraction for Qwen's model. Some Qwen models only support streaming output."""

    def __init__(self,
                 model="qwen-plus",
                 max_tokens=1024,
                 is_stream=True):
        self.cache = Cache()
        self.model = os.getenv("QWEN_MODEL", model)
        self.max_tokens = os.getenv("QWEN_MAX_TOKEN", max_tokens)
        self.is_stream = os.getenv("QWEN_IS_STREAM", is_stream)

    def inference(self, payload: list[dict[str, str]]) -> list[str]:
        if self.cache is not None:
            cache_result = self.cache.get_from_cache(payload)
            if cache_result is not None:
                return cache_result
        client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        print("===== prompt =====")
        print(payload)
        try:
            response = client.chat.completions.create(
                messages=payload,  # type: ignore
                model=self.model,
                max_tokens=self.max_tokens,
                n=1,  # The response count to generate
                timeout=60,
                stop=[],
                stream=self.is_stream
            )
        except Exception as e:
            print(f"Exception: {repr(e)}")
            raise e

        reasoning_content = ""
        answer_content = ""
        is_answering = False

        for chunk in response:
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                    reasoning_content += delta.reasoning_content
                else:
                    if delta.content != "" and is_answering is False:
                        is_answering = True
                    answer_content += delta.content

        return [answer_content]

    def run(self, payload: list[dict[str, str]]) -> list[str]:
        response = self.inference(payload)
        if self.cache is not None:
            self.cache.add_to_cache(payload, response)
            self.cache.save_cache()
        return response


class vLLMClient:
    """Abstraction for local LLM models."""

    def __init__(self,
                 model="Qwen/Qwen2.5-Coder-3B-Instruct",
                 repetition_penalty=1.0,
                 temperature=1.0,
                 top_p=0.95,
                 max_tokens=1024):
        self.cache = Cache()
        self.model = model
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def inference(self, payload: list[dict[str, str]]) -> list[str]:
        if self.cache is not None:
            cache_result = self.cache.get_from_cache(payload)
            if cache_result is not None:
                return cache_result

        client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
        print("===== prompt =====")
        print(payload)
        try:
            response = client.chat.completions.create(
                messages=payload,  # type: ignore
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                n=1,
                timeout=60,
                stop=[],
            )
        except Exception as e:
            print(f"Exception: {repr(e)}")
            raise e

        return [c.message.content for c in response.choices]  # type: ignore

    def run(self, payload: list[dict[str, str]]) -> list[str]:
        response = self.inference(payload)
        if self.cache is not None:
            self.cache.add_to_cache(payload, response)
            self.cache.save_cache()
        return response


class OpenRouterClient:
    """Abstraction for OpenRouter API with support for multiple models."""

    def __init__(self, model="anthropic/claude-3.5-sonnet"):
        self.cache = Cache()
        self.model = model

    def inference(self, payload: list[dict[str, str]]) -> list[str]:
        if self.cache is not None:
            cache_result = self.cache.get_from_cache(payload)
            if cache_result is not None:
                return cache_result

        client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        print("===== prompt =====")
        print(payload)
        try:
            response = client.chat.completions.create(
                messages=payload,  # type: ignore
                model=self.model,
                max_tokens=1024,
                temperature=0.5,
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

        return [c.message.content for c in response.choices]  # type: ignore

    def run(self, payload: list[dict[str, str]]) -> list[str]:
        response = self.inference(payload)
        if self.cache is not None:
            self.cache.add_to_cache(payload, response)
            self.cache.save_cache()
        return response


class LLaMAClient:
    """Abstraction for Meta's LLaMA-3 model."""

    def __init__(self):
        self.cache = Cache()

    def inference(self, payload: list[dict[str, str]]) -> list[str]:
        if self.cache is not None:
            cache_result = self.cache.get_from_cache(payload)
            if cache_result is not None:
                return cache_result

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("===== prompt =====")
        print(payload)
        try:
            response = client.chat.completions.create(
                messages=payload,
                model="llama-3.1-8b-instant",
                max_tokens=1024,
                temperature=0.5,
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

        return [c.message.content for c in response.choices]  # type: ignore

    def run(self, payload: list[dict[str, str]]) -> list[str]:
        response = self.inference(payload)
        if self.cache is not None:
            self.cache.add_to_cache(payload, response)
            self.cache.save_cache()
        return response
