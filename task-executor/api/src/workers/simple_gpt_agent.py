"""Simplified GPT Agent for task executor without external dependencies."""

import os
import openai
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class GPTAgent:
    """Simplified GPT agent that works with OpenRouter or OpenAI directly."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        base_url: Optional[str] = None
    ):
        """Initialize the GPT agent."""
        self.model = model
        self.temperature = temperature
        self.history = []

        # Setup OpenAI client
        if base_url:
            # OpenRouter or custom endpoint
            self.client = openai.OpenAI(
                api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
                base_url=base_url
            )
        else:
            # Direct OpenAI
            self.client = openai.OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY")
            )

    def init_context(self, problem_desc: str, instructions: str, apis: str):
        """Initialize the context for the agent."""
        # Build system message
        system_message = f"""You are an AIOps agent tasked with solving the following problem:

Problem Description:
{problem_desc}

Task Instructions:
{instructions}

Available APIs:
{apis}

Please analyze the problem, use the available APIs to gather information, and solve the issue."""

        self.history = [
            {"role": "system", "content": system_message}
        ]

    async def get_action(self, state: str) -> str:
        """Get the agent's action based on the current state."""
        # Add user message
        self.history.append({"role": "user", "content": state})

        try:
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                temperature=self.temperature,
                max_tokens=2000
            )

            # Extract response
            action = response.choices[0].message.content

            # Add to history
            self.history.append({"role": "assistant", "content": action})

            return action

        except Exception as e:
            error_msg = f"Error calling LLM API: {str(e)}"
            print(f"GPTAgent error: {error_msg}")
            return f"ERROR: {error_msg}"