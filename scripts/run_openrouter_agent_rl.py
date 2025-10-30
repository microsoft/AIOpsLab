#!/usr/bin/env python3
"""
Run a real LLM agent (via OpenRouter) inside the RL environment and report rewards.

Requirements:
  - Environment variable OPENROUTER_API_KEY must be set (sk-or-...)

Usage:
  OPENROUTER_API_KEY=sk-or-... python3 scripts/run_openrouter_agent_rl.py \
    --problem-id container_kill-analysis-1 \
    --model openai/gpt-4o-mini \
    --max-steps 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from aiopslab.orchestrator.orchestrator import Orchestrator
from aiopslab.orchestrator.rl_env import ProblemRLEnvironment, RewardConfig


def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 5000,
) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected OpenRouter response: {json.dumps(data)[:800]}")


def build_system_prompt() -> str:
    return (
        "You are an expert SRE operating a Kubernetes cluster. You have access to these APIs:\n"
        "- exec_shell(command: str) -> Execute a shell command on the control plane.\n"
        "- submit(solution: dict) -> Submit your final diagnosis.\n\n"
        "Return exactly one API call per step, wrapped in a Python markdown code block. Examples:\n\n"
        "```python\nexec_shell(\"kubectl get pods -n <namespace>\")\n```\n\n"
        "```python\nsubmit({\"system_level\": \"Application\", \"fault_type\": \"Dependency Problem\"})\n```\n"
    )


def build_user_prompt(state: str, actions_left: int, step: int) -> str:
    return (
        f"Step {step}, actions left: {actions_left}.\n\n"
        f"Observation:\n{state}\n\n"
        "Choose the next best action. Return a single API call in a Python code block."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an OpenRouter LLM agent in the RL environment")
    parser.add_argument("--problem-id", default="container_kill-analysis-1", help="Problem ID to load")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="OpenRouter model to use")
    parser.add_argument("--max-steps", type=int, default=12, help="Maximum episode steps")
    parser.add_argument("--temperature", type=float, default=0.3, help="LLM temperature")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        return 2

    print("Initializing orchestrator and RL environment...")
    orchestrator = Orchestrator(results_dir=Path(__file__).parent.parent / "data" / "results" / "openrouter")
    orchestrator.agent_name = "openrouter-rl-agent"

    reward_config = RewardConfig(command_match_multiplier=0.1)
    env = ProblemRLEnvironment(
        orchestrator=orchestrator,
        max_steps=args.max_steps,
        reward_config=reward_config,
        ground_truth_dir=Path(__file__).parent.parent / "ground_truth",
    )

    print(f"Resetting environment for problem: {args.problem_id}")
    observation, info = env.reset(problem_id=args.problem_id)
    print("Environment ready. Beginning episode...\n")

    system_prompt = build_system_prompt()
    total_reward = 0.0
    step = 1
    done = False

    while not done and step <= args.max_steps:
        state = observation.get("state", "")
        actions_left = observation.get("actions_left", 0)

        user_prompt = build_user_prompt(state, actions_left, step)

        try:
            reply = call_openrouter(
                api_key=api_key,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=args.temperature,
            )
        except Exception as e:
            print(f"OpenRouter error: {e}")
            return 3

        print(f"\n--- Agent reply (truncated) ---\n{reply[:800]}\n-------------------------------\n")

        observation, reward, done, info = env.step(reply)
        total_reward += reward

        print(f"Step {step} reward: {reward:.3f} | cumulative: {total_reward:.3f}")
        if info.get("terminated"):
            print("Episode terminated by submission.")
        if info.get("truncated"):
            print("Episode truncated (max steps).")

        step += 1

    print("\nEpisode finished.")
    print(f"Total steps: {step-1}")
    print(f"Total reward: {total_reward:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


