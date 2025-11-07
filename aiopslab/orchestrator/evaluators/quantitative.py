# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper functions for quantiative evaluation of solutions."""

import tiktoken

from aiopslab.session import SessionItem

# Constants
token_model = "gpt-3.5-turbo"
tokenizer = tiktoken.encoding_for_model(token_model)


def num_steps_taken(trace: list[SessionItem]) -> int:
    """Return the number of steps taken in the trace."""
    return len([item for item in trace if item.role == "assistant"])


def out_tokens(trace: list[SessionItem]) -> int:
    """Return the (approx) total token cost of the agent's output."""
    # NOTE: not dollar value, since depends on Agent's model

    agent_steps = "".join([item.content for item in trace if item.role == "assistant"])
    return len(tokenizer.encode(agent_steps, disallowed_special=()))


def in_tokens(trace: list[SessionItem]) -> int:
    """Return the (approx) total token cost of the env's input."""
    # NOTE: not dollar value, since depends on Agent's model

    user_steps = "".join([item.content for item in trace if item.role != "assistant"])
    return len(tokenizer.encode(user_steps))


def is_exact_match(pred: int | str | list, target: int | str | list) -> bool:
    """Return True if the prediction is an exact match to the target.
    Also considers ["x"] and "x" as equivalent.
    """
    # Normalize both sides to lists for consistent comparison
    def normalize(value: int | str | list) -> list:
        if isinstance(value, list):
            return value
        return [value]
    
    return normalize(pred) == normalize(target)


def is_exact_match_lower(pred: str, target: str) -> bool:
    """Return True if the prediction is an exact match to the target."""
    return pred.strip().lower() == target.strip().lower()


def is_in_range(pred: int | float, target: int | float, tolerance: float) -> bool:
    """Return True if the prediction is within the target range."""
    return target - tolerance <= pred <= target + tolerance


def is_subset(pred: list, target: list) -> bool:
    """Return True if the prediction is a subset of the target."""
    return set(pred).issubset(set(target))


def is_superset(pred: list, target: list) -> bool:
    """Return True if the prediction is a superset of the target."""
    return set(pred).issuperset(set(target))


# TODO: once observability is setup, use metrics, traces, logs,
# and wrk2's logs to also observe the (side)-effects of agents' actions
# e.g., latency, throughput, etc.
