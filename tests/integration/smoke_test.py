# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration smoke test: full pipeline using a zero-cost dummy agent.

Requires a live Kubernetes cluster (kind) with aiopslab/config.yml present.
Run via GitHub Actions CI or locally after `kind create cluster`:

    poetry run pytest tests/integration/smoke_test.py -v -s

The test exercises the complete orchestrator path for the lightest problem in
the registry (noop_detection_hotel_reservation-1):
  deploy app → inject no-op fault → run workload → dispatch submit() action
  → evaluate → recover → cleanup

No LLM is invoked; the DummyAgent immediately submits the correct answer.
"""

import asyncio
import pytest

from aiopslab.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Dummy agent
# ---------------------------------------------------------------------------

class DummyAgent:
    """Zero-cost agent for CI smoke testing — makes no LLM or API calls.

    For a no-op detection task the correct answer is always "No" (no fault was
    injected), so we submit that immediately on the first step.
    """

    async def get_action(self, input: str) -> str:  # noqa: A002
        return '```\nsubmit("No")\n```'


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

PROBLEM_ID = "noop_detection_hotel_reservation-1"


@pytest.mark.integration
def test_noop_hotel_reservation_smoke():
    """Smoke test: run noop_detection_hotel_reservation-1 end-to-end."""
    agent = DummyAgent()
    orchestrator = Orchestrator()
    orchestrator.register_agent(agent, name="dummy")

    # --- init_problem: deploys HotelReservation, injects no-op, starts workload
    problem_desc, instructions, apis = orchestrator.init_problem(PROBLEM_ID)

    assert problem_desc, "init_problem must return a non-empty problem description"
    assert instructions, "init_problem must return non-empty instructions"
    assert apis, "init_problem must return a non-empty actions dict"
    assert "submit" in "\n".join(apis.keys()), (
        "available actions must include 'submit'"
    )

    # --- start_problem: agent loop (max 1 step — DummyAgent submits immediately)
    output = asyncio.run(orchestrator.start_problem(max_steps=1))

    assert output is not None, "start_problem must return a result dict"

    results = output.get("results", {})
    assert results, f"results dict must be non-empty; got: {output}"
    assert results.get("Detection Accuracy") == "Correct", (
        f"Expected Detection Accuracy='Correct', got: {results}"
    )
