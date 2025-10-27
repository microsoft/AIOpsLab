"""FastAPI surface for interacting with ProblemRLEnvironment sessions.

This module wraps the imperative helpers in :mod:`service` and exposes them as a
REST interface so external orchestrators (for example Echo's rollout workers)
can create environments, advance them step-by-step, and tear them down via
simple HTTP calls.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import service


app = FastAPI(
    title="AIOpsLab RL Environment API",
    description=(
        "Expose ProblemRLEnvironment reset/step/close operations so external"
        " reinforcement-learning systems can interact with the simulator over"
        " HTTP."
    ),
    version="0.1.0",
)


class RewardConfigPayload(BaseModel):
    success: Optional[float] = None
    invalid_submission: Optional[float] = None
    step: Optional[float] = None
    timeout: Optional[float] = None
    command_match_multiplier: Optional[float] = None


class ResetRequest(BaseModel):
    problem_id: str = Field(..., description="Identifier of the orchestrator problem to load.")
    max_steps: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional override for the number of turns before timeout.",
    )
    reward_config: Optional[RewardConfigPayload] = Field(
        default=None,
        description="Overrides for the reward parameters applied to the episode.",
    )
    ground_truth_dir: Optional[str] = Field(
        default=None,
        description="Path to an alternate ground-truth power model directory.",
    )


class ResetResponse(BaseModel):
    env_id: str
    observation: Dict[str, Any]
    info: Dict[str, Any]


class StepRequest(BaseModel):
    step: int = Field(..., ge=0, description="Sequential step number for the episode.")
    action: Optional[str] = Field(
        default=None,
        description="Agent action to execute on this step (required when step > 0).",
    )
    llm_response: Optional[str] = Field(
        default=None,
        description="Optional reasoning text from the policy to include in the trace.",
    )
    llm_raw_response: Optional[str] = Field(
        default=None,
        description="Optional raw generation from the policy to include in the trace.",
    )


class StepResponse(BaseModel):
    state: Any
    actions_left: int
    actions: Dict[str, Any]
    reward: float
    info: Dict[str, Any]


@app.get("/health")
def health() -> Dict[str, str]:
    """Return a heartbeat payload for monitoring probes."""

    return service.health_check()


@app.get("/problems")
def problems() -> Dict[str, Any]:
    """Expose the available problem identifiers."""

    return {"problems": service.list_problems()}


@app.get("/agents")
def agents() -> Dict[str, Any]:
    """Expose the available agent identifiers."""

    return {"agents": service.list_agents()}


def _build_reward_config(payload: RewardConfigPayload | None):
    if payload is None:
        return None

    from aiopslab.orchestrator.rl_env import RewardConfig

    kwargs = payload.model_dump(exclude_none=True)
    return RewardConfig(**kwargs)


def _handle_service_error(exc: Exception) -> HTTPException:
    if isinstance(exc, service.RLEnvironmentNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, service.RLEnvironmentFinishedError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, (service.RLEnvironmentError, ValueError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@app.post("/rl/reset", response_model=ResetResponse)
def reset(payload: ResetRequest) -> ResetResponse:
    """Create and reset a managed RL environment."""

    reward_config = _build_reward_config(payload.reward_config)
    try:
        handle = service.reset_rl_environment(
            payload.problem_id,
            max_steps=payload.max_steps,
            reward_config=reward_config,
            ground_truth_dir=payload.ground_truth_dir,
        )
        observation, info = service.get_rl_environment_state(handle.env_id)
    except Exception as exc:  # pragma: no cover - defensive mapping
        raise _handle_service_error(exc) from exc

    return ResetResponse(env_id=handle.env_id, observation=observation, info=info)


@app.post("/rl/{env_id}/step", response_model=StepResponse)
def step(env_id: str, payload: StepRequest) -> StepResponse:
    """Advance the specified environment by one step."""

    try:
        step_result = service.step_rl_environment(
            env_id,
            step=payload.step,
            action=payload.action,
            llm_response=payload.llm_response,
            llm_raw_response=payload.llm_raw_response,
        )
    except Exception as exc:  # pragma: no cover - defensive mapping
        raise _handle_service_error(exc) from exc

    return StepResponse(**asdict(step_result))


@app.delete("/rl/{env_id}")
def close(env_id: str = Path(..., description="Environment identifier returned by reset.")) -> JSONResponse:
    """Close and discard the specified environment."""

    try:
        service.close_rl_environment(env_id)
    except Exception as exc:  # pragma: no cover - defensive mapping
        raise _handle_service_error(exc) from exc

    return JSONResponse(status_code=204, content={})
