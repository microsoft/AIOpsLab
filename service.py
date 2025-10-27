from __future__ import annotations

import asyncio
import logging
import os
import traceback
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:  # pragma: no cover - import only for static analysis
    from aiopslab.orchestrator import ProblemRLEnvironment
    from aiopslab.orchestrator.rl_env import RewardConfig


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("aiopslab-service")


class SimulationError(RuntimeError):
    """Raised when a simulation cannot be executed."""


class RLEnvironmentError(RuntimeError):
    """Base class for managed RL environment errors."""


class RLEnvironmentNotFoundError(RLEnvironmentError):
    """Raised when callers reference an unknown environment identifier."""


class RLEnvironmentFinishedError(RLEnvironmentError):
    """Raised when callers attempt to interact with a finished environment."""


@dataclass
class SimulationRequest:
    problem_id: str
    agent_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    max_steps: Optional[int] = None
    # vLLM specific parameters
    model: Optional[str] = "Qwen/Qwen2.5-Coder-3B-Instruct"
    repetition_penalty: Optional[float] = 1.0
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 1024  # Aligned with vLLMAgent default


@dataclass
class SimulationResponse:
    agent: str
    session_id: str
    problem_id: str
    start_time: float
    end_time: float
    trace: List[Dict[str, Any]]
    results: Dict[str, Any]


@dataclass
class _ManagedRLEnvironment:
    """Container that tracks the lifecycle of a managed RL environment."""

    env: "ProblemRLEnvironment"
    initial_observation: Dict[str, Any]
    initial_info: Dict[str, Any]
    done: bool = False


@dataclass
class RLEnvironmentHandle:
    env_id: str


@dataclass
class RLEnvironmentStep:
    state: Any
    actions_left: int
    actions: Dict[str, Any]
    reward: float
    info: Dict[str, Any]


_RL_ENVIRONMENTS: Dict[str, _ManagedRLEnvironment] = {}
_RL_ENV_LOCK = Lock()


def _create_rl_environment(
    *,
    max_steps: Optional[int] = None,
    reward_config: "RewardConfig" | None = None,
    ground_truth_dir: Optional[os.PathLike[str] | str] = None,
) -> "ProblemRLEnvironment":
    """Factory used to create environments (patchable in tests)."""

    kwargs: Dict[str, Any] = {}
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    if reward_config is not None:
        kwargs["reward_config"] = reward_config
    if ground_truth_dir is not None:
        kwargs["ground_truth_dir"] = ground_truth_dir
    from aiopslab.orchestrator import ProblemRLEnvironment as _ProblemRLEnvironment

    return _ProblemRLEnvironment(**kwargs)


def _get_managed_env(env_id: str) -> _ManagedRLEnvironment:
    with _RL_ENV_LOCK:
        managed = _RL_ENVIRONMENTS.get(env_id)
    if managed is None:
        raise RLEnvironmentNotFoundError(
            f"Environment '{env_id}' not found. Did you call reset_rl_environment first?"
        )
    return managed


def list_problems() -> List[str]:
    """Return the IDs of available problems."""

    from aiopslab.orchestrator.problems.registry import ProblemRegistry

    registry = ProblemRegistry()
    return registry.get_problem_ids()


def list_agents() -> List[str]:
    """Return the IDs of registered agents."""

    from clients.registry import AgentRegistry

    registry = AgentRegistry()
    return registry.get_agent_ids()


def health_check() -> Dict[str, str]:
    """Return a simple heartbeat payload for monitoring integrations."""

    return {"status": "healthy", "service": "AIOpsLab"}


def simulate(req: SimulationRequest) -> SimulationResponse:
    """Run a full simulation for a given problem and agent."""

    from aiopslab.orchestrator import Orchestrator
    from aiopslab.orchestrator.problems.registry import ProblemRegistry
    from clients.registry import AgentRegistry

    logger.info(
        "Starting simulation with problem=%s, agent=%s, max_steps=%s",
        req.problem_id,
        req.agent_name,
        req.max_steps,
    )

    problem_registry = ProblemRegistry()
    problem = problem_registry.get_problem(req.problem_id)
    if problem is None:
        available = problem_registry.get_problem_ids()
        logger.error("Problem %s not found", req.problem_id)
        raise SimulationError(
            f"Problem {req.problem_id} not found. Available problems: {available}"
        )

    agent_registry = AgentRegistry()
    agent_cls = agent_registry.get_agent(req.agent_name)
    if agent_cls is None:
        available_agents = agent_registry.get_agent_ids()
        logger.error("Agent %s not registered", req.agent_name)
        raise SimulationError(
            f"Agent {req.agent_name} not registered. Available agents: {available_agents}"
        )

    if req.agent_name == "vllm":
        vllm_params = {
            "model": req.model,
            "repetition_penalty": req.repetition_penalty,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
        }
        agent = agent_cls(**vllm_params)
    else:
        agent = agent_cls()
    logger.info("Created agent: %s", req.agent_name)

    max_steps = req.max_steps if req.max_steps is not None else 10

    orchestrator = Orchestrator()
    orchestrator.register_agent(agent, name=f"{req.agent_name}-agent")

    try:
        problem_desc, instructs, apis = orchestrator.init_problem(req.problem_id)
        agent.init_context(problem_desc, instructs, apis)
        asyncio.run(orchestrator.start_problem(max_steps=max_steps))

        raw = orchestrator.session.to_dict()
        raw["trace"].insert(0, {"role": "system", "content": agent.system_message})
        raw["trace"].insert(1, {"role": "user", "content": agent.task_message})
        if raw["trace"] and raw["trace"][-1].get("role") == "env":
            raw["trace"].pop()

        return SimulationResponse(**raw)
    except Exception as exc:
        logger.error("Error during simulation: %s", exc)
        traceback.print_exc()
        raise SimulationError(f"Error during simulation: {exc}") from exc


def reset_rl_environment(
    problem_id: str,
    *,
    max_steps: Optional[int] = None,
    reward_config: "RewardConfig" | None = None,
    ground_truth_dir: Optional[os.PathLike[str] | str] = None,
) -> RLEnvironmentHandle:
    """Create and reset a managed RL environment for the requested problem."""

    env = _create_rl_environment(
        max_steps=max_steps,
        reward_config=reward_config,
        ground_truth_dir=ground_truth_dir,
    )
    try:
        observation, info = env.reset(problem_id)
    except Exception as exc:  # pragma: no cover - defensive path
        raise RLEnvironmentError(f"Failed to reset environment: {exc}") from exc

    env_id = uuid4().hex
    managed = _ManagedRLEnvironment(
        env=env,
        initial_observation=observation,
        initial_info=info,
    )
    with _RL_ENV_LOCK:
        _RL_ENVIRONMENTS[env_id] = managed

    return RLEnvironmentHandle(env_id=env_id)


def step_rl_environment(
    env_id: str,
    *,
    step: int,
    action: Optional[str] = None,
    llm_response: Optional[str] = None,
    llm_raw_response: Optional[str] = None,
) -> RLEnvironmentStep:
    """Advance a managed RL environment by one step."""

    managed = _get_managed_env(env_id)
    env = managed.env

    if step < 0:
        raise ValueError("step must be a non-negative integer")

    done = managed.done
    reward = 0.0
    info: Dict[str, Any] = managed.initial_info
    observation: Any = managed.initial_observation

    if step == 0:
        pass
    else:
        if managed.done:
            raise RLEnvironmentFinishedError(
                "Environment episode already finished. Please reset for a new run."
            )
        if not action:
            raise ValueError("An action is required for step > 0")
        try:
            observation, reward, done_flag, info = env.step(action)
        except Exception as exc:  # pragma: no cover - defensive path
            raise RLEnvironmentError(f"Environment step failed: {exc}") from exc

        managed.done = done_flag
        done = done_flag
        if done_flag:
            try:
                env.close()
            finally:
                with _RL_ENV_LOCK:
                    _RL_ENVIRONMENTS.pop(env_id, None)
        else:
            with _RL_ENV_LOCK:
                _RL_ENVIRONMENTS[env_id] = managed

    actions: Dict[str, Any] = {}
    if isinstance(info, dict):
        actions = info.get("actions", {})
    if not actions and isinstance(managed.initial_info, dict):
        actions = managed.initial_info.get("actions", {})

    session = getattr(env.orchestrator, "session", None)
    history_len = 0
    if session is not None and hasattr(session, "history"):
        history_len = len(getattr(session, "history"))

    env_metadata: Dict[str, Any]
    if isinstance(info, dict):
        env_metadata = dict(info)
    else:
        env_metadata = {"raw": info}
    env_metadata["done"] = done

    response_info: Dict[str, Any] = {
        "llm_response": llm_response,
        "llm_raw_response": llm_raw_response,
        "len": history_len,
        "environment": env_metadata,
    }

    state_value = observation.get("state") if isinstance(observation, dict) else observation
    actions_left = 0
    if isinstance(observation, dict):
        actions_left = observation.get("actions_left", 0)

    return RLEnvironmentStep(
        state=state_value,
        actions_left=actions_left,
        actions=actions,
        reward=reward,
        info=response_info,
    )


def get_rl_environment_state(env_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return the cached observation/info for a managed environment."""

    managed = _get_managed_env(env_id)
    return managed.initial_observation, managed.initial_info


def close_rl_environment(env_id: str) -> None:
    """Terminate and discard a managed RL environment."""

    managed = _get_managed_env(env_id)

    try:
        managed.env.close()
    finally:
        managed.done = True
        with _RL_ENV_LOCK:
            _RL_ENVIRONMENTS.pop(env_id, None)
