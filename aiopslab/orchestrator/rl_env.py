"""Reinforcement learning interface for orchestrator problems.

This module exposes a light-weight environment wrapper that allows
reinforcement-learning agents to interact with the existing orchestrator
problems using the familiar ``reset``/``step`` APIs.  The wrapper keeps track
of the orchestrator session state, generates textual observations that include
the problem description/instructions, and applies configurable reward rules
based on whether the agent successfully submits a solution, fails, or exhausts
its turn budget.

The implementation purposefully avoids bringing in a hard dependency on any
particular RL framework (Gymnasium, PettingZoo, etc.) so that downstream
projects can adapt the returned observation/metadata dictionaries to their own
training pipelines.
"""

from __future__ import annotations

import json
import re

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Tuple

from aiopslab.orchestrator.parser import ResponseParser
from aiopslab.session import SessionItem
from aiopslab.utils.status import (
    InvalidActionError,
    ResponseParsingError,
    SubmissionStatus,
)

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from aiopslab.orchestrator.orchestrator import Orchestrator


@dataclass(frozen=True)
class RewardConfig:
    """Reward values applied by :class:`ProblemRLEnvironment`.

    Attributes
    ----------
    success : float
        Reward applied when the agent produces a valid submission.
    invalid_submission : float
        Penalty applied when the submission is invalid.
    step : float
        Reward (typically a small penalty) applied for each regular step.
    timeout : float
        Penalty applied when the agent hits the maximum number of steps
        without submitting a solution.
    """

    success: float = 1.0
    invalid_submission: float = -1.0
    step: float = -0.01
    timeout: float = -0.5
    command_match_multiplier: float = 0.1


@dataclass(frozen=True)
class PowerCommand:
    """Representation of a single command from the power model."""

    api_name: str
    command: str
    importance_score: float
    type: str | None = None
    description: str | None = None
    sequence_number: int | None = None
    _pattern: re.Pattern[str] = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_pattern", self._build_pattern(self.command))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PowerCommand":
        command = payload.get("command")
        if not isinstance(command, str):
            raise ValueError("Power model command entries must include a 'command' string.")

        api_name = command.split("(", 1)[0].strip()
        importance = float(payload.get("importance_score", 0.0))

        return cls(
            api_name=api_name,
            command=command,
            importance_score=importance,
            type=payload.get("type"),
            description=payload.get("description"),
            sequence_number=payload.get("sequence_number"),
        )

    @staticmethod
    def _build_pattern(command: str) -> re.Pattern[str]:
        inner = command.split("(", 1)[1].rsplit(")", 1)[0] if "(" in command else ""
        escaped = re.escape(inner)
        pattern = re.sub(r"\\<[^>]+\\>", ".+", escaped)
        pattern = pattern.replace(r"\ ", r"\s+")
        return re.compile(f"^{pattern}$")

    def matches(self, api_name: str, args: Iterable[Any], kwargs: Mapping[str, Any]) -> bool:
        if api_name != self.api_name:
            return False
        formatted = self._format_call(args, kwargs)
        return bool(self._pattern.fullmatch(formatted))

    @staticmethod
    def _format_call(args: Iterable[Any], kwargs: Mapping[str, Any]) -> str:
        parts = [PowerCommand._format_value(arg) for arg in args]
        for key, value in kwargs.items():
            parts.append(f"{key}={PowerCommand._format_value(value)}")
        return ", ".join(parts)

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True)
        if isinstance(value, str):
            return json.dumps(value)
        return repr(value)


class PowerModelEpisode:
    """Stateful matcher that scores commands during an episode."""

    def __init__(self, commands: Iterable[PowerCommand]):
        self._remaining: List[PowerCommand] = list(commands)

    def score(self, api_name: str, args: Iterable[Any], kwargs: Mapping[str, Any]) -> float:
        for idx, command in enumerate(self._remaining):
            if command.matches(api_name, args, kwargs):
                self._remaining.pop(idx)
                return command.importance_score
        return 0.0

    def remaining(self) -> List[str]:
        return [cmd.command for cmd in self._remaining]


class PowerModel:
    """Container for power model command sequences keyed by problem ID."""

    def __init__(self, data: Mapping[str, Iterable[Mapping[str, Any]]]):
        self._commands: Dict[str, List[PowerCommand]] = {}
        for problem_id, commands in data.items():
            parsed = [PowerCommand.from_dict(cmd) for cmd in commands]
            parsed.sort(key=lambda cmd: cmd.sequence_number or 0)
            self._commands[problem_id] = parsed

    @classmethod
    def from_results(cls, results: Iterable[Mapping[str, Any]]) -> "PowerModel":
        mapping: Dict[str, List[Mapping[str, Any]]] = {}
        for record in results:
            problem_id = record.get("problem_id")
            if not problem_id:
                continue
            commands = record.get("key_commands", [])
            if isinstance(commands, Mapping):
                commands = [commands]
            if isinstance(commands, (str, bytes)):
                continue
            if not isinstance(commands, Iterable):
                continue
            mapping.setdefault(problem_id, []).extend(commands)  # type: ignore[arg-type]
        return cls(mapping)

    def start_episode(self, problem_id: str) -> Optional[PowerModelEpisode]:
        commands = self._commands.get(problem_id)
        if not commands:
            return None
        return PowerModelEpisode(commands)



class ProblemRLEnvironment:
    """RL-style environment wrapper around an orchestrator problem.

    The environment keeps the orchestrator session open so that a training
    loop can repeatedly call :meth:`reset` and :meth:`step` while receiving
    structured observations, rewards and metadata suitable for RL pipelines.
    """

    def __init__(
        self,
        orchestrator: Optional["Orchestrator"] = None,
        *,
        max_steps: int = 30,
        reward_config: Optional[RewardConfig] = None,
        power_model: Optional[PowerModel] = None,
    ) -> None:
        if orchestrator is None:
            from aiopslab.orchestrator.orchestrator import Orchestrator as _Orchestrator

            orchestrator = _Orchestrator()

        self.orchestrator = orchestrator
        self.max_steps = max_steps
        self.reward_config = reward_config or RewardConfig()
        self.power_model = power_model

        # ``Orchestrator`` expects an agent name for bookkeeping.  RL training
        # loops drive the environment directly, so we inject a lightweight name
        # without registering a full agent implementation.
        if not hasattr(self.orchestrator, "agent_name"):
            self.orchestrator.agent_name = "rl-agent"  # type: ignore[attr-defined]
        if not hasattr(self.orchestrator, "parser"):
            self.orchestrator.parser = ResponseParser()  # type: ignore[attr-defined]

        self.problem_id: Optional[str] = None
        self._actions_catalog: Dict[str, str] = {}
        self._step_count = 0
        self._done = False
        self._finalized = False
        self._last_action: Optional[str] = None
        self._last_env_response: Optional[Any] = None
        self._results: Optional[Dict[str, Any]] = None
        self._problem_desc: Optional[str] = None
        self._instructions: Optional[str] = None
        self._power_episode: Optional[PowerModelEpisode] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(
        self,
        problem_id: Optional[str] = None,
        *,
        max_steps: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment and (re)deploy the specified problem.

        Parameters
        ----------
        problem_id:
            Identifier of the problem to load.  If omitted, the previously
            loaded problem is reused.
        max_steps:
            Optional override for the number of steps available before a
            timeout occurs.

        Returns
        -------
        observation, info: tuple(dict, dict)
            Observation capturing the textual state and an ``info`` payload
            with metadata such as the available APIs.
        """

        if max_steps is not None:
            self.max_steps = max_steps

        if problem_id is not None:
            self.problem_id = problem_id
        if self.problem_id is None:
            raise ValueError("Problem ID must be provided on the initial reset().")

        self.close()

        task_desc, instructions, actions = self.orchestrator.init_problem(
            self.problem_id
        )
        self._problem_desc = task_desc
        self._instructions = instructions
        self._actions_catalog = dict(actions)

        self._power_episode = None
        if self.power_model is not None and self.problem_id is not None:
            self._power_episode = self.power_model.start_episode(self.problem_id)

        # ``init_problem`` creates a fresh session but leaves it idle.
        session = self.orchestrator.session
        if session is None:
            raise RuntimeError("Orchestrator did not create a session.")
        session.start()

        self._step_count = 0
        self._done = False
        self._finalized = False
        self._last_action = None
        self._last_env_response = None
        self._results = None

        observation = self._build_observation()
        info = self._build_info(initial=True)
        return observation, info

    def step(
        self, action: str
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute an action and advance the environment by one step."""

        if self._done:
            raise RuntimeError("Environment episode already finished. Call reset().")

        session = self.orchestrator.session
        if session is None:
            raise RuntimeError("Environment has not been reset. Call reset() first.")

        if not isinstance(action, str):
            raise TypeError("Actions must be provided as strings containing API calls.")

        # Record the agent action in the session history for downstream eval.
        session.add(SessionItem(role="assistant", content=action))
        parsed = self._parse_action(action)

        env_response = self._execute_action(parsed)
        session.add(SessionItem(role="env", content=str(env_response)))

        self._step_count += 1
        self._last_action = parsed.get("raw", action)
        self._last_env_response = env_response

        reward, terminated, truncated = self._compute_rewards(parsed, env_response)
        done = terminated or truncated

        if done:
            self._finalize_session(env_response, truncated)

        observation = self._build_observation(env_response)
        info = self._build_info(
            terminated=terminated,
            truncated=truncated,
            env_response=env_response,
        )

        self._done = done
        return observation, reward, done, info

    def close(self) -> None:
        """Close any running session and clean up cluster state."""

        if getattr(self.orchestrator, "session", None) is None:
            return

        if not self._finalized and not self._done:
            # No submission was made, but we still have to tidy resources.
            self._finalize_session(self._last_env_response, truncated=True)

        self._done = True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _parse_action(self, action: str) -> Dict[str, Any]:
        parser = getattr(self.orchestrator, "parser", None)
        if parser is None:
            parser = ResponseParser()
            self.orchestrator.parser = parser  # type: ignore[attr-defined]

        try:
            parsed = parser.parse(action)
        except ResponseParsingError as exc:  # pragma: no cover - exercised in tests
            return {
                "api_name": "",
                "args": [],
                "kwargs": {},
                "error": str(exc),
                "raw": action,
            }

        parsed["raw"] = action
        return parsed

    def _execute_action(self, parsed: Dict[str, Any]) -> Any:
        session = self.orchestrator.session
        assert session is not None  # guarded by caller

        if "error" in parsed:
            return parsed["error"]

        api_name = parsed["api_name"]
        args = parsed.get("args", [])
        kwargs = parsed.get("kwargs", {})

        if not api_name:
            return "No API call found!"

        if api_name == "submit":
            session.set_solution(args[0] if len(args) == 1 else args)

        try:
            response = session.problem.perform_action(api_name, *args, **kwargs)
        except InvalidActionError as exc:
            response = str(exc)
        except Exception as exc:  # pragma: no cover - defensive path
            response = str(exc)

        if hasattr(response, "error"):
            response = str(response)

        return response

    def _compute_rewards(
        self, parsed: Dict[str, Any], env_response: Any
    ) -> Tuple[float, bool, bool]:
        terminated = False
        truncated = False

        if env_response == SubmissionStatus.VALID_SUBMISSION:
            reward = self.reward_config.success
            terminated = True
        elif env_response == SubmissionStatus.INVALID_SUBMISSION:
            reward = self.reward_config.invalid_submission
            terminated = True
        elif self._step_count >= self.max_steps:
            reward = self.reward_config.timeout
            truncated = True
        else:
            reward = self.reward_config.step
            if self._power_episode is not None and "api_name" in parsed:
                api_name = parsed.get("api_name", "")
                args = parsed.get("args", [])
                kwargs = parsed.get("kwargs") or {}
                score = self._power_episode.score(api_name, args, kwargs)
                reward += score * self.reward_config.command_match_multiplier

        return reward, terminated, truncated

    def _finalize_session(self, env_response: Any, truncated: bool = False) -> None:
        if self._finalized:
            return

        session = getattr(self.orchestrator, "session", None)
        if session is None:
            return

        try:
            session.end()
        except Exception:  # pragma: no cover - defensive path
            pass

        duration = 0.0
        try:
            duration = session.get_duration()
        except Exception:
            pass

        results: Dict[str, Any] = {}
        should_eval = env_response == SubmissionStatus.VALID_SUBMISSION
        if should_eval or session.solution is not None:
            try:
                results = session.problem.eval(
                    session.solution,
                    session.history,
                    duration,
                )
            except Exception as exc:  # pragma: no cover - defensive path
                results = {"error": str(exc)}

        if truncated and not results:
            results = {"success": False, "reason": "timeout"}

        try:
            session.set_results(results)
        except Exception:  # pragma: no cover - defensive path
            pass

        self._results = results

        try:
            session.to_json()
        except Exception:
            pass

        # Clean up the injected fault and tear down application resources.
        problem = getattr(session, "problem", None)
        if problem is not None:
            try:
                problem.recover_fault()
            except Exception:
                pass

            app = getattr(problem, "app", None)
            if app is not None:
                try:
                    app.cleanup()
                except Exception:
                    pass

        self._finalized = True

    def _build_observation(self, env_response: Any | None = None) -> Dict[str, Any]:
        state_parts = []
        if self._problem_desc:
            state_parts.append(self._problem_desc.strip())
        if self._instructions:
            state_parts.append("Instructions:\n" + self._instructions.strip())
        if env_response is not None:
            state_parts.append("Environment:\n" + self._format_response(env_response))
        elif self._last_env_response is not None:
            state_parts.append(
                "Environment:\n" + self._format_response(self._last_env_response)
            )

        observation = {
            "state": "\n\n".join(state_parts),
            "actions_left": max(self.max_steps - self._step_count, 0),
            "last_action": self._last_action,
            "last_response": self._format_response(self._last_env_response),
            "problem_id": self.problem_id,
        }

        return observation

    def _build_info(
        self,
        *,
        initial: bool = False,
        terminated: bool = False,
        truncated: bool = False,
        env_response: Any | None = None,
    ) -> Dict[str, Any]:
        info = {
            "available_actions": list(self._actions_catalog.keys()),
            "actions": self._actions_catalog,
            "step": self._step_count,
            "terminated": terminated,
            "truncated": truncated,
            "raw_response": self._format_response(env_response),
            "results": self._results if self._results is not None else None,
        }

        if self._power_episode is not None:
            info["power_commands_remaining"] = self._power_episode.remaining()

        if initial:
            info["problem_id"] = self.problem_id
            info["task_description"] = self._problem_desc
            info["instructions"] = self._instructions
            if self._power_episode is not None:
                info["power_commands"] = info["power_commands_remaining"]

        return info

    @staticmethod
    def _format_response(response: Any | None) -> Optional[str]:
        if response is None:
            return None
        if isinstance(response, SubmissionStatus):
            return response.name
        return str(response)


__all__ = [
    "PowerCommand",
    "PowerModel",
    "ProblemRLEnvironment",
    "RewardConfig",
]

