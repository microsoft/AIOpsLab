"""FastAPI surface for interacting with ProblemRLEnvironment sessions.

This module wraps the imperative helpers in :mod:`service` and exposes them as a
REST interface so external orchestrators (for example Echo's rollout workers)
can create environments, advance them step-by-step, and tear them down via
simple HTTP calls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import asyncio
import atexit
import functools
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TypeVar
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic import AnyHttpUrl

import service

T = TypeVar("T")


app = FastAPI(
    title="AIOpsLab RL Environment API",
    description=(
        "Expose ProblemRLEnvironment reset/step/close operations so external"
        " reinforcement-learning systems can interact with the simulator over"
        " HTTP."
    ),
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting AIOpsLab RL Environment API...")
    logger.info("Service initialized and ready to accept connections.")
    logger.info("Available endpoints: /rl/reset, /rl/{env_id}/step, /rl/{env_id}")
    logger.info(
        "Blocking executor configured with %s worker threads.",
        _BLOCKING_POOL_SIZE,
    )


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down blocking executor...")
    _shutdown_blocking_executor(wait=True)


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


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Ensure handler exists
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

_DEFAULT_POOL_SIZE = max(32, (os.cpu_count() or 4) * 4)
_BLOCKING_POOL_SIZE = int(
    os.getenv("AIOPSLAB_BLOCKING_POOL_SIZE", str(_DEFAULT_POOL_SIZE))
)
_BLOCKING_EXECUTOR = ThreadPoolExecutor(max_workers=_BLOCKING_POOL_SIZE)
_BLOCKING_EXECUTOR_SHUTDOWN = False


def _shutdown_blocking_executor(wait: bool = False) -> None:
    global _BLOCKING_EXECUTOR_SHUTDOWN
    if _BLOCKING_EXECUTOR_SHUTDOWN:
        return
    _BLOCKING_EXECUTOR.shutdown(wait=wait)
    _BLOCKING_EXECUTOR_SHUTDOWN = True


atexit.register(_shutdown_blocking_executor, False)


async def _run_blocking(func: Callable[..., T], /, *args, **kwargs) -> T:
    """Run a blocking callable in the shared executor to avoid starving asyncio workers."""
    loop = asyncio.get_running_loop()
    bound = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(_BLOCKING_EXECUTOR, bound)


_DEFAULT_SYSTEM_PROMPT = (
    "You are an SRE assistant interacting with a Kubernetes-based incident "
    "simulation. Respond with exactly one valid API call enclosed in a markdown "
    "code block on each turn."
)


class ChatCompletionConfig(BaseModel):
    api_key: Optional[str] = Field(
        default=None,
        description="Optional API key for the chat completion endpoint.",
    )
    model: str = Field(..., min_length=1, description="Model identifier exposed by the chat completion endpoint.")
    base_url: Optional[AnyHttpUrl] = Field(
        default=None,
        description="Optional base URL for non-OpenAI compatible deployments.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature passed to the chat completion API.",
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter forwarded to the chat completion API.",
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        description="Maximum number of tokens to request for each action.",
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty applied during generation.",
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty applied during generation.",
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None,
        description="Optional stop sequences forwarded to the chat completion API.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt prepended to every dialogue.",
    )
    timeout: float = Field(
        default=60.0,
        gt=0.0,
        description="Timeout (seconds) for chat completion requests.",
    )
    extra_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional HTTP headers to include in chat completion requests.",
    )


class EchoServerConfig(BaseModel):
    url: AnyHttpUrl = Field(
        ...,
        description="Base URL for the Echo job callback server (e.g. http://127.0.0.1:8098).",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional bearer token used when contacting the Echo server.",
    )
    timeout: float = Field(
        default=30.0,
        gt=0.0,
        description="Timeout (seconds) for Echo synchronization requests.",
    )
    extra_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional HTTP headers attached to Echo synchronization requests.",
    )


class ProblemRunPayload(BaseModel):
    problem_id: str = Field(..., description="Identifier of the problem to deploy.")
    runs: int = Field(
        default=1,
        ge=1,
        description="Number of independent episodes to execute for this problem.",
    )
    max_steps: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional override for the environment step budget.",
    )
    reward_config: Optional[RewardConfigPayload] = Field(
        default=None,
        description="Problem-specific reward configuration overrides.",
    )
    ground_truth_dir: Optional[str] = Field(
        default=None,
        description="Optional override for the ground truth directory.",
    )


class BatchRunRequest(BaseModel):
    problems: List[ProblemRunPayload] = Field(
        ...,
        min_length=1,
        description="Collection of problems and repetition counts to execute.",
    )
    concurrency: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Maximum number of concurrent episodes to run.",
    )
    chat: ChatCompletionConfig
    echo: Optional[EchoServerConfig] = Field(
        default=None,
        description="Optional configuration for streaming rollouts to an Echo server.",
    )


class JobRunStep(BaseModel):
    step: int
    observation: Optional[Dict[str, Any]] = None
    action: Optional[str] = None
    state: Optional[Any] = None
    reward: Optional[float] = None
    actions_left: Optional[int] = None
    info: Dict[str, Any] = Field(default_factory=dict)
    llm_response: Optional[str] = None
    llm_raw_response: Optional[str] = None


class JobRunResult(BaseModel):
    run_id: str
    problem_id: str
    run_index: int
    env_id: str
    steps: List[JobRunStep]
    total_reward: float
    done: bool
    echo_post_status: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    total_runs: int
    completed_runs: int
    failed_runs: int
    echo_failures: int
    error: Optional[str] = None


class JobResultsResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    runs: List[JobRunResult]


class _ActionProvider:
    async def generate(self, messages: Sequence[Dict[str, str]]) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class _OpenAIActionProvider(_ActionProvider):
    def __init__(self, config: ChatCompletionConfig) -> None:
        if not config.api_key:
            raise ValueError("ChatCompletionConfig.api_key is required for OpenAI-compatible action provider.")
        headers = dict(config.extra_headers)
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=str(config.base_url) if config.base_url is not None else None,
            default_headers=headers or None,
            timeout=config.timeout,
        )
        self._config = config

    async def generate(self, messages: Sequence[Dict[str, str]]) -> str:
        response = await self._client.chat.completions.create(
            model=self._config.model,
            messages=list(messages),
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            max_tokens=self._config.max_tokens,
            presence_penalty=self._config.presence_penalty,
            frequency_penalty=self._config.frequency_penalty,
            stop=self._config.stop_sequences,
        )
        if not response.choices:
            raise RuntimeError("Chat completion response did not contain choices.")
        message = response.choices[0].message
        content = message.content if message is not None else None
        if not content:
            raise RuntimeError("Chat completion response did not include any content.")
        return content.strip()


class _HttpActionProvider(_ActionProvider):
    def __init__(self, config: ChatCompletionConfig) -> None:
        if config.base_url is None:
            raise ValueError("base_url must be provided when api_key is omitted.")
        self._config = config
        base_url = str(config.base_url).rstrip("/")
        self._endpoint = f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        headers.update(config.extra_headers)
        self._headers = headers

    async def generate(self, messages: Sequence[Dict[str, str]]) -> str:
        payload: Dict[str, Any] = {
            "model": self._config.model,
            "messages": list(messages),
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "max_tokens": self._config.max_tokens,
            "presence_penalty": self._config.presence_penalty,
            "frequency_penalty": self._config.frequency_penalty,
        }
        if self._config.stop_sequences is not None:
            payload["stop"] = self._config.stop_sequences

        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            response = await client.post(self._endpoint, json=payload, headers=self._headers)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            snippet = response.text[:200]
            raise RuntimeError(f"Chat completion request failed: {exc.response.status_code} {snippet}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            snippet = response.text[:200]
            raise RuntimeError(f"Chat completion response was not valid JSON: {snippet}") from exc

        choices = data.get("choices")
        if not choices:
            raise RuntimeError("Chat completion response did not contain choices.")
        choice = choices[0]
        content = None
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
        if not content:
            # Some implementations use 'text' instead of structured messages.
            content = choice.get("text")
        if not content:
            raise RuntimeError("Chat completion response did not include any content.")
        return str(content).strip()


@dataclass
class CachedEnvironment:
    env_id: str
    problem_id: str
    dirty: bool = False
    in_use: bool = False


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


_ENABLE_ENV_REUSE = _env_flag("AIOPSLAB_ENABLE_ENV_REUSE", False)


@dataclass
class _TrainingJob:
    job_id: str
    request: BatchRunRequest
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"] = "pending"
    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    echo_failures: int = 0
    error: Optional[str] = None
    results: List[JobRunResult] = field(default_factory=list)
    task: Optional[asyncio.Task] = None
    echo_client: Optional[_EchoJobClient] = None
    echo_job_id: Optional[str] = None
    env_cache: Dict[str, List[CachedEnvironment]] = field(default_factory=dict)

    def status_payload(self) -> JobStatusResponse:
        return JobStatusResponse(
            job_id=self.job_id,
            status=self.status,
            total_runs=self.total_runs,
            completed_runs=self.completed_runs,
            failed_runs=self.failed_runs,
            echo_failures=self.echo_failures,
            error=self.error,
        )

    def results_payload(self) -> JobResultsResponse:
        return JobResultsResponse(job_id=self.job_id, status=self.status, runs=self.results)


_TRAINING_JOBS: Dict[str, _TrainingJob] = {}
_TRAINING_JOBS_LOCK = asyncio.Lock()


def _create_action_provider(config: ChatCompletionConfig) -> _ActionProvider:
    if config.api_key:
        return _OpenAIActionProvider(config)
    return _HttpActionProvider(config)


def _create_echo_client(config: EchoServerConfig) -> "_EchoJobClient":
    return _EchoJobClient(config)


class _EchoJobClient:
    """Thin HTTP client used to interact with the Echo job lifecycle API."""

    def __init__(self, config: EchoServerConfig) -> None:
        self._config = config
        self._base_url = str(config.url).rstrip("/")
        headers = {"Content-Type": "application/json"}
        headers.update(config.extra_headers)
        if config.api_key:
            headers.setdefault("Authorization", f"Bearer {config.api_key}")
        self._headers = headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not path.startswith("/"):
            path = "/" + path
        url = f"{self._base_url}{path}"
        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            kwargs: Dict[str, Any] = {"headers": self._headers}
            if json_payload is not None:
                kwargs["json"] = json_payload
            response = await client.request(method, url, **kwargs)
        response.raise_for_status()
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            return {}

    async def create_job(self, problems: Sequence["ProblemRunPayload"]) -> Dict[str, Any]:
        payload = {
            "problems": [
                {"id": problem.problem_id, "runs": problem.runs}
                for problem in problems
            ]
        }
        return await self._request("POST", "/jobs", json_payload=payload)

    async def post_event(self, job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", f"/jobs/{job_id}/events", json_payload=payload)

    async def get_status(self, job_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/jobs/{job_id}/status")

    async def get_results(self, job_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/jobs/{job_id}/results")

    async def get_events(self, job_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/jobs/{job_id}/events")


def _dump_json(payload: Any) -> str:
    try:
        return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return json.dumps(str(payload))


def _trim_conversation_from_end(
    conversation: List[Dict[str, str]], 
    max_tokens: int = 28000
) -> List[Dict[str, str]]:
    """从后往前截断对话历史，保留最新消息
    
    策略：始终保留系统提示（conversation[0]），从后往前累加消息
    直到达到token限制。适合AIOps场景，因为最新的诊断信息最重要。
    
    Args:
        conversation: 对话历史列表
        max_tokens: 最大token数（默认28000，为32768模型上限预留buffer）
    
    Returns:
        截断后的对话历史
    """
    if len(conversation) <= 2:
        return conversation
    
    # 粗略估算token数（4字符≈1token）
    def estimate_tokens(msg: Dict[str, str]) -> int:
        return len(json.dumps(msg, ensure_ascii=False)) // 4
    
    system_msg = conversation[0]  # 始终保留系统提示
    messages = conversation[1:]
    
    kept = []
    token_count = estimate_tokens(system_msg)
    
    # 从后往前累加消息
    for msg in reversed(messages):
        msg_tokens = estimate_tokens(msg)
        if token_count + msg_tokens > max_tokens:
            # 达到限制，停止添加
            trimmed_count = len(messages) - len(kept)
            if trimmed_count > 0:
                logger.warning(
                    f"⚠️ Token limit reached! Trimmed {trimmed_count} old messages "
                    f"(kept {len(kept)} recent messages, ~{token_count} tokens)"
                )
            break
        kept.insert(0, msg)
        token_count += msg_tokens
    
    result = [system_msg] + kept
    logger.info(f"📊 Conversation trimmed: {len(conversation)} → {len(result)} messages (~{token_count} tokens)")
    return result


def _format_observation_message(observation: Dict[str, Any], info: Dict[str, Any]) -> str:
    state = observation.get("state") if isinstance(observation, dict) else None
    metadata = {}
    if isinstance(observation, dict):
        metadata = {k: v for k, v in observation.items() if k != "state"}
    parts: List[str] = []
    if state:
        parts.append(f"Environment state:\n{state}")
    if metadata:
        parts.append(f"Observation metadata:\n{_dump_json(metadata)}")
    if info:
        parts.append(f"Environment info:\n{_dump_json(info)}")
    if not parts:
        parts.append("The environment did not return any textual observation.")
    return "\n\n".join(parts)


def _format_step_feedback(step: service.RLEnvironmentStep) -> str:
    env_info = step.info.get("environment", {}) if isinstance(step.info, dict) else {}
    other_info = {k: v for k, v in step.info.items() if k != "environment"} if isinstance(step.info, dict) else {}
    parts = [f"Reward: {step.reward}", f"Actions remaining: {step.actions_left}"]
    if step.state:
        parts.insert(0, f"Observation after action:\n{step.state}")
    if env_info:
        parts.append(f"Environment metadata:\n{_dump_json(env_info)}")
    if other_info:
        parts.append(f"Auxiliary info:\n{_dump_json(other_info)}")
    return "\n\n".join(parts)


def _extract_action_text(message: str) -> str:
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("Chat completion returned an empty action.")
    fence = re.search(r"```[\s\S]+?```", cleaned)
    if fence:
        return fence.group(0).strip()
    return cleaned


def _build_run_payload(run_result: JobRunResult) -> Dict[str, Any]:
    return {
        "run_id": run_result.run_id,
        "env_id": run_result.env_id,
        "total_reward": run_result.total_reward,
        "done": run_result.done,
        "steps": [step.model_dump() for step in run_result.steps],
    }


def _build_partial_results(job: _TrainingJob) -> List[Dict[str, Any]]:
    return [
        {
            "problem_id": result.problem_id,
            "run_index": result.run_index,
            "payload": _build_run_payload(result),
        }
        for result in job.results
    ]


async def _post_echo_event(job: _TrainingJob, payload: Dict[str, Any]) -> Optional[str]:
    if job.echo_client is None or job.echo_job_id is None:
        return None
    try:
        await job.echo_client.post_event(job.echo_job_id, payload)
    except Exception as exc:  # pragma: no cover - exercised via monkeypatch in tests
        logger.exception("Failed to post event to Echo server", exc_info=exc)
        return str(exc)
    return None


async def _run_training_job(job: _TrainingJob) -> None:
    job.status = "running"
    init_error = await _post_echo_event(job, {"event": "initializing"})
    if init_error:
        job.echo_failures += 1
    sem = asyncio.Semaphore(job.request.concurrency)
    tasks = []
    for problem in job.request.problems:
        for idx in range(problem.runs):
            tasks.append(
                asyncio.create_task(
                    _run_single_episode(job, problem, idx, sem),
                    name=f"episode-{problem.problem_id}-{idx}",
                )
            )

    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors: List[str] = []
    for outcome in results:
        if isinstance(outcome, JobRunResult):
            job.results.append(outcome)
            job.completed_runs += 1
        elif isinstance(outcome, Exception):
            job.failed_runs += 1
            errors.append(str(outcome))

    if errors:
        job.status = "failed"
        job.error = "; ".join(errors)
        failure_payload: Dict[str, Any] = {"event": "job_failed", "reason": job.error}
        partial = _build_partial_results(job)
        if partial:
            failure_payload["partial"] = partial
        err = await _post_echo_event(job, failure_payload)
        if err:
            job.echo_failures += 1
    else:
        job.status = "succeeded"
        err = await _post_echo_event(job, {"event": "job_finished"})
        if err:
            job.echo_failures += 1

    if _ENABLE_ENV_REUSE:
        await _cleanup_reusable_environments(job)


async def _run_single_episode(
    job: _TrainingJob,
    problem: ProblemRunPayload,
    run_index: int,
    semaphore: asyncio.Semaphore,
) -> JobRunResult:
    async with semaphore:
        logger.info(f"Starting episode {run_index} for problem {problem.problem_id}")
        reward_config = _build_reward_config(problem.reward_config)
        reuse_env = _ENABLE_ENV_REUSE
        cached_env: Optional[CachedEnvironment] = None
        if reuse_env:
            cached_env = _acquire_cached_environment(job, problem.problem_id)

        logger.info(f"Resetting RL environment for {problem.problem_id}...")
        start_time = asyncio.get_event_loop().time()
        try:
            handle = await _run_blocking(
                service.reset_rl_environment,
                problem.problem_id,
                max_steps=problem.max_steps,
                reward_config=reward_config,
                ground_truth_dir=problem.ground_truth_dir,
                env_id=cached_env.env_id if cached_env else None,
            )
            if cached_env is None and reuse_env:
                cached_env = _register_cached_environment(job, problem.problem_id, handle.env_id)
            end_time = asyncio.get_event_loop().time()
            logger.info(
                f"Environment reset complete for {problem.problem_id} (took {end_time - start_time:.2f}s). Env ID: {handle.env_id}"
            )
        except Exception as e:
            logger.error(f"Failed to reset environment for {problem.problem_id}: {e}")
            if reuse_env and cached_env:
                cached_env.dirty = True
                cached_env.in_use = False
                try:
                    await _run_blocking(service.close_rl_environment, cached_env.env_id)
                except service.RLEnvironmentNotFoundError:
                    pass
                _remove_cached_environment(job, cached_env)
            raise e

        env_id = handle.env_id
        try:
            logger.info(f"Getting initial state for Env ID: {env_id}")
            observation, info = await _run_blocking(service.get_rl_environment_state, env_id)
            if not isinstance(observation, dict):
                observation = {"state": observation}
            if not isinstance(info, dict):
                info = {"raw": info}

            steps: List[JobRunStep] = [
                JobRunStep(step=0, observation=observation, info=info)
            ]

            system_prompt = job.request.chat.system_prompt or _DEFAULT_SYSTEM_PROMPT
            conversation: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _format_observation_message(observation, info)},
            ]

            logger.info(f"Posting run_started event to Echo for Env ID: {env_id}")
            start_error = await _post_echo_event(
                job,
                {
                    "event": "run_started",
                    "problem_id": problem.problem_id,
                    "run_index": run_index,
                },
            )
            if start_error:
                logger.error(f"Failed to post run_started for {env_id}: {start_error}")
                job.echo_failures += 1
            else:
                logger.info(f"Successfully posted run_started for {env_id}")

            action_provider = _create_action_provider(job.request.chat)

            done = False
            total_reward = 0.0
            step_index = 0
            max_steps = problem.max_steps or observation.get("actions_left") or 30

            while not done and step_index < max_steps:
                logger.info(f"Env {env_id} | Step {step_index + 1}/{max_steps} | Requesting LLM action...")
                try:
                    # Token限制：从后往前截断对话历史
                    conversation = _trim_conversation_from_end(list(conversation), max_tokens=28000)
                    llm_message = await action_provider.generate(list(conversation))
                    logger.info(f"Env {env_id} | Step {step_index + 1} | LLM Response received: {llm_message[:100]}...")
                except Exception as e:
                    logger.error(f"Env {env_id} | Step {step_index + 1} | LLM generation failed: {e}")
                    raise e

                action_text = _extract_action_text(llm_message)
                step_index += 1
                
                logger.info(f"Env {env_id} | Step {step_index} | Executing action: {action_text}")
                step_start = asyncio.get_event_loop().time()
                step_result = await _run_blocking(
                    service.step_rl_environment,
                    env_id,
                    step=step_index,
                    action=action_text,
                    llm_response=llm_message,
                    llm_raw_response=llm_message,
                )
                step_end = asyncio.get_event_loop().time()
                logger.info(f"Env {env_id} | Step {step_index} | Execution complete ({step_end - step_start:.2f}s). Reward: {step_result.reward}")

                total_reward += step_result.reward
                done_flag = False
                if isinstance(step_result.info, dict):
                    done_flag = step_result.info.get("environment", {}).get("done", False)
                done = done_flag or step_result.actions_left <= 0
                
                if done:
                    logger.info(f"Env {env_id} | Episode finished. Total Reward: {total_reward}")

                steps.append(
                    JobRunStep(
                        step=step_index,
                        action=action_text,
                        state=step_result.state,
                        reward=step_result.reward,
                        actions_left=step_result.actions_left,
                        info=step_result.info if isinstance(step_result.info, dict) else {"raw": step_result.info},
                        llm_response=llm_message,
                        llm_raw_response=llm_message,
                    )
                )

                conversation.append({"role": "assistant", "content": llm_message})
                conversation.append({"role": "user", "content": _format_step_feedback(step_result)})

            run_result = JobRunResult(
                run_id=uuid4().hex,
                problem_id=problem.problem_id,
                run_index=run_index,
                env_id=env_id,
                steps=steps,
                total_reward=total_reward,
                done=done,
            )

            logger.info(f"Posting run_finished event to Echo for Env ID: {env_id}")
            echo_error = await _post_echo_event(
                job,
                {
                    "event": "run_finished",
                    "problem_id": problem.problem_id,
                    "run_index": run_index,
                    "payload": _build_run_payload(run_result),
                },
            )
            if echo_error:
                logger.error(f"Failed to post run_finished for {env_id}: {echo_error}")
                job.echo_failures += 1
                run_result.echo_post_status = f"error: {echo_error}"
            else:
                logger.info(f"Successfully posted run_finished for {env_id}")
                run_result.echo_post_status = "ok"

            return run_result
        except Exception as e:
            logger.error(f"Error in episode loop for {env_id}: {e}")
            if reuse_env and cached_env:
                cached_env.dirty = True
                cached_env.in_use = False
                _remove_cached_environment(job, cached_env)
            try:
                await _run_blocking(service.close_rl_environment, env_id)
                logger.info(f"Closed environment {env_id} after error.")
            except service.RLEnvironmentNotFoundError:
                pass
            raise
        finally:
            if reuse_env:
                if cached_env:
                    if cached_env.dirty:
                        try:
                            await _run_blocking(service.close_rl_environment, env_id)
                            logger.info(f"Closed environment {env_id} (cleanup, dirty).")
                        except service.RLEnvironmentNotFoundError:
                            pass
                        _remove_cached_environment(job, cached_env)
                    else:
                        cached_env.in_use = False
            else:
                try:
                    await _run_blocking(service.close_rl_environment, env_id)
                    logger.info(f"Closed environment {env_id} (cleanup).")
                except service.RLEnvironmentNotFoundError:
                    pass


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


def _acquire_cached_environment(job: _TrainingJob, problem_id: str) -> Optional[CachedEnvironment]:
    entries = job.env_cache.get(problem_id)
    if not entries:
        return None
    for entry in entries:
        if not entry.in_use and not entry.dirty:
            entry.in_use = True
            return entry
    return None


def _register_cached_environment(job: _TrainingJob, problem_id: str, env_id: str) -> CachedEnvironment:
    cached = CachedEnvironment(env_id=env_id, problem_id=problem_id, in_use=True)
    job.env_cache.setdefault(problem_id, []).append(cached)
    return cached


def _remove_cached_environment(job: _TrainingJob, cached_env: CachedEnvironment) -> None:
    entries = job.env_cache.get(cached_env.problem_id)
    if not entries:
        return
    try:
        entries.remove(cached_env)
    except ValueError:
        return
    if not entries:
        job.env_cache.pop(cached_env.problem_id, None)


async def _cleanup_reusable_environments(job: _TrainingJob) -> None:
    if not job.env_cache:
        return
    tasks = [
        _run_blocking(service.close_rl_environment, env.env_id)
        for envs in job.env_cache.values()
        for env in envs
    ]
    await asyncio.gather(*tasks, return_exceptions=True)
    job.env_cache.clear()


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


@app.post("/echo/jobs", response_model=JobStatusResponse)
async def start_training_job(payload: BatchRunRequest) -> JobStatusResponse:
    total_runs = sum(problem.runs for problem in payload.problems)
    if total_runs <= 0:
        raise HTTPException(status_code=400, detail="At least one episode must be requested.")

    job_id = uuid4().hex
    echo_client: Optional[_EchoJobClient] = None
    if payload.echo is not None:
        try:
            echo_client = _create_echo_client(payload.echo)
            creation = await echo_client.create_job(payload.problems)
        except Exception as exc:
            # Log the full error for debugging
            logger.exception("Failed to register job with Echo server", exc_info=exc)
            raise HTTPException(
                status_code=502,
                detail=f"Failed to register job with Echo server: {exc}",
            ) from exc

        remote_job_id = creation.get("job_id")
        if not remote_job_id:
            raise HTTPException(status_code=502, detail="Echo server response did not include a job_id.")
        job_id = str(remote_job_id)
        expected_runs = creation.get("expected_runs")
        if expected_runs is not None and expected_runs != total_runs:
            logger.warning(
                "Echo server expected %s runs but request specified %s; continuing.",
                expected_runs,
                total_runs,
            )

    job = _TrainingJob(
        job_id=job_id,
        request=payload,
        total_runs=total_runs,
        echo_client=echo_client,
        echo_job_id=job_id if echo_client is not None else None,
    )
    async with _TRAINING_JOBS_LOCK:
        _TRAINING_JOBS[job.job_id] = job

    job.task = asyncio.create_task(_run_training_job(job), name=f"echo-job-{job.job_id}")
    return job.status_payload()


def _get_job(job_id: str) -> _TrainingJob:
    job = _TRAINING_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@app.get("/echo/jobs/{job_id}", response_model=JobStatusResponse)
async def get_training_job(job_id: str) -> JobStatusResponse:
    async with _TRAINING_JOBS_LOCK:
        job = _get_job(job_id)
        return job.status_payload()


@app.get("/echo/jobs/{job_id}/results", response_model=JobResultsResponse)
async def get_training_job_results(job_id: str) -> JobResultsResponse:
    async with _TRAINING_JOBS_LOCK:
        job = _get_job(job_id)
        return job.results_payload()
