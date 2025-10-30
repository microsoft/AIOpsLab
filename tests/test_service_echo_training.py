from __future__ import annotations

import asyncio
import types

import pytest
from fastapi.testclient import TestClient

import service
import service_api


class StubRLEnvironment:
    def __init__(self) -> None:
        self.orchestrator = types.SimpleNamespace(
            session=types.SimpleNamespace(history=[])
        )
        self.closed = False
        self._step_count = 0
        self.problem_id = None

    def reset(self, problem_id):
        self.problem_id = problem_id
        observation = {"state": f"start {problem_id}", "actions_left": 2}
        info = {"actions": {"exec_shell": "run"}, "available_actions": ["exec_shell"]}
        return observation, info

    def step(self, action):
        self._step_count += 1
        observation = {
            "state": f"state {self._step_count}",
            "actions_left": max(2 - self._step_count, 0),
        }
        done = self._step_count >= 2
        info = {
            "actions": {"exec_shell": "run"},
            "terminated": done,
            "truncated": False,
            "raw_response": f"output {self._step_count}",
        }
        return observation, float(self._step_count), done, info

    def close(self):
        self.closed = True


class StubActionProvider:
    def __init__(self) -> None:
        self.messages: list[list[dict[str, str]]] = []

    async def generate(self, messages):
        self.messages.append(messages)
        idx = len(self.messages)
        return f"```python\nexec_shell(\"cmd-{idx}\")\n```"


class FailingActionProvider:
    async def generate(self, messages):
        raise RuntimeError("model failure")


@pytest.fixture(autouse=True)
def clear_state():
    service._RL_ENVIRONMENTS.clear()
    service_api._TRAINING_JOBS.clear()
    yield
    service._RL_ENVIRONMENTS.clear()
    service_api._TRAINING_JOBS.clear()


@pytest.fixture()
def client(monkeypatch):
    envs: list[StubRLEnvironment] = []

    def _factory(*args, **kwargs):
        env = StubRLEnvironment()
        envs.append(env)
        return env

    monkeypatch.setattr(service, "_create_rl_environment", _factory)

    sent_payloads: list[dict] = []

    async def _send(config, job, run_result):
        sent_payloads.append({"job_id": job.job_id, "run": run_result})
        return None

    monkeypatch.setattr(service_api, "_send_to_echo_server", _send)

    test_client = TestClient(service_api.app)
    return test_client, envs, sent_payloads


def test_training_job_executes_and_records_results(client, monkeypatch):
    test_client, envs, sent_payloads = client
    providers: list[StubActionProvider] = []

    def _make_provider(config):
        provider = StubActionProvider()
        providers.append(provider)
        return provider

    monkeypatch.setattr(service_api, "_create_action_provider", _make_provider)

    payload = {
        "problems": [{"problem_id": "prob-1", "runs": 1}],
        "concurrency": 1,
        "chat": {"api_key": "test", "model": "gpt-test"},
        "echo": {"url": "https://echo.example/rollouts"},
    }

    response = test_client.post("/echo/jobs", json=payload)
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    job = service_api._TRAINING_JOBS[job_id]
    if job.task is not None:
        job.task.cancel()
        job.task = None

    asyncio.run(service_api._run_training_job(job))

    status_response = test_client.get(f"/echo/jobs/{job_id}")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["status"] == "succeeded"
    assert status["completed_runs"] == 1
    assert status["failed_runs"] == 0

    results_response = test_client.get(f"/echo/jobs/{job_id}/results")
    assert results_response.status_code == 200
    results = results_response.json()
    assert results["job_id"] == job_id
    assert len(results["runs"]) == 1
    run = results["runs"][0]
    assert run["problem_id"] == "prob-1"
    assert run["echo_post_status"] == "ok"
    assert len(run["steps"]) == 3
    assert run["steps"][0]["observation"]["state"] == "start prob-1"
    assert run["steps"][1]["action"].startswith("```")
    assert run["steps"][2]["actions_left"] == 0

    assert len(sent_payloads) == 1
    assert providers and len(providers[0].messages) == 2
    assert envs and envs[0].closed is True


def test_training_job_failure_reports_error(client, monkeypatch):
    test_client, envs, sent_payloads = client

    def _make_provider(config):
        return FailingActionProvider()

    monkeypatch.setattr(service_api, "_create_action_provider", _make_provider)

    payload = {
        "problems": [{"problem_id": "prob-err", "runs": 1}],
        "concurrency": 1,
        "chat": {"api_key": "test", "model": "gpt-test"},
    }

    response = test_client.post("/echo/jobs", json=payload)
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    job = service_api._TRAINING_JOBS[job_id]
    if job.task is not None:
        job.task.cancel()
        job.task = None

    asyncio.run(service_api._run_training_job(job))

    status_response = test_client.get(f"/echo/jobs/{job_id}")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["status"] == "failed"
    assert status["failed_runs"] == 1
    assert status["completed_runs"] == 0
    assert status["error"]

    results_response = test_client.get(f"/echo/jobs/{job_id}/results")
    assert results_response.status_code == 200
    results = results_response.json()
    assert results["runs"] == []

    assert envs and envs[0].closed is True
    assert sent_payloads == []
