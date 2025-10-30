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


class StubEchoClient:
    def __init__(self, job_id: str = "echo-job-123") -> None:
        self.job_id = job_id
        self.created_jobs: list[list[tuple[str, int]]] = []
        self.events: list[dict[str, object]] = []
        self.status_payload: dict[str, object] = {"job_id": job_id, "state": "pending"}
        self.expected_runs = 0

    async def create_job(self, problems):
        job_spec: list[tuple[str, int]] = []
        self.expected_runs = 0
        for problem in problems:
            pid = getattr(problem, "problem_id", None) or getattr(problem, "id", None)
            runs = getattr(problem, "runs", 0)
            job_spec.append((pid, runs))
            self.expected_runs += runs
        self.created_jobs.append(job_spec)
        self.status_payload["expected_runs"] = self.expected_runs
        return {"job_id": self.job_id, "expected_runs": self.expected_runs}

    async def post_event(self, job_id: str, payload: dict):
        assert job_id == self.job_id
        self.events.append(payload)
        event = payload.get("event")
        if event == "job_finished":
            self.status_payload["state"] = "done"
        elif event == "job_failed":
            self.status_payload["state"] = "failed"
            if "reason" in payload:
                self.status_payload["failure_reason"] = payload["reason"]
        return {"status": "ok"}

    async def get_status(self, job_id: str):
        assert job_id == self.job_id
        return dict(self.status_payload)

    async def get_results(self, job_id: str):
        assert job_id == self.job_id
        return {"job_id": self.job_id, "runs": []}

    async def get_events(self, job_id: str):
        assert job_id == self.job_id
        return {"events": list(self.events)}


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

    echo_clients: list[StubEchoClient] = []

    def _make_echo_client(config):
        client_id = f"echo-job-{len(echo_clients) + 1}"
        client = StubEchoClient(job_id=client_id)
        echo_clients.append(client)
        return client

    monkeypatch.setattr(service_api, "_create_echo_client", _make_echo_client)

    test_client = TestClient(service_api.app)
    return test_client, envs, echo_clients


def test_training_job_executes_and_records_results(client, monkeypatch):
    test_client, envs, echo_clients = client
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
        "echo": {"url": "http://echo.example"},
    }

    response = test_client.post("/echo/jobs", json=payload)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert job_id == "echo-job-1"
    assert echo_clients and echo_clients[0].job_id == job_id

    job = service_api._TRAINING_JOBS[job_id]
    if job.task is not None:
        job.task.cancel()
        job.task = None
    if echo_clients:
        echo_clients[0].events.clear()

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

    events = echo_clients[0].events
    assert [evt["event"] for evt in events] == [
        "initializing",
        "run_started",
        "run_finished",
        "job_finished",
    ]
    finished = events[2]
    assert finished["problem_id"] == "prob-1"
    assert finished["run_index"] == 0
    payload_dict = finished["payload"]
    assert payload_dict["done"] is True
    assert len(payload_dict["steps"]) == 3
    assert payload_dict["steps"][1]["action"].startswith("```python")
    assert payload_dict["total_reward"] == pytest.approx(3.0)

    assert providers and len(providers[0].messages) == 2
    assert envs and envs[0].closed is True


def test_training_job_failure_reports_error(client, monkeypatch):
    test_client, envs, echo_clients = client

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
    assert echo_clients == []
