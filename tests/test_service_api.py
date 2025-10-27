from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient

import service
import service_api


class StubRLEnvironment:
    def __init__(self):
        self.problems = {}
        self.orchestrator = types.SimpleNamespace(
            session=types.SimpleNamespace(history=[])
        )
        self.closed = False
        self._step_count = 0

    def reset(self, problem_id):
        self.problems[problem_id] = True
        observation = {
            "state": f"initial state for {problem_id}",
            "actions_left": 3,
        }
        info = {
            "actions": {"exec": "execute"},
            "available_actions": ["exec"],
        }
        return observation, info

    def step(self, action):
        self._step_count += 1
        self.orchestrator.session.history.append({"role": "assistant"})
        observation = {
            "state": f"after {action}",
            "actions_left": max(3 - self._step_count, 0),
        }
        done = self._step_count >= 2
        info = {
            "actions": {"exec": "execute"},
            "terminated": done,
            "truncated": False,
        }
        return observation, 1.0, done, info

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def clear_registry():
    service._RL_ENVIRONMENTS.clear()
    yield
    service._RL_ENVIRONMENTS.clear()


@pytest.fixture()
def client(monkeypatch):
    stub_envs: list[StubRLEnvironment] = []

    def _factory(*args, **kwargs):
        env = StubRLEnvironment()
        stub_envs.append(env)
        return env

    monkeypatch.setattr(service, "_create_rl_environment", _factory)
    return TestClient(service_api.app), stub_envs


def test_reset_and_step_endpoint(client):
    test_client, stub_envs = client

    response = test_client.post("/rl/reset", json={"problem_id": "prob-1", "max_steps": 5})
    assert response.status_code == 200
    payload = response.json()
    env_id = payload["env_id"]
    assert payload["observation"]["state"] == "initial state for prob-1"
    assert payload["info"]["actions"] == {"exec": "execute"}

    step_zero = test_client.post(f"/rl/{env_id}/step", json={"step": 0})
    assert step_zero.status_code == 200
    assert step_zero.json()["reward"] == 0.0

    step_one = test_client.post(
        f"/rl/{env_id}/step",
        json={"step": 1, "action": "exec", "llm_response": "analysis"},
    )
    assert step_one.status_code == 200
    assert step_one.json()["state"] == "after exec"

    step_two = test_client.post(f"/rl/{env_id}/step", json={"step": 2, "action": "exec"})
    assert step_two.status_code == 200
    assert step_two.json()["info"]["environment"]["done"] is True
    assert stub_envs[0].closed is True

    missing = test_client.post(f"/rl/{env_id}/step", json={"step": 3, "action": "exec"})
    assert missing.status_code == 404


def test_close_endpoint(client):
    test_client, stub_envs = client

    response = test_client.post("/rl/reset", json={"problem_id": "prob-1"})
    env_id = response.json()["env_id"]
    assert stub_envs[0].closed is False

    close = test_client.delete(f"/rl/{env_id}")
    assert close.status_code == 204
    assert stub_envs[0].closed is True

    second_close = test_client.delete(f"/rl/{env_id}")
    assert second_close.status_code == 404
