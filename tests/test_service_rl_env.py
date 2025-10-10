import types

import pytest

import service


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
            "power_commands_remaining": ["exec"],
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


def test_rl_service_reset_and_steps(monkeypatch):
    stub_env = StubRLEnvironment()

    class DummyRewardConfig:
        def __init__(self, success: float) -> None:
            self.success = success

    reward_config = DummyRewardConfig(success=10.0)

    def _factory(max_steps=None, reward_config=None, ground_truth_dir=None):
        assert max_steps == 5
        assert reward_config is not None
        assert reward_config.success == 10.0
        assert ground_truth_dir == "ground"
        return stub_env

    monkeypatch.setattr(service, "_create_rl_environment", _factory)

    handle = service.reset_rl_environment(
        "prob-1",
        max_steps=5,
        reward_config=reward_config,
        ground_truth_dir="ground",
    )
    env_id = handle.env_id
    assert env_id in service._RL_ENVIRONMENTS

    step_zero = service.step_rl_environment(env_id, step=0)
    assert step_zero.state.startswith("initial state")
    assert step_zero.actions_left == 3
    assert step_zero.reward == 0.0
    assert step_zero.actions == {"exec": "execute"}
    assert step_zero.info["environment"]["done"] is False
    assert step_zero.info["len"] == 0
    assert step_zero.info["environment"]["power_commands_remaining"] == ["exec"]

    step_one = service.step_rl_environment(
        env_id,
        step=1,
        action="exec",
        llm_response="analysis",
        llm_raw_response="analysis",
    )
    assert step_one.state == "after exec"
    assert step_one.reward == 1.0
    assert step_one.info["llm_response"] == "analysis"
    assert step_one.info["llm_raw_response"] == "analysis"
    assert step_one.info["len"] == 1
    assert step_one.info["environment"]["done"] is False

    step_two = service.step_rl_environment(env_id, step=2, action="exec")
    assert step_two.info["environment"]["done"] is True
    assert stub_env.closed is True
    assert env_id not in service._RL_ENVIRONMENTS
