import importlib.util
import json
import sys
import tempfile
import types
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _register_module(name: str, module: types.ModuleType):
    original = sys.modules.get(name)
    sys.modules[name] = module
    return original


# Stub lightweight modules so we can import the RL environment without a full
# cluster/configuration setup.
_original_modules = {}

aiopslab_pkg = types.ModuleType("aiopslab")
aiopslab_pkg.__path__ = [str(REPO_ROOT / "aiopslab")]
_original_modules["aiopslab"] = _register_module("aiopslab", aiopslab_pkg)

orchestrator_pkg = types.ModuleType("aiopslab.orchestrator")
orchestrator_pkg.__path__ = [str(REPO_ROOT / "aiopslab" / "orchestrator")]
_original_modules["aiopslab.orchestrator"] = _register_module(
    "aiopslab.orchestrator", orchestrator_pkg
)

status_module = types.ModuleType("aiopslab.utils.status")


class SubmissionStatus(Enum):
    VALID_SUBMISSION = "valid"
    INVALID_SUBMISSION = "invalid"


class InvalidActionError(Exception):
    def __init__(self, action_name: str):
        super().__init__(f"Invalid action: {action_name}")


class ResponseParsingError(Exception):
    pass


status_module.SubmissionStatus = SubmissionStatus
status_module.InvalidActionError = InvalidActionError
status_module.ResponseParsingError = ResponseParsingError
_original_modules["aiopslab.utils.status"] = _register_module(
    "aiopslab.utils.status", status_module
)

session_module = types.ModuleType("aiopslab.session")


class SessionItem:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    @classmethod
    def model_validate(cls, data):
        return cls(**data)

    def model_dump(self):
        return {"role": self.role, "content": self.content}


session_module.SessionItem = SessionItem
_original_modules["aiopslab.session"] = _register_module(
    "aiopslab.session", session_module
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parser_module = _load_module(
    "aiopslab.orchestrator.parser", REPO_ROOT / "aiopslab" / "orchestrator" / "parser.py"
)
rl_env_module = _load_module(
    "aiopslab.orchestrator.rl_env",
    REPO_ROOT / "aiopslab" / "orchestrator" / "rl_env.py",
)

ResponseParser = parser_module.ResponseParser
ProblemRLEnvironment = rl_env_module.ProblemRLEnvironment
RewardConfig = rl_env_module.RewardConfig


class StubApp:
    def __init__(self):
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1


class StubProblem:
    def __init__(self):
        self.app = StubApp()
        self.recover_calls = 0
        self.eval_invocations = []

    def perform_action(self, api_name, *args, **kwargs):
        if api_name == "exec_shell":
            command = args[0] if args else ""
            return f"ran: {command}"
        if api_name == "submit":
            return SubmissionStatus.VALID_SUBMISSION
        raise InvalidActionError(api_name)

    def eval(self, soln, trace, duration):
        self.eval_invocations.append((soln, trace, duration))
        return {"success": soln == "good"}

    def recover_fault(self):
        self.recover_calls += 1


class StubSession:
    def __init__(self):
        self.history = []
        self.solution = None
        self.results = {}
        self.problem = StubProblem()
        self.pid = None
        self.canonical_pid = None

    def start(self):
        self.start_time = 0.0

    def end(self):
        self.end_time = 1.0

    def get_duration(self):
        return 1.0

    def add(self, item):
        if isinstance(item, SessionItem):
            self.history.append(item)
        elif isinstance(item, dict):
            self.history.append(SessionItem.model_validate(item))
        else:
            raise TypeError

    def set_solution(self, solution):
        self.solution = solution

    def set_results(self, results):
        self.results = results

    def to_json(self):
        pass


class StubOrchestrator:
    def __init__(self):
        self.agent_name = "rl-test"
        self.parser = ResponseParser()
        self.session = None
        self.init_calls = 0

    def init_problem(self, problem_id):
        self.init_calls += 1
        self.session = StubSession()
        self.session.pid = problem_id
        self.session.canonical_pid = problem_id
        description = f"Task description for {problem_id}"
        instructions = "Respond with APIs."
        actions = {"exec_shell": "Run shell", "submit": "Submit"}
        return description, instructions, actions


def _action(api_call: str) -> str:
    return f"Action:```\n{api_call}\n```"


@contextmanager
def ground_truth_dir(data):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for problem_id, commands in data.items():
            payload = {"problem_id": problem_id, "key_commands": commands}
            (directory / f"{problem_id}.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
        yield directory


def test_reset_returns_initial_observation():
    orchestrator = StubOrchestrator()
    with ground_truth_dir({"problem-1": []}) as gt_dir:
        env = ProblemRLEnvironment(
            orchestrator=orchestrator, max_steps=5, ground_truth_dir=gt_dir
        )

        obs, info = env.reset("problem-1")

        assert "problem-1" in obs["state"]
        assert obs["actions_left"] == 5
        assert info["available_actions"] == ["exec_shell", "submit"]
        assert info["problem_id"] == "problem-1"


def test_step_exec_shell_returns_step_penalty():
    reward_cfg = RewardConfig(step=-0.25)
    orchestrator = StubOrchestrator()
    with ground_truth_dir({"problem-2": []}) as gt_dir:
        env = ProblemRLEnvironment(
            orchestrator=orchestrator,
            max_steps=3,
            reward_config=reward_cfg,
            ground_truth_dir=gt_dir,
        )
        env.reset("problem-2")

        obs, reward, done, info = env.step(_action('exec_shell("ls")'))

        assert "ran: ls" in obs["state"]
        assert reward == -0.25
        assert done is False
        assert info["terminated"] is False
        assert info["truncated"] is False


def test_power_model_bonus_applied_to_matching_command():
    orchestrator = StubOrchestrator()
    reward_cfg = RewardConfig(step=-0.25, command_match_multiplier=0.01)
    commands = [
        {
            "command": 'exec_shell("ls")',
            "importance_score": 6,
            "sequence_number": 1,
        }
    ]

    with ground_truth_dir({"problem-2": commands}) as gt_dir:
        env = ProblemRLEnvironment(
            orchestrator=orchestrator,
            max_steps=3,
            reward_config=reward_cfg,
            ground_truth_dir=gt_dir,
        )

        obs, info = env.reset("problem-2")
        assert "power_commands" in info
        assert info["power_commands_remaining"] == ['exec_shell("ls")']

        _, reward, done, step_info = env.step(_action('exec_shell("ls")'))

        assert done is False
        assert reward == -0.25 + 0.06
        assert step_info["power_commands_remaining"] == []


def test_submit_ends_episode_and_records_results():
    orchestrator = StubOrchestrator()
    with ground_truth_dir({"problem-3": []}) as gt_dir:
        env = ProblemRLEnvironment(
            orchestrator=orchestrator, max_steps=4, ground_truth_dir=gt_dir
        )
        env.reset("problem-3")

        env.step(_action('exec_shell("ls")'))
        obs, reward, done, info = env.step(_action('submit("good")'))

        assert done is True
        assert reward == env.reward_config.success
        assert info["terminated"] is True
        assert info["results"] == {"success": True}

        problem = orchestrator.session.problem
        assert problem.recover_calls == 1
        assert problem.app.cleanup_calls == 1
        assert len(problem.eval_invocations) == 1
        assert orchestrator.session.results == {"success": True}


def test_timeout_truncates_episode():
    orchestrator = StubOrchestrator()
    with ground_truth_dir({"problem-4": []}) as gt_dir:
        env = ProblemRLEnvironment(
            orchestrator=orchestrator, max_steps=1, ground_truth_dir=gt_dir
        )
        env.reset("problem-4")

        obs, reward, done, info = env.step(_action('exec_shell("status")'))

        assert done is True
        assert info["truncated"] is True
        assert reward == env.reward_config.timeout
        assert info["results"] == {"success": False, "reason": "timeout"}

