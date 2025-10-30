import asyncio
import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def _load_retry_components():
    originals: dict[str, types.ModuleType | None] = {}

    def register(name: str, module: types.ModuleType) -> None:
        originals[name] = sys.modules.get(name)
        sys.modules[name] = module

    # Minimal stub packages so retry_orchestrator can import without heavy deps
    aiopslab_pkg = types.ModuleType("aiopslab")
    aiopslab_pkg.__path__ = [str(REPO_ROOT / "aiopslab")]
    register("aiopslab", aiopslab_pkg)

    orchestrator_pkg = types.ModuleType("aiopslab.orchestrator")
    orchestrator_pkg.__path__ = []
    register("aiopslab.orchestrator", orchestrator_pkg)

    orchestrator_module = types.ModuleType("aiopslab.orchestrator.orchestrator")

    class _StubOrchestrator:
        pass

    orchestrator_module.Orchestrator = _StubOrchestrator
    register("aiopslab.orchestrator.orchestrator", orchestrator_module)

    tasks_pkg = types.ModuleType("aiopslab.orchestrator.tasks")
    tasks_pkg.__path__ = []
    register("aiopslab.orchestrator.tasks", tasks_pkg)

    variant_module = types.ModuleType("aiopslab.orchestrator.tasks.variant_task")

    class VariantGenerator:
        def __init__(self, base_config):
            self.base_config = base_config
            self.generated_variants = []

        def generate_variants(self, num_variants: int = 3):
            return []

        def get_next_variant(self):
            if not self.generated_variants:
                self.generated_variants = self.generate_variants()
            if self.generated_variants:
                return self.generated_variants.pop(0)
            return None

    class VariantTask:
        def __init__(self, variant_generator=None):
            self.variant_generator = variant_generator
            self.current_variant = None
            self.variant_history = []

        def apply_variant(self, variant_config):
            self.current_variant = variant_config
            self.variant_history.append(variant_config)
            for key, value in variant_config.items():
                setattr(self, key, value)

        def get_next_variant(self):
            if self.variant_generator:
                return self.variant_generator.get_next_variant()
            return None

        def reset_to_base(self):
            if self.variant_generator:
                for key, value in self.variant_generator.base_config.items():
                    setattr(self, key, value)
            self.current_variant = None

        def get_variant_summary(self):
            if self.current_variant:
                items = [f"{k}={v}" for k, v in self.current_variant.items()]
                return f"Variant: {', '.join(items)}"
            return "Base configuration"

    variant_module.VariantGenerator = VariantGenerator
    variant_module.VariantTask = VariantTask
    register("aiopslab.orchestrator.tasks.variant_task", variant_module)

    session_module = types.ModuleType("aiopslab.session")

    class Session:
        def __init__(self, results_dir=None):
            self.results_dir = results_dir
            self.problem = None
            self.pid = None
            self.canonical_pid = None
            self.agent_name = None

        def set_problem(self, problem, pid=None, canonical_pid=None):
            self.problem = problem
            self.pid = pid
            self.canonical_pid = canonical_pid

        def set_agent(self, agent_name):
            self.agent_name = agent_name

    session_module.Session = Session
    register("aiopslab.session", session_module)

    status_module = types.ModuleType("aiopslab.utils.status")

    class SubmissionStatus:
        VALID_SUBMISSION = object()
        INVALID_SUBMISSION = object()

    class ResponseParsingError(Exception):
        pass

    class InvalidActionError(Exception):
        pass

    status_module.SubmissionStatus = SubmissionStatus
    status_module.ResponseParsingError = ResponseParsingError
    status_module.InvalidActionError = InvalidActionError
    register("aiopslab.utils.status", status_module)

    spec = importlib.util.spec_from_file_location(
        "aiopslab.orchestrator.retry_orchestrator",
        REPO_ROOT / "aiopslab" / "orchestrator" / "retry_orchestrator.py",
    )
    retry_module = importlib.util.module_from_spec(spec)
    register("aiopslab.orchestrator.retry_orchestrator", retry_module)
    spec.loader.exec_module(retry_module)  # type: ignore[arg-type]

    try:
        yield {
            "RetryOrchestrator": retry_module.RetryOrchestrator,
            "VariantTaskBase": retry_module.VariantTask,
            "VariantGeneratorBase": variant_module.VariantGenerator,
            "SessionBase": session_module.Session,
        }
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        for mod_name in (
            "aiopslab.utils.status",
            "aiopslab.orchestrator.parser",
            "aiopslab.orchestrator.retry_orchestrator",
        ):
            sys.modules.pop(mod_name, None)


def test_retry_orchestrator_retries_with_variants():
    with _load_retry_components() as env:
        RetryOrchestrator = env["RetryOrchestrator"]
        VariantTaskBase = env["VariantTaskBase"]
        VariantGeneratorBase = env["VariantGeneratorBase"]
        SessionBase = env["SessionBase"]

        class DummySession(SessionBase):
            pass

        class DummyApp:
            def __init__(self):
                self.deploy_calls = 0
                self.cleanup_calls = 0

            def deploy(self):
                self.deploy_calls += 1

            def cleanup(self):
                self.cleanup_calls += 1

        class ListVariantGenerator(VariantGeneratorBase):
            def __init__(self, base_config, variants):
                super().__init__(base_config)
                self._variants = [variant.copy() for variant in variants]

            def generate_variants(self, num_variants: int = 3):
                generated = []
                for _ in range(min(num_variants, len(self._variants))):
                    config = self.base_config.copy()
                    config.update(self._variants.pop(0))
                    generated.append(config)
                return generated

        class DummyVariantProblem(VariantTaskBase):
            def __init__(self, variants):
                self.app = DummyApp()
                self.faulty_service = "base"
                self.recover_calls = 0
                self.injected_services = []
                self.workload_calls = 0
                generator = ListVariantGenerator({"faulty_service": "base"}, variants)
                super().__init__(generator)

            def inject_fault(self):
                self.injected_services.append(self.faulty_service)

            def recover_fault(self):
                self.recover_calls += 1

            def start_workload(self):
                self.workload_calls += 1

        class DummyOrchestrator:
            def __init__(self, problem, attempt_results):
                self.session = DummySession()
                self.session.set_problem(problem, pid="dummy-problem", canonical_pid="dummy-problem")
                self.session.set_agent("agent")
                self.agent_name = "agent"
                self._attempt_results = list(attempt_results)
                self.start_calls = 0
                self.results_dir = None
                self.probs = types.SimpleNamespace(
                    variant_mode="static",
                    get_problem_instance=lambda pid: problem,
                    get_canonical_id=lambda pid: pid,
                )

            async def start_problem(self, max_steps):
                result = self._attempt_results[self.start_calls]
                self.start_calls += 1
                return result

            def register_agent(self, agent, name="agent"):
                self.agent_name = name

        problem = DummyVariantProblem([{"faulty_service": "checkout"}])
        problem.inject_fault()

        orchestrator = DummyOrchestrator(
            problem,
            [
                {"results": {"success": False}},
                {"results": {"success": True}},
            ],
        )

        retry = RetryOrchestrator(
            orchestrator=orchestrator,
            max_retries=1,
            enable_variants=True,
            retry_delay=0,
        )

        results = asyncio.run(retry.start_problem(max_steps=1))

        assert results["final_success"] is True
        assert results["retry_count"] == 1
        assert len(results["attempts"]) == 2
        assert results["attempts"][0]["results"]["results"]["success"] is False
        assert results["attempts"][1]["results"]["results"]["success"] is True
        assert results["attempts"][1]["variant"] == {"faulty_service": "checkout"}
        assert problem.injected_services == ["base", "checkout"]
        assert problem.app.deploy_calls == 1
        assert problem.app.cleanup_calls == 1
        assert problem.recover_calls == 1
        assert problem.workload_calls == 1
        assert orchestrator.start_calls == 2
