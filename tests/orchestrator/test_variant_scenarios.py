import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

config_module = types.ModuleType("aiopslab.config")


class DummyConfig:
    def __init__(self, *args, **kwargs):
        self._values = {"data_dir": "data"}

    def get(self, key, default=None):
        return self._values.get(key, default)


def _dummy_get_kube_context():
    return None


config_module.Config = DummyConfig
config_module.get_kube_context = _dummy_get_kube_context
sys.modules.setdefault("aiopslab.config", config_module)

(REPO_ROOT / "aiopslab" / "data").mkdir(parents=True, exist_ok=True)

observer_pkg = types.ModuleType("aiopslab.observer")
observer_pkg.__path__ = []
metric_module = types.ModuleType("aiopslab.observer.metric_api")


class DummyPrometheusAPI:
    def __init__(self, *args, **kwargs):
        pass


metric_module.PrometheusAPI = DummyPrometheusAPI
observer_pkg.metric_api = metric_module
trace_module = types.ModuleType("aiopslab.observer.trace_api")


class DummyTraceAPI:
    def __init__(self, *args, **kwargs):
        pass


trace_module.TraceAPI = DummyTraceAPI
log_module = types.ModuleType("aiopslab.observer.log_api")


class DummyLogAPI:
    def __init__(self, *args, **kwargs):
        pass


log_module.logAPI = DummyLogAPI
observer_pkg.trace_api = trace_module
observer_pkg.log_api = log_module
sys.modules.setdefault("aiopslab.observer", observer_pkg)
sys.modules.setdefault("aiopslab.observer.metric_api", metric_module)
sys.modules.setdefault("aiopslab.observer.trace_api", trace_module)
sys.modules.setdefault("aiopslab.observer.log_api", log_module)

from aiopslab.orchestrator.problems.misconfig_app import misconfig_app_variant as misconfig_variant
from aiopslab.orchestrator.problems.network_loss import network_loss_variant
from aiopslab.orchestrator.problems.operator_misoperation import (
    operator_misoperation_variant,
)
from aiopslab.orchestrator.problems.pod_kill import pod_kill_variant
from aiopslab.orchestrator.problems.registry import ProblemRegistry


class DummyContainerStatus:
    def __init__(self):
        state = types.SimpleNamespace(
            waiting=None, terminated=types.SimpleNamespace(reason="Completed")
        )
        self.state = state
        self.ready = True
        self.name = "container"


class DummyPod:
    def __init__(self):
        status = types.SimpleNamespace(container_statuses=[DummyContainerStatus()])
        self.status = status


class DummyPodList:
    items = [DummyPod()]


class DummyKubeCtl:
    def __init__(self, *args, **kwargs):
        pass

    def create_namespace_if_not_exist(self, *args, **kwargs):
        pass

    def get_container_runtime(self):
        return "docker"

    def exec_command(self, *args, **kwargs):
        return ""

    def create_or_update_configmap(self, *args, **kwargs):
        pass

    def apply_configs(self, *args, **kwargs):
        pass

    def wait_for_ready(self, *args, **kwargs):
        pass

    def delete_configs(self, *args, **kwargs):
        pass

    def delete_namespace(self, *args, **kwargs):
        pass

    def list_pods(self, *args, **kwargs):
        return DummyPodList()

    def get_deployment(self, *args, **kwargs):
        spec = types.SimpleNamespace(replicas=1)
        status = types.SimpleNamespace(available_replicas=1)
        return types.SimpleNamespace(spec=spec, status=status)


class DummyWrk:
    def __init__(self, *args, **kwargs):
        pass

    def start_workload(self, *args, **kwargs):
        pass


class DummyHotelReservation:
    helm_configs: dict

    def __init__(self):
        self.namespace = "test-namespace"
        self.helm_configs = {}
        self.frontend_service = "frontend"
        self.frontend_port = 5000

    def get_app_summary(self):
        return "HotelReservation summary"


class DummyTiDBCluster:
    helm_configs: dict

    def __init__(self):
        self.namespace = "tidb"
        self.helm_configs = {}

    def get_app_summary(self):
        return "TiDB summary"


class DummySymptomFaultInjector:
    def __init__(self, namespace: str):
        self.namespace = namespace


class DummyOperatorFaultInjector:
    def __init__(self, namespace: str):
        self.namespace = namespace


@pytest.fixture(autouse=True)
def stubbed_env(monkeypatch):
    monkeypatch.setattr("aiopslab.service.kubectl.KubeCtl", DummyKubeCtl)
    monkeypatch.setattr("aiopslab.orchestrator.tasks.base.KubeCtl", DummyKubeCtl)

    for module in (pod_kill_variant, network_loss_variant, misconfig_variant):
        monkeypatch.setattr(module, "HotelReservation", DummyHotelReservation)
        monkeypatch.setattr(module, "KubeCtl", DummyKubeCtl)
        monkeypatch.setattr(module, "Wrk", DummyWrk)

    monkeypatch.setattr(pod_kill_variant, "SymptomFaultInjector", DummySymptomFaultInjector)
    monkeypatch.setattr(network_loss_variant, "SymptomFaultInjector", DummySymptomFaultInjector)

    monkeypatch.setattr(
        operator_misoperation_variant, "K8SOperatorFaultInjector", DummyOperatorFaultInjector
    )
    monkeypatch.setattr(operator_misoperation_variant, "TiDBCluster", DummyTiDBCluster)

    from aiopslab.orchestrator.tasks import analysis as analysis_module
    from aiopslab.orchestrator.tasks import mitigation as mitigation_module

    original_analysis_init = analysis_module.AnalysisTask.__init__

    def patched_analysis_init(self, app=None):
        if app is None:
            app = DummyHotelReservation()
        original_analysis_init(self, app)

    monkeypatch.setattr(analysis_module.AnalysisTask, "__init__", patched_analysis_init)

    original_mitigation_init = mitigation_module.MitigationTask.__init__

    def patched_mitigation_init(self, app=None):
        if app is None:
            app = DummyHotelReservation()
        original_mitigation_init(self, app)

    monkeypatch.setattr(
        mitigation_module.MitigationTask, "__init__", patched_mitigation_init
    )

    yield


@pytest.mark.parametrize(
    "analysis_cls, mitigation_cls",
    [
        (pod_kill_variant.PodKillVariantAnalysis, pod_kill_variant.PodKillVariantMitigation),
        (
            network_loss_variant.NetworkLossVariantAnalysis,
            network_loss_variant.NetworkLossVariantMitigation,
        ),
    ],
)
def test_symptom_injector_wiring(analysis_cls, mitigation_cls):
    analysis = analysis_cls(enable_variants=False)
    mitigation = mitigation_cls(enable_variants=False)

    assert isinstance(analysis.injector, DummySymptomFaultInjector)
    assert analysis.injector.namespace == analysis.namespace

    assert isinstance(mitigation.injector, DummySymptomFaultInjector)
    assert mitigation.injector.namespace == mitigation.namespace


def test_operator_injector_wiring():
    analysis = operator_misoperation_variant.K8SOperatorMisoperationVariantAnalysis(
        enable_variants=False
    )
    mitigation = operator_misoperation_variant.K8SOperatorMisoperationVariantMitigation(
        enable_variants=False
    )

    assert isinstance(analysis.injector, DummyOperatorFaultInjector)
    assert analysis.injector.namespace == "tidb-cluster"

    assert isinstance(mitigation.injector, DummyOperatorFaultInjector)
    assert mitigation.injector.namespace == "tidb-cluster"


def test_registry_returns_variant_rca_mitigation_tasks():
    registry = ProblemRegistry(variant_mode="variant")

    pod_kill_analysis = registry.get_problem_instance("pod_kill_hotel_res-analysis")
    assert isinstance(pod_kill_analysis, pod_kill_variant.PodKillVariantAnalysis)

    misconfig_mitigation = registry.get_problem_instance("misconfig_app_hotel_res-mitigation")
    assert isinstance(misconfig_mitigation, misconfig_variant.MisconfigAppVariantMitigation)

    network_loss_analysis = registry.get_problem_instance("network_loss_hotel_res-analysis")
    assert isinstance(network_loss_analysis, network_loss_variant.NetworkLossVariantAnalysis)


def test_misconfig_variant_analysis_semantics():
    analysis = misconfig_variant.MisconfigAppVariantAnalysis(enable_variants=False)
    good = analysis.eval(
        {"system_level": "Application", "fault_type": "Misconfiguration"},
        [],
        1.0,
    )

    assert good["success"] is True
    assert good["system_level_correct"] is True
    assert good["fault_type_correct"] is True

    analysis = misconfig_variant.MisconfigAppVariantAnalysis(enable_variants=False)
    bad = analysis.eval(
        {"system_level": "Hardware", "fault_type": "Operation Error"},
        [],
        1.0,
    )

    assert bad["success"] is False
    assert bad["system_level_correct"] is False
    assert bad["fault_type_correct"] is False
