from __future__ import annotations

from pathlib import Path
import types
from unittest.mock import MagicMock, patch

CONFIG_PATH = Path(__file__).resolve().parent.parent / "aiopslab" / "config.yml"
if not CONFIG_PATH.exists():
    CONFIG_PATH.write_text("data_dir: data\n")
    (CONFIG_PATH.parent / "data").mkdir(exist_ok=True)

import pytest

from aiopslab.generators.fault.inject_app import ApplicationFaultInjector
from aiopslab.generators.fault.inject_operator import K8SOperatorFaultInjector
from aiopslab.generators.fault.inject_symp import SymptomFaultInjector
from aiopslab.generators.fault.inject_virtual import VirtualizationFaultInjector


class _FakeKubectl:
    def __init__(self):
        self.create_namespace_if_not_exist = MagicMock()
        self.get_container_runtime = MagicMock(return_value="docker")
        self.exec_command = MagicMock(return_value="ok")


@pytest.fixture
def symptom_injector():
    fake_kubectl = _FakeKubectl()
    with patch(
        "aiopslab.generators.fault.inject_symp.KubeCtl", return_value=fake_kubectl
    ), patch("aiopslab.generators.fault.inject_symp.Helm.install"), patch(
        "aiopslab.generators.fault.inject_symp.Helm.add_repo"
    ):
        injector = SymptomFaultInjector("test")
    return injector


def test_symptom_network_loss_uses_variant_values(symptom_injector):
    captured = []
    with patch.object(
        symptom_injector,
        "create_chaos_experiment",
        side_effect=lambda spec, name: captured.append(spec),
    ):
        symptom_injector.inject_network_loss(
            ["svc"], duration="9s", loss=42, correlation="17"
        )

    assert captured, "Chaos experiment was not created"
    experiment = captured[0]
    assert experiment["spec"]["duration"] == "9s"
    assert experiment["spec"]["loss"] == {"loss": "42", "correlation": "17"}


def test_symptom_network_delay_uses_variant_values(symptom_injector):
    captured = []
    with patch.object(
        symptom_injector,
        "create_chaos_experiment",
        side_effect=lambda spec, name: captured.append(spec),
    ):
        symptom_injector.inject_network_delay(
            ["svc"], duration="11s", latency="30ms", jitter="7ms", correlation="42"
        )

    experiment = captured[0]
    assert experiment["spec"]["duration"] == "11s"
    assert experiment["spec"]["delay"] == {
        "latency": "30ms",
        "correlation": "42",
        "jitter": "7ms",
    }


def test_symptom_container_kill_normalizes_containers(symptom_injector):
    captured = []
    with patch.object(
        symptom_injector,
        "create_chaos_experiment",
        side_effect=lambda spec, name: captured.append(spec),
    ):
        symptom_injector.inject_container_kill(
            "svc",
            containers=("c1", "c2"),
            duration="5s",
        )

    experiment = captured[0]
    assert experiment["spec"]["duration"] == "5s"
    assert experiment["spec"]["containerNames"] == ["c1", "c2"]


@pytest.fixture
def virtualization_setup():
    kubectl = MagicMock()
    kubectl.exec_command.return_value = "ok"
    kubectl.get_service_json.return_value = {
        "spec": {"ports": [{"targetPort": 8080}]}
    }
    kubectl.patch_service = MagicMock()
    kubectl.list_pods.return_value = types.SimpleNamespace(items=[])
    docker = MagicMock()
    with patch(
        "aiopslab.generators.fault.inject_virtual.KubeCtl", return_value=kubectl
    ), patch(
        "aiopslab.generators.fault.inject_virtual.Docker", return_value=docker
    ):
        injector = VirtualizationFaultInjector("ns")
    return injector, kubectl


def test_virtualization_misconfig_ports_accepts_arguments(virtualization_setup):
    injector, kubectl = virtualization_setup
    kubectl.patch_service.reset_mock()

    injector.inject_misconfig_k8s(["svc"], from_port=8080, to_port=7070)

    patched_config = kubectl.patch_service.call_args[0][2]
    ports = patched_config["spec"]["ports"]
    assert ports[0]["targetPort"] == 7070


def test_virtualization_scaling_uses_requested_replica(virtualization_setup):
    injector, kubectl = virtualization_setup
    kubectl.exec_command.reset_mock()

    injector.inject_scale_pods_to_zero(["svc"], replicas=4)
    injector.recover_scale_pods_to_zero(["svc"], replicas=2)

    commands = [call.args[0] for call in kubectl.exec_command.call_args_list]
    assert "--replicas=4" in commands[0]
    assert "--replicas=2" in commands[1]


def test_virtualization_node_selector_accepts_variant(virtualization_setup):
    injector, kubectl = virtualization_setup
    selector = {"custom": "value"}
    with patch.object(
        injector,
        "_get_deployment_yaml",
        return_value={"spec": {"template": {"spec": {}}}},
    ), patch.object(
        injector, "_write_yaml_to_file", return_value="/tmp/file.yaml"
    ) as write_mock:
        injector.inject_assign_to_non_existent_node(["svc"], node_selector=selector)

    written_yaml = write_mock.call_args[0][1]
    assert written_yaml["spec"]["template"]["spec"]["nodeSelector"] == selector


@pytest.fixture
def application_setup():
    kubectl = MagicMock()
    kubectl.exec_command.return_value = "ok"
    with patch(
        "aiopslab.generators.fault.inject_app.KubeCtl", return_value=kubectl
    ):
        injector = ApplicationFaultInjector("ns")
    return injector, kubectl


def _pods(names):
    return [types.SimpleNamespace(metadata=types.SimpleNamespace(name=name)) for name in names]


def test_application_revoke_auth_uses_variant_scripts(application_setup):
    injector, kubectl = application_setup
    kubectl.list_pods.return_value = types.SimpleNamespace(
        items=_pods(["mongodb-rate-0", "rate-app-0"])
    )
    with patch("aiopslab.generators.fault.inject_app.time.sleep"), patch.object(
        injector, "delete_service_pods"
    ) as delete_mock:
        injector.inject_revoke_auth(
            [
                {
                    "service": "mongodb-rate",
                    "inject_script": "/tmp/custom-inject.sh",
                    "recover_script": "/tmp/custom-recover.sh",
                    "service_pod_selector": "rate-app",
                }
            ]
        )

    commands = [call.args[0] for call in kubectl.exec_command.call_args_list]
    assert any("/tmp/custom-inject.sh" in command for command in commands)
    delete_mock.assert_called_once()
    assert delete_mock.call_args[0][0] == ["rate-app-0"]


def test_application_revoke_auth_recover_uses_variant_scripts(application_setup):
    injector, kubectl = application_setup
    kubectl.list_pods.return_value = types.SimpleNamespace(
        items=_pods(["mongodb-rate-0", "rate-app-0"])
    )
    with patch.object(injector, "delete_service_pods") as delete_mock:
        injector.recover_revoke_auth(
            [
                {
                    "service": "mongodb-rate",
                    "inject_script": "/tmp/custom-inject.sh",
                    "recover_script": "/tmp/custom-recover.sh",
                    "service_pod_selector": "rate-app",
                }
            ]
        )

    commands = [call.args[0] for call in kubectl.exec_command.call_args_list]
    assert any("/tmp/custom-recover.sh" in command for command in commands)
    delete_mock.assert_called_once()
    assert delete_mock.call_args[0][0] == ["rate-app-0"]


def test_application_misconfig_app_uses_variant_images(application_setup):
    injector, kubectl = application_setup
    container = types.SimpleNamespace(name="hotel-reserv-geo", image="baseline")
    deployment = types.SimpleNamespace(
        spec=types.SimpleNamespace(
            template=types.SimpleNamespace(
                spec=types.SimpleNamespace(containers=[container])
            )
        )
    )
    kubectl.get_deployment.return_value = deployment
    with patch("aiopslab.generators.fault.inject_app.time.sleep"):
        injector.inject_misconfig_app(
            [
                {
                    "service": "geo",
                    "inject_image": "buggy:image",
                    "recover_image": "baseline:image",
                }
            ]
        )

    assert container.image == "buggy:image"

    injector.recover_misconfig_app(
        [
            {
                "service": "geo",
                "inject_image": "buggy:image",
                "recover_image": "baseline:image",
            }
        ]
    )
    assert container.image == "baseline:image"


@pytest.fixture
def operator_setup():
    kubectl = MagicMock()
    kubectl.exec_command.return_value = "ok"
    with patch(
        "aiopslab.generators.fault.inject_operator.KubeCtl", return_value=kubectl
    ):
        injector = K8SOperatorFaultInjector("ns")
    return injector


def test_operator_overload_variant_applies_replica_counts(operator_setup):
    injector = operator_setup
    with patch.object(injector, "_apply_yaml"):
        manifest = injector.inject_overload_replicas(
            {"id": "variant-a", "replicas": {"tidb": 5, "tikv": 4}}
        )

    assert manifest["spec"]["tidb"]["replicas"] == 5
    assert manifest["spec"]["tikv"]["replicas"] == 4
    assert (
        manifest["metadata"]["annotations"]["fault.aiopslab/variant-id"]
        == "variant-a"
    )


def test_operator_tolerations_respected(operator_setup):
    injector = operator_setup
    tolerations = [{"effect": "NoExecute"}]
    with patch.object(injector, "_apply_yaml"):
        manifest = injector.inject_invalid_affinity_toleration(
            {"id": "variant-b", "tolerations": tolerations}
        )

    assert manifest["spec"]["tidb"]["tolerations"] == tolerations


def test_operator_storage_class_respected(operator_setup):
    injector = operator_setup
    with patch.object(injector, "_apply_yaml"):
        manifest = injector.inject_non_existent_storage(
            {"id": "variant-c", "storage_class": "fast-ssd"}
        )

    assert manifest["spec"]["pd"]["storageClassName"] == "fast-ssd"
