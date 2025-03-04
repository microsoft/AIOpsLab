# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import subprocess
import threading
import time
import yaml
from typing import List
from aiopslab.service.helm import Helm
from aiopslab.service.kubectl import KubeCtl
from aiopslab.generators.fault.base import FaultInjector

class SymptomFaultInjector(FaultInjector):
    def __init__(self, namespace: str):
        super().__init__(namespace)
        self.namespace = namespace
        self.kubectl = KubeCtl()
        self.kubectl.create_namespace_if_not_exist("chaos-mesh")
        Helm.add_repo("chaos-mesh", "https://charts.chaos-mesh.org")
        chaos_configs = {
            "release_name": "chaos-mesh",
            "chart_path": "chaos-mesh/chaos-mesh",
            "namespace": "chaos-mesh",
            "version": "2.6.2",
        }

        container_runtime = self.kubectl.get_container_runtime()

        if "docker" in container_runtime:
            pass
        elif "containerd" in container_runtime:
            chaos_configs["extra_args"] = [
                "--set chaosDaemon.runtime=containerd",
                "--set chaosDaemon.socketPath=/run/containerd/containerd.sock",
            ]
        else:
            raise ValueError(f"Unsupported container runtime: {container_runtime}")

        Helm.install(**chaos_configs)

    def create_chaos_experiment(self, experiment_yaml: dict, experiment_name: str):
        chaos_yaml_path = f"/tmp/{experiment_name}.yaml"
        with open(chaos_yaml_path, "w") as file:
            yaml.dump(experiment_yaml, file)
        command = f"kubectl apply -f {chaos_yaml_path}"
        result = self.kubectl.exec_command(command)
        print(f"Applied {experiment_name} chaos experiment: {result}")

    def delete_chaos_experiment(self, experiment_name: str):
        chaos_yaml_path = f"/tmp/{experiment_name}.yaml"
        command = f"kubectl delete -f {chaos_yaml_path}"
        result = self.kubectl.exec_command(command)
        print(f"Cleaned up chaos experiment: {result}")

    def recover_pod_failure(self):
        self.delete_chaos_experiment("pod-failure")

    def inject_pod_failure(self, microservices: List[str], duration: str = "200s"):
        """
        Inject a pod failure fault.
        """
        chaos_experiment = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "PodChaos",
            "metadata": {"name": "pod-failure-experiment", "namespace": self.namespace},
            "spec": {
                "action": "pod-failure",
                "mode": "one",
                "duration": duration,
                "selector": {
                    "labelSelectors": {"io.kompose.service": ", ".join(microservices)}
                },
            },
        }

        self.create_chaos_experiment(chaos_experiment, "pod-failure")

    def recover_network_loss(self):
        self.delete_chaos_experiment("network-loss")

    def inject_network_loss(self, microservices: List[str], duration: str = "200s"):
        """
        Inject a network loss fault.
        """
        chaos_experiment = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "NetworkChaos",
            "metadata": {"name": "loss", "namespace": self.namespace},
            "spec": {
                "action": "loss",
                "mode": "one",
                "duration": duration,
                "selector": {
                    "namespaces": [self.namespace],
                    "labelSelectors": {"io.kompose.service": ", ".join(microservices)},
                },
                "loss": {"loss": "99", "correlation": "100"},
            },
        }

        self.create_chaos_experiment(chaos_experiment, "network-loss")

    def inject_container_kill(self, microservice: str, containers: List[str]):
        """
        Inject a container kill.
        """
        chaos_experiment = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "PodChaos",
            "metadata": {"name": "container-kill", "namespace": self.namespace},
            "spec": {
                "action": "container-kill",
                "mode": "one",
                "duration": "200s",
                "selector": {"labelSelectors": {"io.kompose.service": microservice}},
                "containerNames": containers
                if isinstance(containers, list)
                else [containers],
            },
        }

        self.create_chaos_experiment(chaos_experiment, "container-kill")

    def recover_container_kill(self):
        self.delete_chaos_experiment("container-kill")

    def inject_network_delay(
        self,
        microservices: List[str],
        duration: str = "200s",
        latency: str = "10s",
        jitter: str = "0ms",
    ):
        """
        Inject a network delay fault.

        Args:
            microservices (List[str]): A list of microservice names or labels to target.
            duration (str): The duration of the network delay.
            latency (str): The amount of delay to introduce.
            jitter (str): The jitter for the delay.
        """
        chaos_experiment = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "NetworkChaos",
            "metadata": {"name": "delay", "namespace": self.namespace},
            "spec": {
                "action": "delay",
                "mode": "one",
                "duration": duration,
                "selector": {
                    "labelSelectors": {"io.kompose.service": ", ".join(microservices)}
                },
                "delay": {"latency": latency, "correlation": "100", "jitter": jitter},
            },
        }

        self.create_chaos_experiment(chaos_experiment, "network-delay")

    def recover_network_delay(self):
        self.delete_chaos_experiment("network-delay")

    def inject_pod_kill(self, microservices: List[str], duration: str = "200s"):
        """
        Inject a pod kill fault targeting specified microservices by label in the configured namespace.

        Args:
            microservices (List[str]): A list of microservices labels to target for the pod kill experiment.
            duration (str): The duration for which the pod kill fault should be active.
        """
        chaos_experiment = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "PodChaos",
            "metadata": {"name": "pod-kill", "namespace": self.namespace},
            "spec": {
                "action": "pod-kill",
                "mode": "one",
                "duration": duration,
                "selector": {
                    "labelSelectors": {"io.kompose.service": ", ".join(microservices)}
                },
            },
        }

        self.create_chaos_experiment(chaos_experiment, "pod-kill")

    def recover_pod_kill(self):
        self.delete_chaos_experiment("pod-kill")

    # IMPORTANT NOTE:
    # Kernel fault is not working and is a known bug in chaos-mesh 0> https://github.com/xlab-uiuc/agent-ops/pull/10#issuecomment-2468992285
    # This code is untested as we're waiting for a resolution to the bug to retry.
    def inject_kernel_fault(self, microservices: List[str]):
        """
        Injects a kernel fault targeting the specified function in the kernel call chain.
        """
        chaos_experiment = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "KernelChaos",
            "metadata": {"name": "kernel-chaos", "namespace": self.namespace},
            "spec": {
                "mode": "one",
                "selector": {
                    "labelSelectors": {"io.kompose.service": ", ".join(microservices)}
                },
                "failKernRequest": {
                    "callchain": [{"funcname": "__x64_sys_mount"}],
                    "failtype": 0,
                },
            },
        }

        self.create_chaos_experiment(chaos_experiment, "kernel-chaos")

    def recover_kernel_fault(self):
        self.delete_chaos_experiment("kernel-chaos")


if __name__ == "__main__":
    namespace = "test-hotel-reservation"
    microservices = ["geo"]
    fault_type = "pod_failure"
    injector = SymptomFaultInjector(namespace)
    injector._inject(fault_type, microservices, "30s")
    injector._recover(fault_type)
