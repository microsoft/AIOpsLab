# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Inject security faults: workload compromise scenarios.

Unlike the operational fault injectors, these scenarios do not break the
application. The app keeps serving traffic; what changes is its *security
posture* plus a runtime intrusion event. Detecting them therefore requires
security signals (OPA policy violations, Falco runtime alerts) rather than the
usual health/latency telemetry -- which is exactly what makes them useful for
benchmarking a security-capable agent.
"""

import time

from kubernetes import client

from aiopslab.generators.fault.base import FaultInjector
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.security_tooling import SecurityTooling

HOST_VOLUME_NAME = "host-root"


class SecurityFaultInjector(FaultInjector):
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.kubectl = KubeCtl()
        self.tooling = SecurityTooling()
        # Maps a hotel-reservation service (deployment) name to its container name.
        # The HotelReservation manifests name containers `hotel-reserv-<service>`.
        self.container_name_fmt = "hotel-reserv-{service}"

    ############# FAULT LIBRARY ################
    # S.1 - priv_escalation: a workload is running privileged with the host
    #       filesystem mounted, and a runtime intrusion (shell + sensitive-file
    #       read) is executed inside it. OPA/Gatekeeper flags the misconfig;
    #       Falco flags the runtime behavior.
    def inject_priv_escalation(self, microservices: list[str]):
        """Compromise the given service(s): make privileged + simulate an intrusion."""
        # Install the security tooling FIRST so Falco is running (and Gatekeeper is
        # auditing) before the intrusion happens and can therefore capture it.
        self.tooling.deploy_all(app_namespace=self.namespace)

        for service in microservices:
            self._make_privileged(service)
            self._simulate_intrusion(service)

    def recover_priv_escalation(self, microservices: list[str]):
        """Revert the security posture and remove the tooling."""
        for service in microservices:
            self._remove_privileged(service)
        self.tooling.cleanup()

    # --- internal steps --------------------------------------------------------

    def _make_privileged(self, service: str):
        """Patch the deployment to run privileged with a hostPath mount of the node root."""
        deployment = self.kubectl.get_deployment(service, self.namespace)
        if not deployment:
            print(f"[WARN] Deployment {service} not found; cannot inject.")
            return

        container_name = self.container_name_fmt.format(service=service)
        spec = deployment.spec.template.spec

        for container in spec.containers:
            if container.name == container_name:
                container.security_context = client.V1SecurityContext(
                    privileged=True,
                    run_as_user=0,
                    allow_privilege_escalation=True,
                )
                mounts = container.volume_mounts or []
                if not any(m.name == HOST_VOLUME_NAME for m in mounts):
                    mounts.append(
                        client.V1VolumeMount(
                            name=HOST_VOLUME_NAME,
                            mount_path="/host",
                            read_only=True,
                        )
                    )
                container.volume_mounts = mounts

        volumes = spec.volumes or []
        if not any(v.name == HOST_VOLUME_NAME for v in volumes):
            volumes.append(
                client.V1Volume(
                    name=HOST_VOLUME_NAME,
                    host_path=client.V1HostPathVolumeSource(path="/", type="Directory"),
                )
            )
        spec.volumes = volumes

        self.kubectl.update_deployment(service, self.namespace, deployment)
        self._wait_rollout(service)
        print(f"Made {service} privileged with host filesystem mounted.")

    def _simulate_intrusion(self, service: str):
        """Run suspicious activity inside the compromised pod to trip Falco."""
        pod = self._get_pod_name(service)
        if not pod:
            print(f"[WARN] No running pod for {service}; skipping intrusion step.")
            return

        container_name = self.container_name_fmt.format(service=service)
        # Spawning a shell (Falco: "Terminal shell in a container"), reading
        # /etc/shadow (Falco: "Read sensitive file untrusted"), and touching the
        # host mount (container-escape reconnaissance). Repeated a couple of times
        # so the alerts are reliably captured.
        intrusion = (
            "sh -c 'id; head -c 128 /etc/shadow; cat /etc/shadow > /dev/null 2>&1; "
            "ls /host/etc > /dev/null 2>&1'"
        )
        for _ in range(3):
            cmd = (
                f"kubectl exec {pod} -n {self.namespace} -c {container_name} "
                f"-- {intrusion}"
            )
            result = self.kubectl.exec_command(cmd)
            print(f"Intrusion step on {pod}: {result.strip()[:200]}")
            time.sleep(2)

    def _remove_privileged(self, service: str):
        """Revert the deployment to a compliant, non-privileged posture."""
        deployment = self.kubectl.get_deployment(service, self.namespace)
        if not deployment:
            return

        container_name = self.container_name_fmt.format(service=service)
        spec = deployment.spec.template.spec

        for container in spec.containers:
            if container.name == container_name:
                container.security_context = None
                if container.volume_mounts:
                    container.volume_mounts = [
                        m for m in container.volume_mounts if m.name != HOST_VOLUME_NAME
                    ]

        if spec.volumes:
            spec.volumes = [v for v in spec.volumes if v.name != HOST_VOLUME_NAME]

        self.kubectl.update_deployment(service, self.namespace, deployment)
        self._wait_rollout(service)
        print(f"Reverted {service} to a non-privileged posture.")

    # --- helpers ---------------------------------------------------------------

    def _get_pod_name(self, service: str) -> str | None:
        pods = self.kubectl.list_pods(self.namespace)
        for pod in pods.items:
            phase = getattr(pod.status, "phase", None)
            if pod.metadata.name.startswith(service) and phase == "Running":
                return pod.metadata.name
        return None

    def _wait_rollout(self, service: str, timeout: str = "120s"):
        self.kubectl.exec_command(
            f"kubectl rollout status deployment/{service} "
            f"-n {self.namespace} --timeout={timeout}"
        )


if __name__ == "__main__":
    namespace = "test-hotel-reservation"
    microservices = ["geo"]
    injector = SecurityFaultInjector(namespace)
    injector._inject("priv_escalation", microservices)
    # injector._recover("priv_escalation", microservices)
