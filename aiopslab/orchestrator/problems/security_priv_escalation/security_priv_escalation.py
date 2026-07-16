# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Privileged-pod compromise security problem in the HotelReservation application.

Scenario (canonical Falco/OPA demo, turned into a benchmark):
  A workload (`geo`) is running privileged with the node's root filesystem
  mounted, and a runtime intrusion (interactive shell + reading /etc/shadow) has
  been executed inside it. The application stays healthy and serves traffic, so
  ordinary health/latency telemetry looks fine. Two independent security signals
  reveal the incident:
    - OPA Gatekeeper (audit)  -> the pod violates the "no privileged container"
                                 and "no hostPath volume" policies.
    - Falco (runtime)         -> alerts for shell-in-container and reading a
                                 sensitive file.

  There is no one-shot answer: the agent must correlate a static
  misconfiguration with a runtime alert to conclude the workload is compromised.
"""

from typing import Any

from aiopslab.orchestrator.tasks import *
from aiopslab.orchestrator.evaluators.quantitative import *
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.hotelres import HotelReservation
from aiopslab.generators.workload.wrk import Wrk
from aiopslab.generators.fault.inject_security import (
    SecurityFaultInjector,
    HOST_VOLUME_NAME,
)
from aiopslab.session import SessionItem
from aiopslab.paths import TARGET_MICROSERVICES

from .helpers import get_frontend_url

SECURITY_TASK_HINT = """\

    This is a SECURITY investigation. The application may look healthy while
    still being compromised. Two security tools are deployed in the cluster to
    help you, in addition to the standard telemetry APIs:
      - Falco (runtime security) runs as a DaemonSet in the `falco` namespace.
        Its alerts are emitted to the falco pods' logs, e.g.:
        `kubectl logs -n falco -l app.kubernetes.io/name=falco -c falco`
      - OPA Gatekeeper (policy audit) runs in the `gatekeeper-system` namespace.
        Policy violations on running workloads can be inspected with, e.g.:
        `kubectl get k8spspprivilegedcontainer.constraints.gatekeeper.sh -o yaml`
        `kubectl get k8spsphostfilesystem.constraints.gatekeeper.sh -o yaml`
    Correlate configuration (privilege/host-mount) signals with runtime (Falco)
    signals to reach your conclusion.
    """


class SecurityPrivEscalationBaseTask:
    def __init__(self, faulty_service: str = "geo"):
        self.app = HotelReservation()
        self.kubectl = KubeCtl()
        self.namespace = self.app.namespace
        self.faulty_service = faulty_service
        self.payload_script = (
            TARGET_MICROSERVICES
            / "hotelReservation/wrk2/scripts/hotel-reservation/mixed-workload_type_1.lua"
        )

    def start_workload(self):
        print("== Start Workload ==")
        frontend_url = get_frontend_url(self.app)

        wrk = Wrk(rate=10, dist="exp", connections=2, duration=10, threads=2)
        wrk.start_workload(
            payload_script=self.payload_script,
            url=f"{frontend_url}",
        )

    def inject_fault(self):
        print("== Fault Injection ==")
        injector = SecurityFaultInjector(namespace=self.namespace)
        injector._inject(
            fault_type="priv_escalation",
            microservices=[self.faulty_service],
        )
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    def recover_fault(self):
        print("== Fault Recovery ==")
        injector = SecurityFaultInjector(namespace=self.namespace)
        injector._recover(
            fault_type="priv_escalation",
            microservices=[self.faulty_service],
        )
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    # --- shared security-state check ------------------------------------------
    def is_compromised(self) -> bool:
        """True if the faulty service still runs privileged or mounts the host FS."""
        deployment = self.kubectl.get_deployment(self.faulty_service, self.namespace)
        if not deployment:
            return False

        spec = deployment.spec.template.spec
        for container in spec.containers:
            sc = container.security_context
            if sc and (sc.privileged or sc.run_as_user == 0):
                return True

        for volume in spec.volumes or []:
            if volume.host_path is not None or volume.name == HOST_VOLUME_NAME:
                return True

        return False


################## Detection Problem ##################
class SecurityPrivEscalationDetection(
    SecurityPrivEscalationBaseTask, DetectionTask
):
    def __init__(self, faulty_service: str = "geo"):
        SecurityPrivEscalationBaseTask.__init__(self, faulty_service=faulty_service)
        DetectionTask.__init__(self, self.app)
        self.task_desc += SECURITY_TASK_HINT

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")
        expected_solution = "Yes"

        if isinstance(soln, str):
            if soln.strip().lower() == expected_solution.lower():
                print(f"Correct detection: {soln}")
                self.add_result("Detection Accuracy", "Correct")
            else:
                print(f"Incorrect detection: {soln}")
                self.add_result("Detection Accuracy", "Incorrect")
        else:
            print("Invalid solution format")
            self.add_result("Detection Accuracy", "Invalid Format")

        return super().eval(soln, trace, duration)


################## Localization Problem ##################
class SecurityPrivEscalationLocalization(
    SecurityPrivEscalationBaseTask, LocalizationTask
):
    def __init__(self, faulty_service: str = "geo"):
        SecurityPrivEscalationBaseTask.__init__(self, faulty_service=faulty_service)
        LocalizationTask.__init__(self, self.app)
        self.task_desc += SECURITY_TASK_HINT

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")

        if soln is None:
            print("Solution is None")
            self.add_result("Localization Accuracy", 0.0)
            self.results["success"] = False
            self.results["is_subset"] = False
            super().eval(soln, trace, duration)
            return self.results

        is_exact = is_exact_match(soln, self.faulty_service)
        is_sub = is_subset([self.faulty_service], soln)

        if is_exact:
            accuracy = 100.0
            print(f"Exact match: {soln} | Accuracy: {accuracy}%")
        elif is_sub:
            accuracy = (len([self.faulty_service]) / len(soln)) * 100.0
            print(f"Subset match: {soln} | Accuracy: {accuracy:.2f}%")
        else:
            accuracy = 0.0
            print(f"No match: {soln} | Accuracy: {accuracy}%")

        self.add_result("Localization Accuracy", accuracy)
        super().eval(soln, trace, duration)

        self.results["success"] = is_exact or (is_sub and len(soln) == 1)
        self.results["is_subset"] = is_sub

        return self.results


################## Mitigation Problem ##################
class SecurityPrivEscalationMitigation(
    SecurityPrivEscalationBaseTask, MitigationTask
):
    def __init__(self, faulty_service: str = "geo"):
        SecurityPrivEscalationBaseTask.__init__(self, faulty_service=faulty_service)
        MitigationTask.__init__(self, self.app)
        self.task_desc += SECURITY_TASK_HINT

    def eval(self, soln: Any, trace: list[SessionItem], duration: float) -> dict:
        print("== Evaluation ==")
        super().eval(soln, trace, duration)

        # A correct mitigation must (a) remove the insecure posture (no longer
        # privileged / no host mount) AND (b) keep the app healthy. Note this is
        # deliberately stricter than the usual "pods are Running" check, since the
        # compromised pod was Running the whole time.
        still_compromised = self.is_compromised()
        if still_compromised:
            print(f"{self.faulty_service} is still running with an insecure posture.")

        try:
            self.kubectl.wait_for_ready(self.namespace, sleep=5, max_wait=60)
            healthy = True
        except Exception as e:
            print(f"Pods are not all ready: {e}")
            healthy = False

        self.results["security_posture_fixed"] = not still_compromised
        self.results["app_healthy"] = healthy
        self.results["success"] = (not still_compromised) and healthy

        return self.results
