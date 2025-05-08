# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""K8S misconfig fault problem in the SocialNetwork application."""

from typing import Any

from aiopslab.orchestrator.tasks import *
from aiopslab.orchestrator.evaluators.quantitative import *
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.socialnet import SocialNetwork
from aiopslab.generators.workload.wrk import Wrk
from aiopslab.generators.fault.inject_virtual import VirtualizationFaultInjector
from aiopslab.session import SessionItem
from aiopslab.paths import TARGET_MICROSERVICES

from .helpers import get_frontend_url


class K8STargetPortMisconfigBaseTask:
    def __init__(self, faulty_service: str = "user-service"):
        self.app = SocialNetwork()
        self.kubectl = KubeCtl()
        self.namespace = self.app.namespace
        self.faulty_service = faulty_service
        self.payload_script = (
            TARGET_MICROSERVICES
            / "socialNetwork/wrk2/scripts/social-network/compose-post.lua"
        )

    def start_workload(self):
        print("== Start Workload ==")
        frontend_url = get_frontend_url(self.app)

        wrk = Wrk(rate=10, dist="exp", connections=2, duration=10, threads=2)
        wrk.start_workload(
            payload_script=self.payload_script,
            url=f"{frontend_url}/wrk2-api/post/compose",
        )

    def inject_fault(self):
        print("== Fault Injection ==")
        injector = VirtualizationFaultInjector(namespace=self.namespace)
        injector._inject(
            fault_type="misconfig_k8s",
            microservices=[self.faulty_service],
        )
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    def recover_fault(self):
        print("== Fault Recovery ==")
        injector = VirtualizationFaultInjector(namespace=self.namespace)
        injector._recover(
            fault_type="misconfig_k8s",
            microservices=[self.faulty_service],
        )
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")


################## Detection Problem ##################
class K8STargetPortMisconfigDetection(K8STargetPortMisconfigBaseTask, DetectionTask):
    def __init__(self, faulty_service: str = "user-service"):
        K8STargetPortMisconfigBaseTask.__init__(self, faulty_service=faulty_service)
        DetectionTask.__init__(self, self.app)

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
class K8STargetPortMisconfigLocalization(
    K8STargetPortMisconfigBaseTask, LocalizationTask
):
    def __init__(self, faulty_service: str = "user-service"):
        K8STargetPortMisconfigBaseTask.__init__(self, faulty_service=faulty_service)
        LocalizationTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")

        if soln is None:
            print("Solution is None")
            self.add_result("Localization Accuracy", 0.0)
            self.results["success"] = False
            self.results["is_subset"] = False
            super().eval(soln, trace, duration)
            return self.results

        # Calculate exact match and subset
        is_exact = is_exact_match(soln, self.faulty_service)
        is_sub = is_subset([self.faulty_service], soln)

        # Determine accuracy
        if is_exact:
            accuracy = 100.0
            print(f"Exact match: {soln} | Accuracy: {accuracy}%")
        elif is_sub:
            accuracy = (len([self.faulty_service]) / len(soln)) * 100.0
            print(f"Subset match: {soln} | Accuracy: {accuracy:.2f}%")
        else:
            accuracy = 0.0
            print(f"No match: {soln} | Accuracy: {accuracy}%")

        # Add result to the task
        self.add_result("Localization Accuracy", accuracy)

        # Continue with the base evaluation logic
        super().eval(soln, trace, duration)

        self.results["success"] = is_exact or (is_sub and len(soln) == 1)
        self.results["is_subset"] = is_sub

        return self.results


################## Root cause analysis Problem ##################
class K8STargetPortMisconfigAnalysis(K8STargetPortMisconfigBaseTask, AnalysisTask):
    def __init__(self, faulty_service: str = "user-service"):
        K8STargetPortMisconfigBaseTask.__init__(self, faulty_service=faulty_service)
        AnalysisTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")

        if not isinstance(soln, dict):
            print("Solution is not a dictionary")
            self.results["system_level_correct"] = False
            self.results["fault_type_correct"] = False
            self.results["success"] = False
            super().eval(soln, trace, duration)
            return self.results

        is_sys_level_correct = is_exact_match_lower(
            soln.get("system_level", ""), "Virtualization"
        )
        is_fault_type_correct = is_exact_match_lower(
            soln.get("fault_type", ""), "Misconfiguration"
        )

        self.results["system_level_correct"] = is_sys_level_correct
        self.results["fault_type_correct"] = is_fault_type_correct
        self.results["success"] = is_sys_level_correct and is_fault_type_correct

        super().eval(soln, trace, duration)

        return self.results


################## Mitigation Problem ##################
class K8STargetPortMisconfigMitigation(K8STargetPortMisconfigBaseTask, MitigationTask):
    def __init__(self, faulty_service: str = "user-service"):
        K8STargetPortMisconfigBaseTask.__init__(self, faulty_service=faulty_service)
        MitigationTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float) -> dict:
        print("== Evaluation ==")
        super().eval(soln, trace, duration)

        # custom: check if target port has been reset to 9090
        configs = self.kubectl.get_service_json(self.faulty_service, self.namespace)
        target_port = configs["spec"]["ports"][0]["targetPort"]
        all_normal = is_exact_match(target_port, 9090)

        if all_normal:
            # Check if all services (not only faulty service) is back to normal (Running)
            pod_list = self.kubectl.list_pods(self.namespace)
            for pod in pod_list.items:
                if pod.status.container_statuses:
                    # Check container statuses
                    for container_status in pod.status.container_statuses:
                        if (
                            container_status.state.waiting
                            and container_status.state.waiting.reason
                            == "CrashLoopBackOff"
                        ):
                            print(
                                f"Container {container_status.name} is in CrashLoopBackOff"
                            )
                            all_normal = False
                        elif (
                            container_status.state.terminated
                            and container_status.state.terminated.reason != "Completed"
                        ):
                            print(
                                f"Container {container_status.name} is terminated with reason: {container_status.state.terminated.reason}"
                            )
                            all_normal = False
                        elif not container_status.ready:
                            print(f"Container {container_status.name} is not ready")
                            all_normal = False

                if not all_normal:
                    break

        self.results["success"] = all_normal
        return self.results
