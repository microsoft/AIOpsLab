# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Assign pods to non existent node problem for the SocialNetwork application."""

from typing import Any
import time

from aiopslab.orchestrator.tasks import *
from aiopslab.orchestrator.evaluators.quantitative import is_exact_match, is_subset
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.socialnet import SocialNetwork
from aiopslab.generators.workload.wrk import Wrk
from aiopslab.generators.fault.inject_virtual import VirtualizationFaultInjector
from aiopslab.session import SessionItem
from aiopslab.paths import TARGET_MICROSERVICES

from .helpers import get_frontend_url


class AssignNonExistentNodeSocialNetBaseTask:
    def __init__(self):
        self.app = SocialNetwork()
        self.kubectl = KubeCtl()
        self.namespace = self.app.namespace
        self.faulty_service = "user-service"

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
            fault_type="assign_to_non_existent_node",
            microservices=[self.faulty_service],
        )
        time.sleep(25)
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    def recover_fault(self):
        print("== Fault Recovery ==")
        injector = VirtualizationFaultInjector(namespace=self.namespace)
        injector._recover(
            fault_type="assign_to_non_existent_node",
            microservices=[self.faulty_service],
        )
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")


################## Detection Problem ##################
class AssignNonExistentNodeSocialNetDetection(
    AssignNonExistentNodeSocialNetBaseTask, DetectionTask
):
    def __init__(self):
        AssignNonExistentNodeSocialNetBaseTask.__init__(self)
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
class AssignNonExistentNodeSocialNetLocalization(
    AssignNonExistentNodeSocialNetBaseTask, LocalizationTask
):
    def __init__(self):
        AssignNonExistentNodeSocialNetBaseTask.__init__(self)
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

        self.add_result("Localization Accuracy", accuracy)
        super().eval(soln, trace, duration)

        self.results["success"] = is_exact or (is_sub and len(soln) == 1)
        self.results["is_subset"] = is_sub

        return self.results


################## Analysis Problem ##################
class AssignNonExistentNodeSocialNetAnalysis(
    AssignNonExistentNodeSocialNetBaseTask, AnalysisTask
):
    def __init__(self):
        AssignNonExistentNodeSocialNetBaseTask.__init__(self)
        AnalysisTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")
        # Ensure soln is a dictionary
        if isinstance(soln, dict):
            # Expected solution
            expected_system_level = "Virtualization"
            expected_fault_type = "Dependency Problem"

            provided_system_level = soln.get("system_level", "").strip().lower()
            provided_fault_type = soln.get("fault_type", "").strip().lower()

            is_system_level_correct = (
                provided_system_level == expected_system_level.lower()
            )
            is_fault_type_correct = provided_fault_type == expected_fault_type.lower()

            self.results["system_level_correct"] = is_system_level_correct
            self.results["fault_type_correct"] = is_fault_type_correct
            self.results["success"] = is_system_level_correct and is_fault_type_correct
        else:
            self.results["system_level_correct"] = False
            self.results["fault_type_correct"] = False
            self.results["success"] = False

            print(
                "Solution is not a dictionary. Should make it a dictionary with keys: 'system_level' and 'fault_type'."
            )

        super().eval(soln, trace, duration)

        return self.results


################## Mitigation Problem ##################
class AssignNonExistentNodeSocialNetMitigation(
    AssignNonExistentNodeSocialNetBaseTask, MitigationTask
):
    def __init__(self):
        AssignNonExistentNodeSocialNetBaseTask.__init__(self)
        MitigationTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float) -> dict:
        print("== Evaluation ==")
        super().eval(soln, trace, duration)

        # Check if all services (not only faulty service) is back to normal (Running)
        pod_list = self.kubectl.list_pods(self.namespace)
        all_normal = True

        # Check if the faulty service exists
        faulty_service_exists = any(
            self.faulty_service in pod.metadata.name for pod in pod_list.items
        )
        if not faulty_service_exists:
            print(f"Pod named {self.faulty_service} does not exist.")
            all_normal = False
        else:
            for pod in pod_list.items:
                if pod.status.container_statuses:
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
