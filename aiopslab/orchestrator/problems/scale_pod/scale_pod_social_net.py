# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Scale pod replica to zero problem for the SocialNetwork application."""

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


class ScalePodSocialNetBaseTask:
    def __init__(self):
        self.app = SocialNetwork()
        self.kubectl = KubeCtl()
        self.namespace = self.app.namespace
        # self.faulty_service = "url-shorten-mongodb"
        self.faulty_service = "user-service"
        # Choose a very front service to test - this will directly cause an exception
        # self.faulty_service = "nginx-thrift"
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
            fault_type="scale_pods_to_zero",
            microservices=[self.faulty_service],
        )
        # Terminating the pod may take long time when scaling
        time.sleep(30)
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    def recover_fault(self):
        print("== Fault Recovery ==")
        injector = VirtualizationFaultInjector(namespace=self.namespace)
        injector._recover(
            fault_type="scale_pods_to_zero",
            microservices=[self.faulty_service],
        )
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")


################## Detection Problem ##################
class ScalePodSocialNetDetection(ScalePodSocialNetBaseTask, DetectionTask):
    def __init__(self):
        ScalePodSocialNetBaseTask.__init__(self)
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
class ScalePodSocialNetLocalization(ScalePodSocialNetBaseTask, LocalizationTask):
    def __init__(self):
        ScalePodSocialNetBaseTask.__init__(self)
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
class ScalePodSocialNetAnalysis(ScalePodSocialNetBaseTask, AnalysisTask):
    def __init__(self):
        ScalePodSocialNetBaseTask.__init__(self)
        AnalysisTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")

        # Ensure soln is a dictionary
        if isinstance(soln, dict):
            # Expected solution
            expected_system_level = "Virtualization"
            expected_fault_type = "Operation Error"

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
            print(
                "Error: soln is not a dictionary. Please make it a dictionary with keys: 'system_level' and 'fault_type'."
            )
            self.results["system_level_correct"] = False
            self.results["fault_type_correct"] = False
            self.results["success"] = False

        super().eval(soln, trace, duration)

        return self.results


################## Mitigation Problem ##################
class ScalePodSocialNetMitigation(ScalePodSocialNetBaseTask, MitigationTask):
    def __init__(self):
        ScalePodSocialNetBaseTask.__init__(self)
        MitigationTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float) -> dict:
        print("== Evaluation ==")
        super().eval(soln, trace, duration)

        all_normal = True

        deployment = self.kubectl.get_deployment(self.faulty_service, self.namespace)

        if deployment is None:
            print(f"Deployment for {self.faulty_service} not found")
            all_normal = False
        else:
            desired_replicas = deployment.spec.replicas
            available_replicas = deployment.status.available_replicas

            if desired_replicas != 1 or available_replicas != 1:
                print(
                    f"Faulty service {self.faulty_service} has {available_replicas} available replicas out of {desired_replicas} desired"
                )
                all_normal = False

        # Check if all services are running normally
        pod_list = self.kubectl.list_pods(self.namespace)
        for pod in pod_list.items:
            for container_status in pod.status.container_statuses:
                if (
                    container_status.state.waiting
                    and container_status.state.waiting.reason == "CrashLoopBackOff"
                ):
                    print(f"Container {container_status.name} is in CrashLoopBackOff")
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
