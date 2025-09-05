"""Wrong binary usage problem in the HotelReservation application."""

from typing import Any

from aiopslab.orchestrator.tasks import *
from aiopslab.orchestrator.evaluators.quantitative import *
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.hotelres import HotelReservation
from aiopslab.generators.workload.wrk import Wrk
from aiopslab.generators.fault.inject_virtual import VirtualizationFaultInjector
from aiopslab.session import SessionItem
from aiopslab.paths import TARGET_MICROSERVICES

from .helpers import get_frontend_url


class WrongBinUsageBaseTask:
    def __init__(self, faulty_service: str = "profile"):
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
        injector = VirtualizationFaultInjector(namespace=self.namespace)
        injector._inject(
            fault_type="wrong_bin_usage",
            microservices=[self.faulty_service],
        )
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    def recover_fault(self):
        print("== Fault Recovery ==")
        injector = VirtualizationFaultInjector(namespace=self.namespace)
        injector._recover(
            fault_type="wrong_bin_usage",
            microservices=[self.faulty_service],
        )
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")


################## Detection Problem ##################
class WrongBinUsageDetection(WrongBinUsageBaseTask, DetectionTask):
    def __init__(self, faulty_service: str = "profile"):
        WrongBinUsageBaseTask.__init__(self, faulty_service=faulty_service)
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
class WrongBinUsageLocalization(WrongBinUsageBaseTask, LocalizationTask):
    def __init__(self, faulty_service: str = "profile"):
        WrongBinUsageBaseTask.__init__(self, faulty_service=faulty_service)
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


################## Root cause analysis Problem ##################
class WrongBinUsageAnalysis(WrongBinUsageBaseTask, AnalysisTask):
    def __init__(self, faulty_service: str = "profile"):
        WrongBinUsageBaseTask.__init__(self, faulty_service=faulty_service)
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
            soln.get("system_level", ""), "Application"
        )
        is_fault_type_correct = is_exact_match_lower(
            soln.get("fault_type", ""), "Network/Storage Issue"
        )

        self.results["system_level_correct"] = is_sys_level_correct
        self.results["fault_type_correct"] = is_fault_type_correct
        self.results["success"] = is_sys_level_correct and is_fault_type_correct

        super().eval(soln, trace, duration)

        return self.results


################## Mitigation Problem ##################
class WrongBinUsageMitigation(WrongBinUsageBaseTask, MitigationTask):
    def __init__(self, faulty_service: str = "profile"):
        WrongBinUsageBaseTask.__init__(self, faulty_service=faulty_service)
        MitigationTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float) -> dict:
        print("== Evaluation ==")
        super().eval(soln, trace, duration)

        # Check if all services (not only faulty service) is back to normal (Running)
        pod_list = self.kubectl.list_pods(self.namespace)
        all_normal = True

        for pod in pod_list.items:
            # Check container statuses
            for container_status in pod.status.container_statuses:
                if container_status.state.waiting:
                    reason = container_status.state.waiting.reason
                    if reason in ["CrashLoopBackOff", "Error", "ImagePullBackOff", "ErrImagePull"]:
                        print(f"Container {container_status.name} is in error state: {reason}")
                        all_normal = False
                elif container_status.state.terminated and container_status.state.terminated.reason != "Completed":
                    print(f"Container {container_status.name} is terminated with reason: {container_status.state.terminated.reason}")
                    all_normal = False
                elif not container_status.ready:
                    print(f"Container {container_status.name} is not ready")
                    all_normal = False

            if not all_normal:
                print("Pods are not all in a good state.")
                self.results["success"] = False
                return self.results

        # Check if the deployment was updated to use the right binary
        expected_command = "profile" # Command dictates which binary will be ran, we want to run /go/bin/profile and not /go/bin/geo

        try:
            deployment = self.kubectl.get_deployment(self.faulty_service, self.namespace)
            containers = deployment.spec.template.spec.containers

            for container in containers:
                command = container.command or []
                if expected_command not in command:
                    print(
                        f"[FAIL] Deployment for container '{container.name}' is using wrong binary: {command}"
                    )
                    self.results["success"] = False
                    return self.results

            print("[PASS] Deployment is using the correct binary.")
            self.results["success"] = True
            return self.results

        except Exception as e:
            print(f"[ERROR] Exception during evaluation: {e}")
            self.results["success"] = False
            return self.results
