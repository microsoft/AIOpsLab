"""Network delay problem in the HotelReservation application."""

from typing import Any

from aiopslab.orchestrator.tasks import *
from aiopslab.orchestrator.evaluators.quantitative import *
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.hotelres import HotelReservation
from aiopslab.generators.workload.wrk import Wrk
from aiopslab.generators.fault.inject_symp import SymptomFaultInjector
from aiopslab.session import SessionItem
from aiopslab.paths import TARGET_MICROSERVICES

from .helpers import get_frontend_url


class NetworkDelayBaseTask:
    def __init__(self):
        self.app = HotelReservation()
        self.kubectl = KubeCtl()
        self.namespace = self.app.namespace
        self.faulty_service = "user"
        self.payload_script = (
            TARGET_MICROSERVICES
            / "hotelReservation/wrk2/scripts/hotel-reservation/mixed-workload_type_1.lua"
        )
        self.injector = SymptomFaultInjector(namespace=self.namespace)

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
        self.injector.inject_network_delay([self.faulty_service])
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    def recover_fault(self):
        print("== Fault Recovery ==")
        self.injector.recover_network_delay()


def evaluate_network_delay_analysis_solution(task, soln: Any) -> dict:
    """Validate the RCA response for the static network delay scenario."""

    if not isinstance(soln, dict):
        print("Solution is not a dictionary")
        task.results["system_level_correct"] = False
        task.results["fault_type_correct"] = False
        task.results["success"] = False
        return task.results

    system_level = soln.get("system_level", "")
    fault_type = soln.get("fault_type", "")

    is_sys_level_correct = is_exact_match_lower(system_level, "Application")
    is_fault_type_correct = is_exact_match_lower(fault_type, "Network/Storage Issue")

    if is_sys_level_correct:
        print("System level analysis correct: Application")
    else:
        print(f"Incorrect system level analysis: {system_level}")

    if is_fault_type_correct:
        print("Fault type analysis correct: Network/Storage Issue")
    else:
        print(f"Incorrect fault type analysis: {fault_type}")

    task.results["system_level_correct"] = is_sys_level_correct
    task.results["fault_type_correct"] = is_fault_type_correct
    task.results["success"] = is_sys_level_correct and is_fault_type_correct

    return task.results


def evaluate_network_delay_mitigation(task) -> bool:
    """Check that the delayed service's pods are Ready before passing mitigation."""

    pods_ready = task._check_pods_ready(
        {
            "namespace": task.namespace,
            "services": [task.faulty_service],
            "timeout": 60,
            "interval": 5,
        }
    )
    print(
        "Mitigation pods_ready check for service"
        f" '{task.faulty_service}' returned: {pods_ready}"
    )

    task.add_result("mitigation_pods_ready", pods_ready)

    previous_success = task.results.get("success")
    task.results["success"] = pods_ready if previous_success is None else (
        previous_success and pods_ready
    )

    return task.results["success"]


################## Detection Problem ##################
class NetworkDelayDetection(NetworkDelayBaseTask, DetectionTask):
    def __init__(self):
        NetworkDelayBaseTask.__init__(self)
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
class NetworkDelayLocalization(NetworkDelayBaseTask, LocalizationTask):
    def __init__(self):
        NetworkDelayBaseTask.__init__(self)
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
class NetworkDelayAnalysis(NetworkDelayBaseTask, AnalysisTask):
    def __init__(self):
        NetworkDelayBaseTask.__init__(self)
        AnalysisTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")

        evaluate_network_delay_analysis_solution(self, soln)
        super().eval(soln, trace, duration)

        return self.results


################## Mitigation Problem ##################
class NetworkDelayMitigation(NetworkDelayBaseTask, MitigationTask):
    def __init__(self):
        NetworkDelayBaseTask.__init__(self)
        MitigationTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float) -> dict:
        print("== Evaluation ==")

        super().eval(soln, trace, duration)
        evaluate_network_delay_mitigation(self)

        return self.results
