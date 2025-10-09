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


class PodKillBaseTask:
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
        self.injector._inject(
            fault_type="pod_kill", microservices=[self.faulty_service], duration="100s"
        )
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    def recover_fault(self):
        print("== Fault Recovery ==")
        self.injector._recover(
            fault_type="pod_kill",
        )


################## Detection Problem ##################
class PodKillDetection(PodKillBaseTask, DetectionTask):
    def __init__(self):
        PodKillBaseTask.__init__(self)
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
class PodKillLocalization(PodKillBaseTask, LocalizationTask):
    def __init__(self):
        PodKillBaseTask.__init__(self)
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


################## Root Cause Analysis Problem ##################
class PodKillAnalysis(PodKillBaseTask, AnalysisTask):
    def __init__(self):
        PodKillBaseTask.__init__(self)
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

        system_level_correct = is_exact_match_lower(
            soln.get("system_level", ""), "Virtualization"
        )
        fault_type_correct = is_exact_match_lower(
            soln.get("fault_type", ""), "Operation Error"
        )

        self.results["system_level_correct"] = system_level_correct
        self.results["fault_type_correct"] = fault_type_correct
        self.results["success"] = system_level_correct and fault_type_correct

        super().eval(soln, trace, duration)

        return self.results


################## Mitigation Problem ##################
class PodKillMitigation(PodKillBaseTask, MitigationTask):
    def __init__(self):
        PodKillBaseTask.__init__(self)
        MitigationTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")

        super().eval(soln, trace, duration)

        pods_ready = self._check_pods_ready(
            {
                "namespace": self.namespace,
                "services": [self.faulty_service],
            }
        )

        if pods_ready:
            print(
                f"All pods for service '{self.faulty_service}' are back to Ready state."
            )
        else:
            print(
                f"Pods for service '{self.faulty_service}' are not all Ready after mitigation."
            )

        self.add_result("mitigation_pods_ready", pods_ready)
        self.results["success"] = pods_ready

        return self.results
