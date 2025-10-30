# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Network loss problem in the HotelReservation application."""

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


class NetworkLossBaseTask:
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
            fault_type="network_loss",
            microservices=[self.faulty_service],
            duration="200s"
        )
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    def recover_fault(self):
        print("== Fault Recovery ==")
        self.injector._recover(
            fault_type="network_loss",
        )

################## Detection Problem ##################
class NetworkLossDetection(NetworkLossBaseTask, DetectionTask):
    def __init__(self):
        NetworkLossBaseTask.__init__(self)
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
class NetworkLossLocalization(
    NetworkLossBaseTask, LocalizationTask
):
    def __init__(self):
        NetworkLossBaseTask.__init__(self)
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
class NetworkLossAnalysis(NetworkLossBaseTask, AnalysisTask):
    def __init__(self):
        NetworkLossBaseTask.__init__(self)
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
class NetworkLossMitigation(NetworkLossBaseTask, MitigationTask):
    def __init__(self):
        NetworkLossBaseTask.__init__(self)
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
                f"Pods for service '{self.faulty_service}' in namespace '{self.namespace}' are ready."
            )
        else:
            print(
                f"Pods for service '{self.faulty_service}' in namespace '{self.namespace}' are not ready."
            )

        self.add_result("mitigation_pods_ready", pods_ready)
        self.results["success"] = pods_ready

        return self.results

