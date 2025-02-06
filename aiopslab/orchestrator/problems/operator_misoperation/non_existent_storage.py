"""
This fault specifies a non-existent storage class.
"""
from typing import Any

from aiopslab.orchestrator.tasks import *
from aiopslab.orchestrator.evaluators.quantitative import *
from aiopslab.generators.fault.inject_operator import K8SOperatorFaultInjector
from aiopslab.service.apps.tidb_cluster_operator import TiDBCluster
from aiopslab.session import SessionItem
from datetime import datetime, timedelta
import time

class K8SOperatorNonExistentStorageBaseTask:
    def __init__(self):
        self.injector = K8SOperatorFaultInjector("tidb-cluster")
        self.app = TiDBCluster()
        self.faulty_cr = "tidbclusters"

    def start_workload(self):
        print("== Start Workload ==")
        print("Workload is the CR applied to the operator.")
        pass

    def inject_fault(self):
        print("== Fault Injection ==")
        self.injector._inject("non_existent_storage")
        print(f"Injecting non-existent storage failure of the TiDB cluster\n")

    def recover_fault(self):
        print("== Fault Recovery ==")
        self.injector._recover("non_existent_storage")
        print(f"Recovered non-existent failure of the TiDB cluster\n")

################## Detection Problem ##################
class K8SOperatorNonExistentStorageDetection(
    K8SOperatorNonExistentStorageBaseTask, DetectionTask
):
    def __init__(self):
        K8SOperatorNonExistentStorageBaseTask.__init__(self)
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
class K8SOperatorNonExistentStorageLocalization(
    K8SOperatorNonExistentStorageBaseTask, LocalizationTask
):
    def __init__(self):
        K8SOperatorNonExistentStorageBaseTask.__init__(self)
        LocalizationTask.__init__(self, self.app)
        self.task_desc = """\
            You are an expert DevOps engineer who has been tasked with localizing faults in a deployed service.

            The service you are working with today is described below:
            {app_summary}

            You will begin by analyzing the service's state and telemetry, and then submit one of two possible solutions:
            1. list[str]: list of faulty components or custom resources (e.g., service names, CRs)
            2. str: `None` if no faults were detected
            """

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
        is_exact = is_exact_match(soln, self.faulty_cr)
        is_sub = is_subset([self.faulty_cr], soln)

        # Determine accuracy
        if is_exact:
            accuracy = 100.0
            print(f"Exact match: {soln} | Accuracy: {accuracy}%")
        elif is_sub:
            accuracy = (len([self.faulty_cr]) / len(soln)) * 100.0
            print(f"Subset match: {soln} | Accuracy: {accuracy:.2f}%")
        else:
            accuracy = 0.0
            print(f"No match: {soln} | Accuracy: {accuracy}%")

        self.add_result("Localization Accuracy", accuracy)
        super().eval(soln, trace, duration)

        self.results["success"] = is_exact or (is_sub and len(soln) == 1)
        self.results["is_subset"] = is_sub

        return self.results
