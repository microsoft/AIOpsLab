"""Otel demo kafkaQueueProblems feature flag fault."""

from typing import Any

from aiopslab.orchestrator.tasks import *
from aiopslab.orchestrator.evaluators.quantitative import *
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.astronomy_shop import AstronomyShop
from aiopslab.generators.fault.inject_otel import OtelFaultInjector
from aiopslab.session import SessionItem


class KafkaQueueProblemsBaseTask:
    def __init__(self):
        self.app = AstronomyShop()
        self.kubectl = KubeCtl()
        self.namespace = self.app.namespace
        self.injector = OtelFaultInjector(namespace=self.namespace)
        self.faulty_service = "kafka"

    def start_workload(self):
        print("== Start Workload ==")
        print("Workload skipped since AstronomyShop has a built-in load generator.")

    def inject_fault(self):
        print("== Fault Injection ==")
        self.injector.inject_fault("kafkaQueueProblems")
        print(f"Fault: kafkaQueueProblems | Namespace: {self.namespace}\n")

    def recover_fault(self):
        print("== Fault Recovery ==")
        self.injector.recover_fault("kafkaQueueProblems")


################## Detection Problem ##################
class KafkaQueueProblemsDetection(KafkaQueueProblemsBaseTask, DetectionTask):
    def __init__(self):
        KafkaQueueProblemsBaseTask.__init__(self)
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
class KafkaQueueProblemsLocalization(KafkaQueueProblemsBaseTask, LocalizationTask):
    def __init__(self):
        KafkaQueueProblemsBaseTask.__init__(self)
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


################## Mitigation Problem ##################
class KafkaQueueProblemsMitigation(KafkaQueueProblemsBaseTask, MitigationTask):
    def __init__(self):
        KafkaQueueProblemsBaseTask.__init__(self)
        MitigationTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float) -> dict:
        print("== Evaluation ==")
        super().eval(soln, trace, duration)

        # Check if all services (not only faulty service) is back to normal (Running)
        pod_list = self.kubectl.list_pods(self.namespace)
        all_normal = True

        for pod in pod_list.items:
            if pod.status.container_statuses:
                # Check container statuses
                for container_status in pod.status.container_statuses:
                    if (
                        container_status.state.waiting
                        and container_status.state.waiting.reason == "CrashLoopBackOff"
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