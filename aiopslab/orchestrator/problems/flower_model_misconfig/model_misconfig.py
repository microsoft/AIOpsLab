# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model misconfiguration fault in the Flower application."""

import time
from typing import Any

from aiopslab.orchestrator.tasks import *
from aiopslab.service.dock import Docker
from aiopslab.service.apps.flower import Flower
from aiopslab.paths import TARGET_MICROSERVICES
from aiopslab.session import SessionItem
from aiopslab.generators.fault.inject_virtual import VirtualizationFaultInjector


class FlowerModelMisconfigBaseTask:
    def __init__(self, faulty_service: str = "user-service"):
        self.app = Flower()
        self.docker = Docker()
        self.namespace = self.app.namespace
        self.faulty_service = faulty_service
        self.train_dir = TARGET_MICROSERVICES / "flower"

    def start_workload(self):
        print("== Start Workload ==")
        command = "flwr run train local-deployment"
        self.docker.exec_command(command, cwd=self.train_dir)
        
        print("Waiting for workload to start...")
        time.sleep(10)
        
        print("Injecting fault again...")
        self.inject_fault()
        
        print("Waiting for faults to propagate...")
        time.sleep(60)
        print("Faults propagated.")
        
    def inject_fault(self):
        print("== Fault Injection ==")
        injector = VirtualizationFaultInjector(namespace=self.namespace)
        injector._inject(
            fault_type="model_misconfig",
            microservices=[self.faulty_service],
        )
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")
        
    def recover_fault(self):
        print("== Fault Recovery ==")
        injector = VirtualizationFaultInjector(namespace=self.namespace)
        injector._recover(
            fault_type="model_misconfig",
            microservices=[self.faulty_service],
        )
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")


################## Detection Problem ##################
class FlowerModelMisconfigDetection(FlowerModelMisconfigBaseTask, DetectionTask):
    def __init__(self, faulty_service: str = "clientapp-1"):
        FlowerModelMisconfigBaseTask.__init__(self, faulty_service=faulty_service)
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
