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
        
        path = "/app/.flwr/apps"
        check = f""" docker exec -it {self.faulty_service} sh -c "test -d {path} && echo 'exists'" """
        
        print("Waiting for workload to start...")
        while True:
            exists = self.docker.exec_command(check)
            if exists.strip() == "exists":
                break
            time.sleep(1)
        print("Workload started successfully.")
        
        # Inject fault after workload starts, since the required files are created during the workload
        print("Injecting fault...")
        self.inject_fault(inject=True)
        
        print("Waiting for faults to propagate...")
        while True:
            logs = self.docker.get_logs(self.faulty_service)
            if "error" in logs.lower():
                break
            time.sleep(1)
        print("Faults propagated.")
        
    def inject_fault(self, inject: bool = False):
        print("== Fault Injection ==")
        if inject:
            injector = VirtualizationFaultInjector(namespace=self.namespace)
            injector._inject(
                fault_type="model_misconfig",
                microservices=[self.faulty_service],
            )
            print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")
        else:
            print("Fault injection skipped.")
        
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
