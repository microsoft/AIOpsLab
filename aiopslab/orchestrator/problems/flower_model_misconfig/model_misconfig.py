# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model misconfiguration fault in the Flower application."""

import os
import time
from typing import Any, Callable

from aiopslab.orchestrator.tasks import *
from aiopslab.service.dock import Docker
from aiopslab.service.apps.flower import Flower
from aiopslab.paths import TARGET_MICROSERVICES
from aiopslab.session import SessionItem
from aiopslab.generators.fault.inject_virtual import VirtualizationFaultInjector

WORKLOAD_START_TIMEOUT_SECONDS = int(
    os.getenv("AIOPSLAB_FLOWER_WORKLOAD_START_TIMEOUT_SECONDS", "300")
)
FAULT_PROPAGATION_TIMEOUT_SECONDS = int(
    os.getenv("AIOPSLAB_FLOWER_FAULT_PROPAGATION_TIMEOUT_SECONDS", "300")
)


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
        self.docker.exec_command(
            command,
            cwd=self.train_dir,
            timeout=WORKLOAD_START_TIMEOUT_SECONDS,
        )
        
        path = "/app/.flwr/apps"
        check = f""" docker exec -it {self.faulty_service} sh -c "test -d {path} && echo 'exists'" """
        
        print("Waiting for workload to start...")
        self._wait_until(
            "Flower workload start",
            lambda: self.docker.exec_command(check).strip() == "exists",
            WORKLOAD_START_TIMEOUT_SECONDS,
        )
        print("Workload started successfully.")
        
        # Inject fault after workload starts, since the required files are created during the workload
        print("Injecting fault...")
        self.inject_fault(inject=True)
        
        print("Waiting for faults to propagate...")
        self._wait_until(
            "Flower fault propagation",
            lambda: "error" in self.docker.get_logs(self.faulty_service).lower(),
            FAULT_PROPAGATION_TIMEOUT_SECONDS,
        )
        print("Faults propagated.")

    def _wait_until(
        self,
        description: str,
        predicate: Callable[[], bool],
        timeout_seconds: int,
        interval_seconds: float = 1,
    ):
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(interval_seconds)
        raise TimeoutError(f"{description} did not complete within {timeout_seconds} seconds.")
        
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
