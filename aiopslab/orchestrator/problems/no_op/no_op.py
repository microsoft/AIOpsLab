"""No operation problem for HotelReservation or SocialNetwork applications to test false positive."""

from typing import Any

from aiopslab.orchestrator.tasks import *
from aiopslab.orchestrator.evaluators.quantitative import *
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.hotelres import HotelReservation
from aiopslab.service.apps.socialnet import SocialNetwork
from aiopslab.service.apps.astronomy_shop import AstronomyShop
from aiopslab.generators.workload.wrk import Wrk
from aiopslab.generators.fault.inject_noop import NoopFaultInjector
from aiopslab.session import SessionItem
from aiopslab.paths import TARGET_MICROSERVICES

from .helpers import get_frontend_url


class NoOpBaseTask:
    def __init__(self, app_name: str = "hotel"):
        self.app_name = app_name

        if self.app_name == "hotel":
            self.app = HotelReservation()
            self.payload_script = (
                TARGET_MICROSERVICES
                / "hotelReservation/wrk2/scripts/hotel-reservation/mixed-workload_type_1.lua"
            )
            self.faulty_service = "PLACEHOLDER"
        elif self.app_name == "social":
            self.app = SocialNetwork()
            self.payload_script = (
                TARGET_MICROSERVICES
                / "socialNetwork/wrk2/scripts/social-network/compose-post.lua"
            )
            self.faulty_service = "PLACEHOLDER"
        elif self.app_name == "astronomy_shop":
            self.app = AstronomyShop()
            self.faulty_service = "PLACEHOLDER"
        else:
            raise ValueError(f"Unsupported app_name: {app_name}")

        self.kubectl = KubeCtl()
        self.namespace = self.app.namespace
        self.injector = NoopFaultInjector(namespace=self.namespace)

    def start_workload(self):
        if self.app_name != 'astronomy_shop':
            print("== Start Workload ==")
            frontend_url = get_frontend_url(self.app)

            wrk = Wrk(rate=10, dist="exp", connections=2, duration=10, threads=2)
            wrk.start_workload(
                payload_script=self.payload_script,
                url=f"{frontend_url}",
            )
        else:
            # Skip workload since astronomy shop has its own workload generator
            print("== Workload Skipped ==")

    def inject_fault(self):
        print("== Fault Injection ==")
        self.injector._inject(
            fault_type="no_op", microservices=[self.faulty_service], duration="200s"
        )
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    def recover_fault(self):
        print("== Fault Recovery ==")
        self.injector._recover(
            fault_type="no_op",
        )


################## Detection Problem ##################
class NoOpDetection(NoOpBaseTask, DetectionTask):
    def __init__(self, app_name: str = "hotel"):
        NoOpBaseTask.__init__(self, app_name)
        DetectionTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")
        expected_solution = "No"

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
