# Licensed under the MIT License.

"""Container kill problem with variant support."""

from typing import Any, Dict, List

from aiopslab.orchestrator.tasks import (
    AnalysisTask,
    DetectionTask,
    LocalizationTask,
    MitigationTask,
)
from aiopslab.orchestrator.tasks.variant_task import VariantProblemMixin
from aiopslab.orchestrator.variant_generator import VariantGenerator
from aiopslab.orchestrator.evaluators.quantitative import is_exact_match, is_subset
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.hotelres import HotelReservation
from aiopslab.generators.workload.wrk import Wrk
from aiopslab.generators.fault.inject_symp import SymptomFaultInjector
from aiopslab.session import SessionItem
from aiopslab.paths import TARGET_MICROSERVICES

from .helpers import get_frontend_url


class ServiceContainerVariantGenerator(VariantGenerator):
    """Variant generator for service-container pairs."""

    def __init__(self, base_config: Dict[str, Any], pairs: List[Dict[str, str]]):
        super().__init__(base_config)
        self.pairs = pairs

    def generate_variants(self, num_variants: int = 3) -> List[Dict[str, Any]]:
        variants: List[Dict[str, Any]] = []
        for pair in self.pairs[:num_variants]:
            variant = self.base_config.copy()
            variant.update(pair)
            variants.append(variant)
        return variants


class ContainerKillVariantBase(VariantProblemMixin):
    """Base class for container kill with variant support."""

    def __init__(
        self,
        faulty_service: str = "geo",
        faulty_container: str = "hotel-reserv-geo",
        enable_variants: bool = True,
    ):
        variant_generator = None
        if enable_variants:
            base_config = {
                "faulty_service": faulty_service,
                "faulty_container": faulty_container,
            }

            service_containers = [
                {"faulty_service": "geo", "faulty_container": "hotel-reserv-geo"},
                {"faulty_service": "profile", "faulty_container": "hotel-reserv-profile"},
                {"faulty_service": "rate", "faulty_container": "hotel-reserv-rate"},
                {
                    "faulty_service": "recommendation",
                    "faulty_container": "hotel-reserv-recommendation",
                },
                {
                    "faulty_service": "reservation",
                    "faulty_container": "hotel-reserv-reservation",
                },
                {"faulty_service": "search", "faulty_container": "hotel-reserv-search"},
                {"faulty_service": "user", "faulty_container": "hotel-reserv-user"},
                {
                    "faulty_service": "frontend",
                    "faulty_container": "hotel-reserv-frontend",
                },
            ]

            variant_generator = ServiceContainerVariantGenerator(base_config, service_containers)

        super().__init__(variant_generator)

        self.app = HotelReservation()
        self.kubectl = KubeCtl()
        self.namespace = self.app.namespace
        self.faulty_service = faulty_service
        self.faulty_container = faulty_container
        self.payload_script = (
            TARGET_MICROSERVICES
            / "hotelReservation/wrk2/scripts/hotel-reservation/mixed-workload_type_1.lua"
        )
        self.symptom_injector = SymptomFaultInjector(namespace=self.namespace)
        self.experiment_name = "container-kill-mesh"
        self.chaos_type = "podchaos"

        self.set_expected_faulty_components([self.faulty_service])
        self.set_expected_system_level("Virtualization")
        self.set_expected_fault_type("Operation Error")
        self.set_mitigation_expectations(
            pods_ready={
                "namespace": self.namespace,
                "services": [self.faulty_service],
            }
        )
        self.finalize_variant_base_state()

    def apply_variant(self, variant_config: Dict[str, Any]):
        super().apply_variant(variant_config)

        self.set_expected_faulty_components([self.faulty_service])
        self.set_mitigation_expectations(
            pods_ready={
                "namespace": self.namespace,
                "services": [self.faulty_service],
            }
        )
        self.set_variant_id(f"{self.faulty_service}:{self.faulty_container}")

        if "faulty_service" in variant_config:
            print(f"[Variant] Changing faulty service to: {variant_config['faulty_service']}")
        if "faulty_container" in variant_config:
            print(f"[Variant] Changing faulty container to: {variant_config['faulty_container']}")

    def start_workload(self):
        print("== Start Workload ==")
        frontend_url = get_frontend_url(self.app)

        wrk = Wrk(rate=100, dist="exp", connections=2, duration=10, threads=2)
        wrk.start_workload(
            payload_script=self.payload_script,
            url=f"{frontend_url}",
        )

    def inject_fault(self):
        print("== Fault Injection ==")
        print(f"Current variant: {self.get_variant_summary()}")

        self.symptom_injector.inject_container_kill(
            self.faulty_service, self.faulty_container
        )
        print(
            f"Service: {self.faulty_service} | Container: {self.faulty_container} | Namespace: {self.namespace}\n"
        )

    def recover_fault(self):
        print("== Fault Recovery ==")
        self.symptom_injector.recover_container_kill()
        print(f"Recovered Service: {self.faulty_service} | Namespace: {self.namespace}\n")


class ContainerKillVariantDetection(ContainerKillVariantBase, DetectionTask):
    """Detection task with variant support."""

    def __init__(
        self,
        faulty_service: str = "geo",
        faulty_container: str = "hotel-reserv-geo",
        enable_variants: bool = True,
    ):
        ContainerKillVariantBase.__init__(
            self,
            faulty_service=faulty_service,
            faulty_container=faulty_container,
            enable_variants=enable_variants,
        )
        DetectionTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")
        print(f"Evaluating with configuration: {self.get_variant_summary()}")

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


class ContainerKillVariantLocalization(ContainerKillVariantBase, LocalizationTask):
    """Localization task with variant support."""

    def __init__(
        self,
        faulty_service: str = "geo",
        faulty_container: str = "hotel-reserv-geo",
        enable_variants: bool = True,
    ):
        ContainerKillVariantBase.__init__(
            self,
            faulty_service=faulty_service,
            faulty_container=faulty_container,
            enable_variants=enable_variants,
        )
        LocalizationTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")
        print(f"Evaluating with configuration: {self.get_variant_summary()}")

        expected_components = self.get_expected_faulty_components() or [self.faulty_service]

        if soln is None:
            print("Solution is None")
            self.add_result("Localization Accuracy", 0.0)
            self.results["success"] = False
            self.results["is_subset"] = False
            super().eval(soln, trace, duration)
            return self.results

        target = expected_components if len(expected_components) > 1 else expected_components[0]
        is_exact = is_exact_match(soln, target)

        if isinstance(soln, list):
            is_sub = is_subset(expected_components, soln)
            predicted_len = len(soln)
        else:
            is_sub = soln in expected_components
            predicted_len = 1

        if is_exact:
            accuracy = 100.0
            print(f"Exact match: {soln} | Accuracy: {accuracy}%")
        elif is_sub and isinstance(soln, list) and predicted_len > 0:
            accuracy = (len(expected_components) / predicted_len) * 100.0
            print(f"Subset match: {soln} | Accuracy: {accuracy:.2f}%")
        elif is_sub:
            accuracy = 100.0
            print(f"Subset match: {soln} | Accuracy: {accuracy:.2f}%")
        else:
            accuracy = 0.0
            print(f"No match: {soln} | Accuracy: {accuracy}%")

        self.add_result("Localization Accuracy", accuracy)
        super().eval(soln, trace, duration)

        expected_len = len(expected_components)
        self.results["success"] = is_exact or (is_sub and predicted_len == expected_len)
        self.results["is_subset"] = is_sub

        return self.results


class ContainerKillVariantAnalysis(ContainerKillVariantBase, AnalysisTask):
    """Analysis task with variant support."""

    def __init__(
        self,
        faulty_service: str = "geo",
        faulty_container: str = "hotel-reserv-geo",
        enable_variants: bool = True,
    ):
        ContainerKillVariantBase.__init__(
            self,
            faulty_service=faulty_service,
            faulty_container=faulty_container,
            enable_variants=enable_variants,
        )
        AnalysisTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")
        self.evaluate_variant_analysis(soln)
        return super().eval(soln, trace, duration)


class ContainerKillVariantMitigation(ContainerKillVariantBase, MitigationTask):
    """Mitigation task with variant support."""

    def __init__(
        self,
        faulty_service: str = "geo",
        faulty_container: str = "hotel-reserv-geo",
        enable_variants: bool = True,
    ):
        ContainerKillVariantBase.__init__(
            self,
            faulty_service=faulty_service,
            faulty_container=faulty_container,
            enable_variants=enable_variants,
        )
        MitigationTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")
        return super().eval(soln, trace, duration)
