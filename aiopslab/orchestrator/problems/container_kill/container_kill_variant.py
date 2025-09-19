# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Container kill problem with variant support."""

from typing import Any, Dict
import yaml
from aiopslab.orchestrator.tasks import DetectionTask, LocalizationTask
from aiopslab.orchestrator.tasks.variant_task import VariantTask
from aiopslab.orchestrator.variant_generator import ServiceVariantGenerator, ConfigVariantGenerator, CompositeVariantGenerator
from aiopslab.orchestrator.evaluators.quantitative import *
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.hotelres import HotelReservation
from aiopslab.generators.workload.wrk import Wrk
from aiopslab.generators.fault.inject_symp import SymptomFaultInjector
from aiopslab.session import SessionItem
from aiopslab.paths import TARGET_MICROSERVICES
from .helpers import get_frontend_url


class ContainerKillVariantBase(VariantTask):
    """Base class for container kill with variant support."""
    
    def __init__(self, faulty_service: str = "geo", faulty_container: str = "hotel-reserv-geo", enable_variants: bool = True):
        """
        Initialize with variant support.
        
        Args:
            faulty_service: Initial faulty service
            faulty_container: Initial faulty container
            enable_variants: Whether to enable variant generation
        """
        # Set up variant generator
        variant_generator = None
        if enable_variants:
            base_config = {
                "faulty_service": faulty_service,
                "faulty_container": faulty_container
            }
            
            # Service-container pairs for HotelReservation
            service_containers = [
                {"faulty_service": "geo", "faulty_container": "hotel-reserv-geo"},
                {"faulty_service": "profile", "faulty_container": "hotel-reserv-profile"},
                {"faulty_service": "rate", "faulty_container": "hotel-reserv-rate"},
                {"faulty_service": "recommendation", "faulty_container": "hotel-reserv-recommendation"},
                {"faulty_service": "reservation", "faulty_container": "hotel-reserv-reservation"},
                {"faulty_service": "search", "faulty_container": "hotel-reserv-search"},
                {"faulty_service": "user", "faulty_container": "hotel-reserv-user"},
                {"faulty_service": "frontend", "faulty_container": "hotel-reserv-frontend"},
            ]
            
            # Create a custom variant generator for service-container pairs
            variants_config = {
                "service_container": service_containers
            }
            
            # We'll use ConfigVariantGenerator with a special handling
            class ServiceContainerVariantGenerator(VariantGenerator):
                def __init__(self, base_config, pairs):
                    from aiopslab.orchestrator.variant_generator import VariantGenerator
                    super().__init__(base_config)
                    self.pairs = pairs
                    self.used_indices = set()
                    
                def generate_variants(self, num_variants: int = 3):
                    variants = []
                    for i in range(min(num_variants, len(self.pairs))):
                        if i not in self.used_indices:
                            variant = self.base_config.copy()
                            variant.update(self.pairs[i])
                            variants.append(variant)
                            self.used_indices.add(i)
                    return variants
            
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
        
    def apply_variant(self, variant_config: Dict[str, Any]):
        """Apply variant configuration."""
        super().apply_variant(variant_config)
        
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
        print(f"Service: {self.faulty_service} | Container: {self.faulty_container} | Namespace: {self.namespace}\n")
        
    def recover_fault(self):
        print("== Fault Recovery ==")
        self.symptom_injector.recover_container_kill()
        print(f"Recovered Service: {self.faulty_service} | Namespace: {self.namespace}\n")


class ContainerKillVariantDetection(ContainerKillVariantBase, DetectionTask):
    """Detection task with variant support."""
    
    def __init__(self, faulty_service: str = "geo", faulty_container: str = "hotel-reserv-geo", enable_variants: bool = True):
        ContainerKillVariantBase.__init__(self, faulty_service=faulty_service, faulty_container=faulty_container, enable_variants=enable_variants)
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
    
    def __init__(self, faulty_service: str = "geo", faulty_container: str = "hotel-reserv-geo", enable_variants: bool = True):
        ContainerKillVariantBase.__init__(self, faulty_service=faulty_service, faulty_container=faulty_container, enable_variants=enable_variants)
        LocalizationTask.__init__(self, self.app)
        
    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")
        print(f"Evaluating with configuration: {self.get_variant_summary()}")
        
        if soln is None:
            print("Solution is None")
            self.add_result("Localization Accuracy", 0.0)
            self.results["success"] = False
            self.results["is_subset"] = False
            super().eval(soln, trace, duration)
            return self.results
            
        is_exact = is_exact_match(soln, self.faulty_service)
        is_sub = is_subset([self.faulty_service], soln)
        
        if is_exact:
            accuracy = 100.0
            print(f"Exact match: {soln} == {self.faulty_service} | Accuracy: {accuracy}%")
        elif is_sub:
            accuracy = (len([self.faulty_service]) / len(soln)) * 100.0
            print(f"Subset match: {soln} contains {self.faulty_service} | Accuracy: {accuracy:.2f}%")
        else:
            accuracy = 0.0
            print(f"No match: {soln} != {self.faulty_service} | Accuracy: {accuracy}%")
            
        self.add_result("Localization Accuracy", accuracy)
        super().eval(soln, trace, duration)
        
        self.results["success"] = is_exact or (is_sub and len(soln) == 1)
        self.results["is_subset"] = is_sub
        
        return self.results