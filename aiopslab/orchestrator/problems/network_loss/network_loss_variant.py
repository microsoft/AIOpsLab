# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Network loss problem with variant support."""

from typing import Any, Dict
from aiopslab.orchestrator.tasks import DetectionTask, LocalizationTask
from aiopslab.orchestrator.tasks.variant_task import VariantTask
from aiopslab.orchestrator.variant_generator import ServiceVariantGenerator, NumericVariantGenerator, CompositeVariantGenerator
from aiopslab.orchestrator.evaluators.quantitative import *
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.hotelres import HotelReservation
from aiopslab.generators.workload.wrk import Wrk
from aiopslab.generators.fault.inject_symp import SymptomFaultInjector
from aiopslab.session import SessionItem
from aiopslab.paths import TARGET_MICROSERVICES
from .helpers import get_frontend_url


class NetworkLossVariantBase(VariantTask):
    """Base class for network loss with variant support."""
    
    def __init__(self, faulty_service: str = "user", loss_rate: float = 0.1, enable_variants: bool = True):
        """
        Initialize with variant support.
        
        Args:
            faulty_service: Initial faulty service
            loss_rate: Packet loss rate (0.0 to 1.0)
            enable_variants: Whether to enable variant generation
        """
        # Set up variant generator
        variant_generator = None
        if enable_variants:
            base_config = {
                "faulty_service": faulty_service,
                "loss_rate": loss_rate
            }
            
            # Available services for HotelReservation
            available_services = [
                "user", "geo", "profile", "rate", 
                "recommendation", "reservation", "search",
                "frontend"
            ]
            
            # Create composite generator for both service and loss rate variants
            service_gen = ServiceVariantGenerator(base_config, available_services)
            loss_gen = NumericVariantGenerator(
                base_config, 
                "loss_rate",
                values=[0.05, 0.1, 0.2, 0.3, 0.5]  # Different packet loss rates
            )
            
            variant_generator = CompositeVariantGenerator([service_gen, loss_gen])
        
        super().__init__(variant_generator)
        
        self.app = HotelReservation()
        self.kubectl = KubeCtl()
        self.namespace = self.app.namespace
        self.faulty_service = faulty_service
        self.loss_rate = loss_rate
        self.payload_script = (
            TARGET_MICROSERVICES
            / "hotelReservation/wrk2/scripts/hotel-reservation/mixed-workload_type_1.lua"
        )
        self.injector = SymptomFaultInjector(namespace=self.namespace)
        
    def apply_variant(self, variant_config: Dict[str, Any]):
        """Apply variant configuration."""
        super().apply_variant(variant_config)
        
        if "faulty_service" in variant_config:
            print(f"[Variant] Changing faulty service to: {variant_config['faulty_service']}")
        if "loss_rate" in variant_config:
            print(f"[Variant] Using loss rate: {variant_config['loss_rate']*100:.1f}%")
            
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
        print(f"Current variant: {self.get_variant_summary()}")
        
        # Note: The original inject_network_loss might need modification to accept loss_rate parameter
        # For now, we'll call it with the service only as in the original
        self.injector.inject_network_loss([self.faulty_service])
        print(f"Service: {self.faulty_service} | Loss Rate: {self.loss_rate*100:.1f}% | Namespace: {self.namespace}\n")
        
    def recover_fault(self):
        print("== Fault Recovery ==")
        self.injector.recover_network_loss()


class NetworkLossVariantDetection(NetworkLossVariantBase, DetectionTask):
    """Detection task with variant support."""
    
    def __init__(self, faulty_service: str = "user", loss_rate: float = 0.1, enable_variants: bool = True):
        NetworkLossVariantBase.__init__(self, faulty_service=faulty_service, loss_rate=loss_rate, enable_variants=enable_variants)
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


class NetworkLossVariantLocalization(NetworkLossVariantBase, LocalizationTask):
    """Localization task with variant support."""
    
    def __init__(self, faulty_service: str = "user", loss_rate: float = 0.1, enable_variants: bool = True):
        NetworkLossVariantBase.__init__(self, faulty_service=faulty_service, loss_rate=loss_rate, enable_variants=enable_variants)
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