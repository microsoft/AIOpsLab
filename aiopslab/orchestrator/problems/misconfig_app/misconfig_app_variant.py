# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Application misconfiguration problem with variant support."""

from typing import Any, Dict
from time import sleep
from aiopslab.orchestrator.tasks import DetectionTask, LocalizationTask
from aiopslab.orchestrator.tasks.variant_task import VariantTask
from aiopslab.orchestrator.variant_generator import ServiceVariantGenerator, ConfigVariantGenerator, CompositeVariantGenerator
from aiopslab.orchestrator.evaluators.quantitative import *
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.hotelres import HotelReservation
from aiopslab.generators.workload.wrk import Wrk
from aiopslab.generators.fault.inject_app import ApplicationFaultInjector
from aiopslab.session import SessionItem
from aiopslab.paths import TARGET_MICROSERVICES
from .helpers import get_frontend_url


class MisconfigAppVariantBase(VariantTask):
    """Base class for application misconfiguration with variant support."""
    
    def __init__(self, faulty_service: str = "geo", config_type: str = "env", enable_variants: bool = True):
        """
        Initialize with variant support.
        
        Args:
            faulty_service: Initial faulty service
            config_type: Type of configuration to misconfigure (env, port, connection, etc.)
            enable_variants: Whether to enable variant generation
        """
        # Set up variant generator
        variant_generator = None
        if enable_variants:
            base_config = {
                "faulty_service": faulty_service,
                "config_type": config_type
            }
            
            # Available services for HotelReservation
            available_services = [
                "geo", "profile", "rate", 
                "recommendation", "reservation", "search",
                "user", "frontend"
            ]
            
            # Configuration types that can be misconfigured
            config_types = {
                "config_type": ["env", "port", "connection", "memory", "timeout"]
            }
            
            # Create composite generator
            service_gen = ServiceVariantGenerator(base_config, available_services)
            config_gen = ConfigVariantGenerator(base_config, config_types)
            
            variant_generator = CompositeVariantGenerator([service_gen, config_gen])
        
        super().__init__(variant_generator)
        
        self.app = HotelReservation()
        self.kubectl = KubeCtl()
        self.namespace = self.app.namespace
        self.faulty_service = faulty_service
        self.config_type = config_type
        self.payload_script = (
            TARGET_MICROSERVICES
            / "hotelReservation/wrk2/scripts/hotel-reservation/mixed-workload_type_1.lua"
        )
        
    def apply_variant(self, variant_config: Dict[str, Any]):
        """Apply variant configuration."""
        super().apply_variant(variant_config)
        
        if "faulty_service" in variant_config:
            print(f"[Variant] Changing faulty service to: {variant_config['faulty_service']}")
        if "config_type" in variant_config:
            print(f"[Variant] Using config type: {variant_config['config_type']}")
            
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
        
        injector = ApplicationFaultInjector(namespace=self.namespace)
        injector._inject(
            fault_type="misconfig_app",
            microservices=[self.faulty_service],
        )
        print(f"Service: {self.faulty_service} | Config Type: {self.config_type} | Namespace: {self.namespace}\n")
        
    def recover_fault(self):
        print("== Fault Recovery ==")
        injector = ApplicationFaultInjector(namespace=self.namespace)
        injector._recover(
            fault_type="misconfig_app",
            microservices=[self.faulty_service],
        )
        
        # Wait for recovery to take effect
        sleep(30)
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")


class MisconfigAppVariantDetection(MisconfigAppVariantBase, DetectionTask):
    """Detection task with variant support."""
    
    def __init__(self, faulty_service: str = "geo", config_type: str = "env", enable_variants: bool = True):
        MisconfigAppVariantBase.__init__(self, faulty_service=faulty_service, config_type=config_type, enable_variants=enable_variants)
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


class MisconfigAppVariantLocalization(MisconfigAppVariantBase, LocalizationTask):
    """Localization task with variant support."""
    
    def __init__(self, faulty_service: str = "geo", config_type: str = "env", enable_variants: bool = True):
        MisconfigAppVariantBase.__init__(self, faulty_service=faulty_service, config_type=config_type, enable_variants=enable_variants)
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