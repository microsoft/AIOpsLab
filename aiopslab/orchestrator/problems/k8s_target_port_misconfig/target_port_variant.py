# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""K8S port misconfiguration with variant support."""

from typing import Any, List, Dict
from aiopslab.orchestrator.tasks import DetectionTask, LocalizationTask
from aiopslab.orchestrator.tasks.variant_task import VariantTask
from aiopslab.orchestrator.variant_generator import PortMisconfigVariantGenerator, ServiceVariantGenerator
from aiopslab.orchestrator.evaluators.quantitative import *
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.socialnet import SocialNetwork
from aiopslab.generators.workload.wrk import Wrk
from aiopslab.generators.fault.inject_virtual import VirtualizationFaultInjector
from aiopslab.session import SessionItem
from aiopslab.paths import TARGET_MICROSERVICES
from .helpers import get_frontend_url


class K8STargetPortMisconfigVariantBase(VariantTask):
    """Base class for port misconfiguration with variant support."""
    
    def __init__(self, faulty_service: str = "user-service", enable_variants: bool = True):
        """
        Initialize with variant support.
        
        Args:
            faulty_service: Initial faulty service
            enable_variants: Whether to enable variant generation
        """
        # Set up variant generator
        variant_generator = None
        if enable_variants:
            # Create generators for both port and service variants
            base_config = {
                "faulty_service": faulty_service,
                "wrong_port": 8080  # Default wrong port
            }
            
            # Available services for variants
            available_services = [
                "user-service",
                "social-graph-service", 
                "compose-post-service",
                "post-storage-service",
                "user-timeline-service",
                "home-timeline-service",
                "user-mention-service",
                "text-service",
                "media-service"
            ]
            
            # Use service variant generator
            variant_generator = ServiceVariantGenerator(base_config, available_services)
        
        super().__init__(variant_generator)
        
        self.app = SocialNetwork()
        self.kubectl = KubeCtl()
        self.namespace = self.app.namespace
        self.faulty_service = faulty_service
        self.wrong_port = 8080  # Default wrong port
        self.payload_script = (
            TARGET_MICROSERVICES
            / "socialNetwork/wrk2/scripts/social-network/compose-post.lua"
        )
        
    def apply_variant(self, variant_config: Dict[str, Any]):
        """
        Apply variant configuration.
        
        Args:
            variant_config: Configuration to apply
        """
        super().apply_variant(variant_config)
        
        # Log the variant being applied
        if "faulty_service" in variant_config:
            print(f"[Variant] Changing faulty service to: {variant_config['faulty_service']}")
        if "wrong_port" in variant_config:
            print(f"[Variant] Using wrong port: {variant_config['wrong_port']}")
            
    def start_workload(self):
        print("== Start Workload ==")
        frontend_url = get_frontend_url(self.app)
        
        wrk = Wrk(rate=10, dist="exp", connections=2, duration=10, threads=2)
        wrk.start_workload(
            payload_script=self.payload_script,
            url=f"{frontend_url}/wrk2-api/post/compose",
        )
        
    def inject_fault(self):
        print("== Fault Injection ==")
        print(f"Current variant: {self.get_variant_summary()}")
        
        injector = VirtualizationFaultInjector(namespace=self.namespace)
        
        # Inject with current configuration (potentially modified by variant)
        injector._inject(
            fault_type="misconfig_k8s",
            microservices=[self.faulty_service],
            wrong_port=self.wrong_port  # Use the potentially varied port
        )
        print(f"Service: {self.faulty_service} | Wrong Port: {self.wrong_port} | Namespace: {self.namespace}\n")
        
    def recover_fault(self):
        print("== Fault Recovery ==")
        injector = VirtualizationFaultInjector(namespace=self.namespace)
        injector._recover(
            fault_type="misconfig_k8s",
            microservices=[self.faulty_service],
        )
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")


class K8STargetPortMisconfigVariantDetection(K8STargetPortMisconfigVariantBase, DetectionTask):
    """Detection task with variant support."""
    
    def __init__(self, faulty_service: str = "user-service", enable_variants: bool = True):
        K8STargetPortMisconfigVariantBase.__init__(self, faulty_service=faulty_service, enable_variants=enable_variants)
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


class K8STargetPortMisconfigVariantLocalization(K8STargetPortMisconfigVariantBase, LocalizationTask):
    """Localization task with variant support."""
    
    def __init__(self, faulty_service: str = "user-service", enable_variants: bool = True):
        K8STargetPortMisconfigVariantBase.__init__(self, faulty_service=faulty_service, enable_variants=enable_variants)
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
            
        # Calculate accuracy based on current faulty service (which may have been varied)
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