# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Scale pod problem with variant support."""

from typing import Any, Dict
from aiopslab.orchestrator.tasks import DetectionTask, LocalizationTask
from aiopslab.orchestrator.tasks.variant_task import VariantTask
from aiopslab.orchestrator.variant_generator import ServiceVariantGenerator
from aiopslab.orchestrator.evaluators.quantitative import is_exact_match, is_subset
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.socialnet import SocialNetwork
from aiopslab.generators.workload.wrk import Wrk
from aiopslab.generators.fault.inject_virtual import VirtualizationFaultInjector
from aiopslab.session import SessionItem
from aiopslab.paths import TARGET_MICROSERVICES
from .helpers import get_frontend_url


class ScalePodVariantBase(VariantTask):
    """Base class for scale pod with variant support."""
    
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
            base_config = {"faulty_service": faulty_service}
            
            # Available services for SocialNetwork
            available_services = [
                "user-service",
                "social-graph-service", 
                "compose-post-service",
                "post-storage-service",
                "user-timeline-service",
                "home-timeline-service",
                "user-mention-service",
                "text-service",
                "media-service",
                "unique-id-service",
                "url-shorten-service",
                "nginx-thrift",
                "user-mongodb",
                "social-graph-mongodb",
                "post-storage-mongodb",
                "user-timeline-mongodb",
                "home-timeline-mongodb",
                "url-shorten-mongodb",
                "media-mongodb",
                "compose-post-redis",
                "social-graph-redis",
                "user-timeline-redis",
                "home-timeline-redis",
                "post-storage-memcached",
                "user-memcached"
            ]
            
            variant_generator = ServiceVariantGenerator(base_config, available_services)
        
        super().__init__(variant_generator)
        
        self.app = SocialNetwork()
        self.kubectl = KubeCtl()
        self.namespace = self.app.namespace
        self.faulty_service = faulty_service
        self.payload_script = (
            TARGET_MICROSERVICES
            / "socialNetwork/wrk2/scripts/social-network/compose-post.lua"
        )
        
    def apply_variant(self, variant_config: Dict[str, Any]):
        """Apply variant configuration."""
        super().apply_variant(variant_config)
        
        if "faulty_service" in variant_config:
            print(f"[Variant] Changing faulty service to: {variant_config['faulty_service']}")
            
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
        injector._inject(
            fault_type="scale_pods_to_zero",
            microservices=[self.faulty_service],
        )
        print(f"Service: {self.faulty_service} | Scaled to 0 replicas | Namespace: {self.namespace}\n")
        
    def recover_fault(self):
        print("== Fault Recovery ==")
        injector = VirtualizationFaultInjector(namespace=self.namespace)
        injector._recover(
            fault_type="scale_pods_to_zero",
            microservices=[self.faulty_service],
        )


class ScalePodVariantDetection(ScalePodVariantBase, DetectionTask):
    """Detection task with variant support."""
    
    def __init__(self, faulty_service: str = "user-service", enable_variants: bool = True):
        ScalePodVariantBase.__init__(self, faulty_service=faulty_service, enable_variants=enable_variants)
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


class ScalePodVariantLocalization(ScalePodVariantBase, LocalizationTask):
    """Localization task with variant support."""
    
    def __init__(self, faulty_service: str = "user-service", enable_variants: bool = True):
        ScalePodVariantBase.__init__(self, faulty_service=faulty_service, enable_variants=enable_variants)
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