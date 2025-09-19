# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Operator misoperation problems with variant support."""

from typing import Any, Dict
from aiopslab.orchestrator.tasks import DetectionTask, LocalizationTask
from aiopslab.orchestrator.tasks.variant_task import VariantTask
from aiopslab.orchestrator.variant_generator import ConfigVariantGenerator, NumericVariantGenerator, CompositeVariantGenerator
from aiopslab.orchestrator.evaluators.quantitative import *
from aiopslab.generators.fault.inject_operator import K8SOperatorFaultInjector
from aiopslab.service.apps.tidb_cluster_operator import TiDBCluster
from aiopslab.session import SessionItem


class K8SOperatorMisoperationVariantBase(VariantTask):
    """Base class for operator misoperation with variant support."""
    
    def __init__(self, fault_type: str = "overload_replicas", enable_variants: bool = True):
        """
        Initialize with variant support.
        
        Args:
            fault_type: Type of misoperation (overload_replicas, invalid_affinity, security_context, etc.)
            enable_variants: Whether to enable variant generation
        """
        # Set up variant generator
        variant_generator = None
        if enable_variants:
            base_config = {
                "fault_type": fault_type,
                "replica_count": 100000,
                "toleration_effect": "INVALID_EFFECT",
                "run_as_user": -1,
                "update_strategy": "InvalidStrategy",
                "storage_class": "non-existent-storage"
            }
            
            # Different fault types and their configuration variants
            fault_configs = {
                "fault_type": [
                    "overload_replicas", 
                    "invalid_affinity_toleration",
                    "security_context_fault",
                    "wrong_update_strategy",
                    "non_existent_storage"
                ],
                "replica_count": [10000, 50000, 100000, 500000],
                "toleration_effect": ["INVALID_EFFECT", "WRONG_TOLERATION", "BAD_EFFECT", "UNKNOWN"],
                "run_as_user": [-1, -100, -999, 999999],
                "update_strategy": ["InvalidStrategy", "WrongUpdate", "BadStrategy", "UnknownStrategy"],
                "storage_class": ["fake-storage", "invalid-sc", "non-existent", "wrong-class"]
            }
            
            # Create composite generator
            config_gen = ConfigVariantGenerator(base_config, fault_configs)
            replica_gen = NumericVariantGenerator(
                base_config,
                "replica_count",
                values=[1000, 10000, 50000, 100000, 200000]
            )
            
            variant_generator = CompositeVariantGenerator([config_gen, replica_gen])
        
        super().__init__(variant_generator)
        
        self.injector = K8SOperatorFaultInjector("tidb-cluster")
        self.app = TiDBCluster()
        self.fault_type = fault_type
        self.faulty_cr = "tidbclusters"
        
        # Default values for different fault types
        self.replica_count = 100000
        self.toleration_effect = "INVALID_EFFECT"
        self.run_as_user = -1
        self.update_strategy = "InvalidStrategy"
        self.storage_class = "non-existent-storage"
        
    def apply_variant(self, variant_config: Dict[str, Any]):
        """Apply variant configuration."""
        super().apply_variant(variant_config)
        
        if "fault_type" in variant_config:
            print(f"[Variant] Changing fault type to: {variant_config['fault_type']}")
        if "replica_count" in variant_config:
            print(f"[Variant] Using replica count: {variant_config['replica_count']}")
        if "toleration_effect" in variant_config:
            print(f"[Variant] Using toleration effect: {variant_config['toleration_effect']}")
        if "run_as_user" in variant_config:
            print(f"[Variant] Using run_as_user: {variant_config['run_as_user']}")
        if "update_strategy" in variant_config:
            print(f"[Variant] Using update strategy: {variant_config['update_strategy']}")
        if "storage_class" in variant_config:
            print(f"[Variant] Using storage class: {variant_config['storage_class']}")
            
    def start_workload(self):
        print("== Start Workload ==")
        print("Workload is the CR applied to the operator.")
        pass
        
    def inject_fault(self):
        print("== Fault Injection ==")
        print(f"Current variant: {self.get_variant_summary()}")
        
        # Call appropriate injection method based on fault type
        if self.fault_type == "overload_replicas":
            # For overload replicas, we'd need to modify the injector to accept replica_count parameter
            # For now, we'll use the default method
            self.injector._inject("overload_replicas")
            print(f"Injecting overload replica failure with {self.replica_count} replicas")
        elif self.fault_type == "invalid_affinity_toleration":
            self.injector._inject("invalid_affinity_toleration")
            print(f"Injecting invalid affinity toleration with effect: {self.toleration_effect}")
        elif self.fault_type == "security_context_fault":
            self.injector._inject("security_context_fault")
            print(f"Injecting security context fault with runAsUser: {self.run_as_user}")
        elif self.fault_type == "wrong_update_strategy":
            self.injector._inject("wrong_update_strategy")
            print(f"Injecting wrong update strategy: {self.update_strategy}")
        elif self.fault_type == "non_existent_storage":
            self.injector._inject("non_existent_storage")
            print(f"Injecting non-existent storage class: {self.storage_class}")
        else:
            print(f"Unknown fault type: {self.fault_type}")
            
    def recover_fault(self):
        print("== Fault Recovery ==")
        
        # Call appropriate recovery method based on fault type
        if self.fault_type == "overload_replicas":
            self.injector._recover("overload_replicas")
        elif self.fault_type == "invalid_affinity_toleration":
            self.injector._recover("invalid_affinity_toleration")
        elif self.fault_type == "security_context_fault":
            self.injector._recover("security_context_fault")
        elif self.fault_type == "wrong_update_strategy":
            self.injector._recover("wrong_update_strategy")
        elif self.fault_type == "non_existent_storage":
            self.injector._recover("non_existent_storage")
            
        print(f"Recovered {self.fault_type} failure of the TiDB cluster\n")


class K8SOperatorMisoperationVariantDetection(K8SOperatorMisoperationVariantBase, DetectionTask):
    """Detection task with variant support."""
    
    def __init__(self, fault_type: str = "overload_replicas", enable_variants: bool = True):
        K8SOperatorMisoperationVariantBase.__init__(self, fault_type=fault_type, enable_variants=enable_variants)
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


class K8SOperatorMisoperationVariantLocalization(K8SOperatorMisoperationVariantBase, LocalizationTask):
    """Localization task with variant support."""
    
    def __init__(self, fault_type: str = "overload_replicas", enable_variants: bool = True):
        K8SOperatorMisoperationVariantBase.__init__(self, fault_type=fault_type, enable_variants=enable_variants)
        LocalizationTask.__init__(self, self.app)
        self.task_desc = """\
            You are an expert DevOps engineer who has been tasked with localizing faults in a deployed service.

            The service you are working with today is described below:
            {app_summary}

            You will begin by analyzing the service's state and telemetry, and then submit one of two possible solutions:
            1. list[str]: list of faulty components or custom resources (e.g., service names, CRs)
            2. str: `None` if no faults were detected
            """
        
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
            
        # Calculate exact match and subset
        is_exact = is_exact_match(soln, self.faulty_cr)
        is_sub = is_subset([self.faulty_cr], soln)
        
        # Determine accuracy
        if is_exact:
            accuracy = 100.0
            print(f"Exact match: {soln} | Accuracy: {accuracy}%")
        elif is_sub:
            accuracy = (len([self.faulty_cr]) / len(soln)) * 100.0
            print(f"Subset match: {soln} | Accuracy: {accuracy:.2f}%")
        else:
            accuracy = 0.0
            print(f"No match: {soln} | Accuracy: {accuracy}%")
            
        self.add_result("Localization Accuracy", accuracy)
        super().eval(soln, trace, duration)
        
        self.results["success"] = is_exact or (is_sub and len(soln) == 1)
        self.results["is_subset"] = is_sub
        
        return self.results