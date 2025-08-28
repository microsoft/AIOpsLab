# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Variant generator for creating task variations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import random


class VariantGenerator(ABC):
    """Base class for generating task variants."""

    def __init__(self, base_config: Dict[str, Any]):
        """
        Initialize variant generator.
        
        Args:
            base_config: Base configuration for the task
        """
        self.base_config = base_config
        self.generated_variants = []
        
    @abstractmethod
    def generate_variants(self, num_variants: int = 3) -> List[Dict[str, Any]]:
        """
        Generate task variants.
        
        Args:
            num_variants: Number of variants to generate
            
        Returns:
            List of variant configurations
        """
        pass
    
    def get_next_variant(self) -> Dict[str, Any]:
        """
        Get the next variant in sequence.
        
        Returns:
            Next variant configuration or None if no more variants
        """
        if not self.generated_variants:
            self.generated_variants = self.generate_variants()
            
        if self.generated_variants:
            return self.generated_variants.pop(0)
        return None


class PortMisconfigVariantGenerator(VariantGenerator):
    """Generate variants for port misconfiguration tasks."""
    
    def __init__(self, base_config: Dict[str, Any]):
        super().__init__(base_config)
        self.used_ports = set()
        
    def generate_variants(self, num_variants: int = 3) -> List[Dict[str, Any]]:
        """
        Generate port misconfiguration variants.
        
        Each variant uses a different incorrect port number.
        """
        variants = []
        common_wrong_ports = [8080, 8081, 8082, 3000, 5000, 7000, 8000, 9091, 9092]
        
        for _ in range(num_variants):
            variant = self.base_config.copy()
            
            # Select an unused port
            available_ports = [p for p in common_wrong_ports if p not in self.used_ports]
            if not available_ports:
                # If all common ports used, generate random ones
                wrong_port = random.randint(10000, 20000)
                while wrong_port in self.used_ports:
                    wrong_port = random.randint(10000, 20000)
            else:
                wrong_port = random.choice(available_ports)
                
            self.used_ports.add(wrong_port)
            variant['wrong_port'] = wrong_port
            variants.append(variant)
            
        return variants


class ServiceVariantGenerator(VariantGenerator):
    """Generate variants for service-related tasks."""
    
    def __init__(self, base_config: Dict[str, Any], services: List[str]):
        super().__init__(base_config)
        self.services = services
        self.used_services = set()
        
    def generate_variants(self, num_variants: int = 3) -> List[Dict[str, Any]]:
        """
        Generate service variants.
        
        Each variant targets a different service.
        """
        variants = []
        
        for _ in range(min(num_variants, len(self.services))):
            variant = self.base_config.copy()
            
            # Select an unused service
            available_services = [s for s in self.services if s not in self.used_services]
            if not available_services:
                break
                
            target_service = random.choice(available_services)
            self.used_services.add(target_service)
            variant['faulty_service'] = target_service
            variants.append(variant)
            
        return variants


class ReplicaVariantGenerator(VariantGenerator):
    """Generate variants for replica count misconfigurations."""
    
    def generate_variants(self, num_variants: int = 3) -> List[Dict[str, Any]]:
        """
        Generate replica count variants.
        
        Each variant uses different extreme replica counts.
        """
        variants = []
        extreme_counts = [0, 1, 100, 1000, 10000, 100000]
        
        for i in range(min(num_variants, len(extreme_counts))):
            variant = self.base_config.copy()
            variant['replica_count'] = extreme_counts[i]
            variants.append(variant)
            
        return variants


class ConfigVariantGenerator(VariantGenerator):
    """Generate variants for configuration misconfigurations."""
    
    def __init__(self, base_config: Dict[str, Any], config_variants: Dict[str, List[Any]]):
        """
        Initialize config variant generator.
        
        Args:
            base_config: Base configuration
            config_variants: Dictionary mapping config keys to lists of variant values
                           e.g., {"replicas": [10000, 50000, 100000], "storage_class": ["fake-storage", "invalid-sc"]}
        """
        super().__init__(base_config)
        self.config_variants = config_variants
        
    def generate_variants(self, num_variants: int = 3) -> List[Dict[str, Any]]:
        """Generate configuration variants."""
        variants = []
        all_keys = list(self.config_variants.keys())
        
        for i in range(min(num_variants, len(all_keys) * max(len(v) for v in self.config_variants.values()))):
            variant = self.base_config.copy()
            
            # Select a config key to vary
            key_idx = i % len(all_keys)
            key = all_keys[key_idx]
            values = self.config_variants[key]
            
            # Select a value for this key
            value_idx = i // len(all_keys)
            if value_idx < len(values):
                variant[key] = values[value_idx]
                variants.append(variant)
                
        return variants


class NumericVariantGenerator(VariantGenerator):
    """Generate variants for numeric parameters."""
    
    def __init__(self, base_config: Dict[str, Any], 
                 param_name: str, 
                 values: List[float] = None,
                 min_val: float = None,
                 max_val: float = None):
        """
        Initialize numeric variant generator.
        
        Args:
            base_config: Base configuration
            param_name: Name of the numeric parameter to vary
            values: Explicit list of values to use
            min_val: Minimum value for random generation
            max_val: Maximum value for random generation
        """
        super().__init__(base_config)
        self.param_name = param_name
        self.values = values
        self.min_val = min_val
        self.max_val = max_val
        
    def generate_variants(self, num_variants: int = 3) -> List[Dict[str, Any]]:
        """Generate numeric parameter variants."""
        variants = []
        
        if self.values:
            # Use provided values
            for i in range(min(num_variants, len(self.values))):
                variant = self.base_config.copy()
                variant[self.param_name] = self.values[i]
                variants.append(variant)
        else:
            # Generate random values between min and max
            for _ in range(num_variants):
                variant = self.base_config.copy()
                if isinstance(self.min_val, int) and isinstance(self.max_val, int):
                    variant[self.param_name] = random.randint(self.min_val, self.max_val)
                else:
                    variant[self.param_name] = random.uniform(self.min_val, self.max_val)
                variants.append(variant)
                
        return variants


class CompositeVariantGenerator(VariantGenerator):
    """Combine multiple variant generators."""
    
    def __init__(self, generators: List[VariantGenerator]):
        self.generators = generators
        
    def generate_variants(self, num_variants: int = 3) -> List[Dict[str, Any]]:
        """
        Generate composite variants by combining multiple generators.
        """
        all_variants = []
        
        for generator in self.generators:
            variants = generator.generate_variants(num_variants)
            all_variants.extend(variants)
            
        # Shuffle and limit to requested number
        random.shuffle(all_variants)
        return all_variants[:num_variants]