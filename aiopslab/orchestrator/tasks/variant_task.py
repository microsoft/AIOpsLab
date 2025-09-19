# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base task class with variant support."""

from typing import Dict, Any, Optional
from aiopslab.orchestrator.tasks.base import Task
from aiopslab.orchestrator.variant_generator import VariantGenerator


class VariantTask(Task):
    """Task that supports variant generation for retry scenarios."""
    
    def __init__(self, variant_generator: Optional[VariantGenerator] = None):
        """
        Initialize variant task.
        
        Args:
            variant_generator: Optional variant generator for creating task variations
        """
        super().__init__()
        self.variant_generator = variant_generator
        self.current_variant = None
        self.variant_history = []
        
    def apply_variant(self, variant_config: Dict[str, Any]):
        """
        Apply a variant configuration to the task.
        
        Args:
            variant_config: Configuration to apply
        """
        self.current_variant = variant_config
        self.variant_history.append(variant_config)
        
        # Apply variant-specific configurations
        for key, value in variant_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def get_next_variant(self) -> Optional[Dict[str, Any]]:
        """
        Get the next variant from the generator.
        
        Returns:
            Next variant configuration or None if no more variants
        """
        if self.variant_generator:
            return self.variant_generator.get_next_variant()
        return None
    
    def reset_to_base(self):
        """Reset task to base configuration."""
        if self.variant_generator:
            for key in self.variant_generator.base_config:
                if hasattr(self, key):
                    setattr(self, key, self.variant_generator.base_config[key])
        self.current_variant = None
        
    def get_variant_summary(self) -> str:
        """
        Get a summary of the current variant.
        
        Returns:
            String describing the current variant
        """
        if self.current_variant:
            items = [f"{k}={v}" for k, v in self.current_variant.items()]
            return f"Variant: {', '.join(items)}"
        return "Base configuration"