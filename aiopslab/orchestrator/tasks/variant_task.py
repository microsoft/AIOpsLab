# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base task class with variant support."""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, Optional

from aiopslab.orchestrator.tasks.base import Task
from aiopslab.orchestrator.variant_generator import VariantGenerator


class VariantTask(Task):
    """Task that supports variant generation for retry scenarios."""

    def __init__(self, variant_generator: Optional[VariantGenerator] = None):
        """Initialize the variant-enabled task."""
        # Call Task.__init__ directly to avoid invoking sibling Task subclasses
        # (e.g., DetectionTask.__init__(app)) via MRO in multiple inheritance.
        # This prevents TypeError when Variant mixins are constructed before
        # concrete Detection/Localization/Analysis/Mitigation tasks supply `app`.
        Task.__init__(self)
        self.variant_generator = variant_generator
        self.current_variant: Optional[Dict[str, Any]] = None
        self.variant_history: list[Dict[str, Any]] = []

    def apply_variant(self, variant_config: Dict[str, Any]):
        """Apply a variant configuration to the task."""
        variant_config = variant_config or {}
        self.current_variant = variant_config
        self.variant_history.append(variant_config)

        for key, value in variant_config.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_next_variant(self) -> Optional[Dict[str, Any]]:
        """Get the next variant from the configured generator."""
        if self.variant_generator:
            return self.variant_generator.get_next_variant()
        return None

    def reset_to_base(self):
        """Reset task to its generator-defined base configuration."""
        if self.variant_generator:
            for key, value in self.variant_generator.base_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        self.current_variant = None

    def get_variant_id(self) -> str:
        """Return a concise identifier for the current variant."""
        if not self.current_variant:
            return "base"

        keys = [key for key in sorted(self.current_variant.keys()) if not key.startswith("_")]
        parts: list[str] = []

        for key in keys:
            value = self.current_variant[key]
            if isinstance(value, (list, tuple, set)):
                value = ",".join(str(item) for item in list(value)[:3])
            elif isinstance(value, dict):
                items = list(value.items())[:3]
                value = ",".join(f"{k}:{v}" for k, v in items)
            parts.append(f"{key}={value}")
            if len(parts) >= 3:
                break

        return "|".join(parts) if parts else "variant"

    def get_variant_summary(self) -> str:
        """Get a human readable summary of the current variant."""
        if self.current_variant:
            items = [f"{k}={v}" for k, v in self.current_variant.items()]
            return f"Variant[{self.get_variant_id()}]: {', '.join(items)}"
        return "Base configuration"


class VariantProblemMixin(VariantTask):
    """Mixin that augments tasks with variant metadata for evaluation."""

    def __init__(self, variant_generator: Optional[VariantGenerator] = None):
        super().__init__(variant_generator)
        self.variant_metadata: Dict[str, Any] = {
            "faulty_components": [],
            "system_level": None,
            "fault_type": None,
            "mitigation": {},
            "variant_id": "base",
        }
        self._base_metadata: Dict[str, Any] = copy.deepcopy(self.variant_metadata)

    # ------------------------------------------------------------------
    # Expectation helpers
    # ------------------------------------------------------------------
    def set_expected_faulty_components(self, components: Iterable[str] | None):
        """Store the expected faulty components for evaluation."""
        components = components or []
        self.variant_metadata["faulty_components"] = list(dict.fromkeys(str(c) for c in components if c))

    def get_expected_faulty_components(self) -> list[str]:
        """Return the faulty components expected for the current variant."""
        return list(self.variant_metadata.get("faulty_components", []))

    def set_expected_system_level(self, system_level: Optional[str]):
        self.variant_metadata["system_level"] = system_level

    def get_expected_system_level(self) -> Optional[str]:
        return self.variant_metadata.get("system_level")

    def set_expected_fault_type(self, fault_type: Optional[str]):
        self.variant_metadata["fault_type"] = fault_type

    def get_expected_fault_type(self) -> Optional[str]:
        return self.variant_metadata.get("fault_type")

    def set_mitigation_expectations(self, checks: Optional[Dict[str, Any]] = None, **kwargs: Any):
        """Update mitigation health check expectations."""
        mitigation = self.variant_metadata.setdefault("mitigation", {})
        updates = {}
        if checks:
            updates.update(checks)
        updates.update(kwargs)

        for key, value in updates.items():
            if value is None:
                mitigation.pop(key, None)
            else:
                mitigation[key] = value

    def get_mitigation_expectations(self) -> Dict[str, Any]:
        return copy.deepcopy(self.variant_metadata.get("mitigation", {}))

    def set_variant_id(self, variant_id: Optional[str]):
        self.variant_metadata["variant_id"] = variant_id or "base"

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def finalize_variant_base_state(self):
        """Capture the current expectations as the base state."""
        self._base_metadata = copy.deepcopy(self.variant_metadata)

    def apply_variant(self, variant_config: Dict[str, Any]):  # type: ignore[override]
        self.variant_metadata = copy.deepcopy(self._base_metadata)
        super().apply_variant(variant_config or {})

        config = variant_config or {}

        expectations = {}
        if "expectations" in config and isinstance(config["expectations"], dict):
            expectations.update(config["expectations"])
        for key in (
            "expected_faulty_components",
            "expected_system_level",
            "expected_fault_type",
            "mitigation_checks",
            "variant_id",
        ):
            if key in config:
                expectations[key] = config[key]

        # Derive defaults when expectations aren't provided explicitly.
        if "expected_faulty_components" not in expectations:
            derived: list[str] = []
            if "faulty_service" in config:
                derived = [config["faulty_service"]]
            elif "faulty_services" in config and isinstance(config["faulty_services"], Iterable):
                derived = [str(item) for item in config["faulty_services"]]
            elif "faulty_cr" in config:
                derived = [config["faulty_cr"]]
            if derived:
                expectations["expected_faulty_components"] = derived

        if expectations:
            self._ingest_expectation_metadata(expectations)

        self._after_variant_applied(config)

        if not self.variant_metadata.get("variant_id"):
            self.variant_metadata["variant_id"] = super().get_variant_id()

    def _ingest_expectation_metadata(self, metadata: Dict[str, Any]):
        if "expected_faulty_components" in metadata:
            self.set_expected_faulty_components(metadata["expected_faulty_components"])
        if "expected_system_level" in metadata:
            self.set_expected_system_level(metadata["expected_system_level"])
        if "expected_fault_type" in metadata:
            self.set_expected_fault_type(metadata["expected_fault_type"])
        if "mitigation_checks" in metadata and isinstance(metadata["mitigation_checks"], dict):
            self.set_mitigation_expectations(metadata["mitigation_checks"])
        if "variant_id" in metadata:
            self.set_variant_id(metadata["variant_id"])

    def _after_variant_applied(self, variant_config: Dict[str, Any]):
        """Hook for subclasses to extend variant application."""
        if not self.get_expected_faulty_components():
            derived = []
            if hasattr(self, "faulty_service"):
                derived = [str(getattr(self, "faulty_service"))]
            elif hasattr(self, "faulty_cr"):
                derived = [str(getattr(self, "faulty_cr"))]
            if derived:
                self.set_expected_faulty_components(derived)

        if not self.variant_metadata.get("variant_id"):
            self.variant_metadata["variant_id"] = super().get_variant_id()

    def reset_to_base(self):  # type: ignore[override]
        super().reset_to_base()
        self.variant_metadata = copy.deepcopy(self._base_metadata)

    def get_variant_id(self) -> str:  # type: ignore[override]
        variant_id = self.variant_metadata.get("variant_id")
        return str(variant_id) if variant_id else super().get_variant_id()

    def get_variant_summary(self) -> str:  # type: ignore[override]
        summary = super().get_variant_summary()
        variant_id = self.get_variant_id()
        if summary.startswith("Base"):
            return summary
        return f"{summary} (id={variant_id})"
