import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

aiopslab_pkg = types.ModuleType("aiopslab")
aiopslab_pkg.__path__ = [str(REPO_ROOT / "aiopslab")]
sys.modules.setdefault("aiopslab", aiopslab_pkg)

orchestrator_pkg = types.ModuleType("aiopslab.orchestrator")
orchestrator_pkg.__path__ = [str(REPO_ROOT / "aiopslab" / "orchestrator")]
sys.modules.setdefault("aiopslab.orchestrator", orchestrator_pkg)

spec = importlib.util.spec_from_file_location(
    "aiopslab.orchestrator.variant_generator",
    REPO_ROOT / "aiopslab" / "orchestrator" / "variant_generator.py",
)
variant_module = importlib.util.module_from_spec(spec)
sys.modules["aiopslab.orchestrator.variant_generator"] = variant_module
assert spec.loader is not None
spec.loader.exec_module(variant_module)

from aiopslab.orchestrator.variant_generator import (  # type: ignore  # noqa: E402
    CompositeVariantGenerator,
    VariantGenerator,
)


class _StaticVariantGenerator(VariantGenerator):
    """A deterministic generator that yields predefined updates."""

    def __init__(self, base_config: Dict[str, Any], updates: List[Dict[str, Any]]):
        super().__init__(base_config)
        self._updates = [update.copy() for update in updates]

    def generate_variants(self, num_variants: int = 3) -> List[Dict[str, Any]]:
        if not self._updates:
            return []

        count = min(num_variants, len(self._updates))
        return [self._updates[i].copy() for i in range(count)]


class _DummyVariantTask:
    """Minimal task implementation for exercising variant reset logic."""

    def __init__(self, variant_generator: VariantGenerator, base_config: Dict[str, Any]):
        self.variant_generator = variant_generator
        for key, value in base_config.items():
            setattr(self, key, value)

    def get_next_variant(self) -> Dict[str, Any]:
        return self.variant_generator.get_next_variant()

    def apply_variant(self, variant_config: Dict[str, Any]) -> None:
        for key, value in variant_config.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def reset_to_base(self) -> None:
        for key, value in self.variant_generator.base_config.items():
            if hasattr(self, key):
                setattr(self, key, value)


def test_composite_generator_reset_to_base_restores_task_attributes() -> None:
    base_config: Dict[str, Any] = {
        "faulty_service": "catalog",
        "wrong_port": 8080,
        "extra": False,
        "namespace": "default",
    }

    service_gen = _StaticVariantGenerator(base_config, [{"faulty_service": "payments"}])
    port_gen = _StaticVariantGenerator(base_config, [{"wrong_port": 9091, "extra": True}])

    generator = CompositeVariantGenerator(base_config, [service_gen, port_gen])
    task = _DummyVariantTask(generator, base_config)

    variant = task.get_next_variant()

    assert variant["faulty_service"] == "payments"
    assert variant["wrong_port"] == 9091
    assert variant["extra"] is True
    assert variant["namespace"] == "default"

    task.apply_variant(variant)
    assert task.faulty_service == "payments"
    assert task.wrong_port == 9091
    assert task.extra is True
    assert task.namespace == "default"

    task.reset_to_base()
    assert task.faulty_service == base_config["faulty_service"]
    assert task.wrong_port == base_config["wrong_port"]
    assert task.extra is False
    assert task.namespace == "default"
