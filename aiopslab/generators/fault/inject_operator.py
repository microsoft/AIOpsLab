import copy
import time
from typing import Any, Dict

import yaml

from aiopslab.service.kubectl import KubeCtl
from aiopslab.generators.fault.base import FaultInjector


class K8SOperatorFaultInjector(FaultInjector):
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.kubectl = KubeCtl()
        self.kubectl.create_namespace_if_not_exist(namespace)

    def _apply_yaml(self, cr_name: str, cr_yaml: dict):
        yaml_path = f"/tmp/{cr_name}.yaml"
        with open(yaml_path, "w") as file:
            yaml.dump(cr_yaml, file)

        command = f"kubectl apply -f {yaml_path} -n {self.namespace}"
        result = self.kubectl.exec_command(command)
        print(f"Injected {cr_name}: {result}")

    def _delete_yaml(self, cr_name: str):
        yaml_path = f"/tmp/{cr_name}.yaml"
        command = f"kubectl delete -f {yaml_path} -n {self.namespace}"
        result = self.kubectl.exec_command(command)
        print(f"Recovered from misconfiguration {cr_name}: {result}")

    def inject_overload_replicas(self, variant: Dict[str, Any]):
        """Inject a TiDB custom resource with variant-controlled replicas."""
        replica_overrides = self._replica_overrides(variant)
        return self._apply_variant("overload-tidbcluster", variant, replica_overrides)

    def recover_overload_replicas(self, variant: Dict[str, Any]):
        self.recover_fault(self._variant_id(variant, "overload-tidbcluster"))

    def inject_invalid_affinity_toleration(self, variant: Dict[str, Any]):
        """Inject a TiDB custom resource with variant-controlled tolerations."""
        tolerations = variant.get("tolerations")
        overrides = {}
        if tolerations is not None:
            overrides = {"spec": {"tidb": {"tolerations": tolerations}}}
        return self._apply_variant("affinity-toleration-fault", variant, overrides)

    def recover_invalid_affinity_toleration(self, variant: Dict[str, Any]):
        self.recover_fault(self._variant_id(variant, "affinity-toleration-fault"))

    def inject_security_context_fault(self, variant: Dict[str, Any]):
        """Inject a TiDB custom resource with variant-controlled pod security context."""
        pod_security_context = variant.get("pod_security_context")
        overrides = {}
        if pod_security_context is not None:
            overrides = {"spec": {"tidb": {"podSecurityContext": pod_security_context}}}
        return self._apply_variant("security-context-fault", variant, overrides)

    def recover_security_context_fault(self, variant: Dict[str, Any]):
        self.recover_fault(self._variant_id(variant, "security-context-fault"))

    def inject_wrong_update_strategy(self, variant: Dict[str, Any]):
        """Inject a TiDB custom resource with variant-controlled update strategy."""
        update_strategy = variant.get("update_strategy")
        overrides = {}
        if update_strategy is not None:
            overrides = {
                "spec": {"tidb": {"statefulSetUpdateStrategy": update_strategy}}
            }
        return self._apply_variant(
            "deployment-update-strategy-fault", variant, overrides
        )

    def recover_wrong_update_strategy(self, variant: Dict[str, Any]):
        self.recover_fault(self._variant_id(variant, "deployment-update-strategy-fault"))

    def inject_non_existent_storage(self, variant: Dict[str, Any]):
        """Inject a TiDB custom resource with variant-controlled storage class."""
        storage_class = variant.get("storage_class")
        overrides = {}
        if storage_class is not None:
            overrides = {"spec": {"pd": {"storageClassName": storage_class}}}
        return self._apply_variant("non-existent-storage-fault", variant, overrides)

    def recover_non_existent_storage(self, variant: Dict[str, Any]):
        self.recover_fault(self._variant_id(variant, "non-existent-storage-fault"))

    def recover_fault(self, cr_name: str):
        self._delete_yaml(cr_name)

    def _apply_variant(
        self,
        default_id: str,
        variant: Dict[str, Any],
        overrides: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        variant_id = self._variant_id(variant, default_id)
        merged_overrides = self._collect_variant_overrides(variant, overrides)
        cr_yaml = self._build_tidbcluster_yaml(variant_id, merged_overrides)
        self._apply_yaml(variant_id, cr_yaml)
        return cr_yaml

    def _variant_id(self, variant: Dict[str, Any], default_id: str) -> str:
        return str(variant.get("id") or variant.get("variant_id") or default_id)

    def _collect_variant_overrides(
        self,
        variant: Dict[str, Any],
        overrides: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        if "overrides" in variant:
            self._deep_merge(merged, copy.deepcopy(variant["overrides"]))
        for key in ("metadata", "spec"):
            if key in variant:
                self._deep_merge(merged, {key: copy.deepcopy(variant[key])})
        if overrides:
            self._deep_merge(merged, copy.deepcopy(overrides))
        return merged

    def _build_tidbcluster_yaml(
        self, variant_id: str, overrides: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        base = self._tidbcluster_template()
        if overrides:
            self._deep_merge(base, copy.deepcopy(overrides))
        metadata = base.setdefault("metadata", {})
        annotations = metadata.setdefault("annotations", {})
        annotations["fault.aiopslab/variant-id"] = variant_id
        return base

    def _tidbcluster_template(self) -> Dict[str, Any]:
        return {
            "apiVersion": "pingcap.com/v1alpha1",
            "kind": "TidbCluster",
            "metadata": {"name": "basic", "namespace": self.namespace},
            "spec": {
                "version": "v3.0.8",
                "timezone": "UTC",
                "pvReclaimPolicy": "Delete",
                "pd": {
                    "baseImage": "pingcap/pd",
                    "replicas": 3,
                    "requests": {"storage": "1Gi"},
                    "config": {},
                },
                "tikv": {
                    "baseImage": "pingcap/tikv",
                    "replicas": 3,
                    "requests": {"storage": "1Gi"},
                    "config": {},
                },
                "tidb": {
                    "baseImage": "pingcap/tidb",
                    "replicas": 2,
                    "service": {"type": "ClusterIP"},
                    "config": {},
                },
            },
        }

    def _replica_overrides(self, variant: Dict[str, Any]) -> Dict[str, Any]:
        replicas = variant.get("replicas")
        spec_overrides: Dict[str, Any] = {}
        if isinstance(replicas, dict):
            for component, value in replicas.items():
                spec_overrides.setdefault(component, {})["replicas"] = value
        elif replicas is not None:
            spec_overrides.setdefault("tidb", {})["replicas"] = replicas
        return {"spec": spec_overrides} if spec_overrides else {}

    def _deep_merge(self, target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in updates.items():
            if (
                isinstance(value, dict)
                and isinstance(target.get(key), dict)
            ):
                self._deep_merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)
        return target


if __name__ == "__main__":
    namespace = "tidb-cluster"
    tidb_fault_injector = K8SOperatorFaultInjector(namespace)

    variant = {"id": "overload-tidbcluster", "replicas": 100000}
    tidb_fault_injector.inject_overload_replicas(variant)
    time.sleep(10)
    tidb_fault_injector.recover_overload_replicas(variant)
