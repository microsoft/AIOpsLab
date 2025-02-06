import yaml
import time
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

    def inject_overload_replicas(self):
        """
        Injects a TiDB misoperation custom resource.
        The misconfiguration sets an unreasonably high number of TiDB replicas.
        """
        cr_name = "overload-tidbcluster"
        cr_yaml = {
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
                    "replicas": 100000,  # Intentional misconfiguration
                    "service": {"type": "ClusterIP"},
                    "config": {},
                },
            },
        }

        self._apply_yaml(cr_name, cr_yaml)

    def recover_overload_replicas(self):
        self.recover_fault("overload-tidbcluster")

    def inject_invalid_affinity_toleration(self):
        """
        This misoperation specifies an invalid toleration effect.
        """
        cr_name = "affinity-toleration-fault"
        cr_yaml = {
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
                    "tolerations": [
                        {
                            "key": "test-keys",
                            "operator": "Equal",
                            "value": "test-value",
                            "effect": "TAKE_SOME_EFFECT",  # Buggy: invalid toleration effect
                            "tolerationSeconds": 0,
                        }
                    ],
                },
            },
        }
        self._apply_yaml(cr_name, cr_yaml)

    def recover_invalid_affinity_toleration(self):
        self.recover_fault("affinity-toleration-fault")

    def inject_security_context_fault(self):
        """
        The fault sets an invalid runAsUser value.
        """
        cr_name = "security-context-fault"
        cr_yaml = {
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
                    "podSecurityContext": {"runAsUser": -1},  # invalid runAsUser value
                },
            },
        }
        self._apply_yaml(cr_name, cr_yaml)

    def recover_security_context_fault(self):
        self.recover_fault("security-context-fault")

    def inject_wrong_update_strategy(self):
        """
        This fault specifies an invalid update strategy.
        """
        cr_name = "deployment-update-strategy-fault"
        cr_yaml = {
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
                    "statefulSetUpdateStrategy": "SomeStrategyForUpdata",  # invalid update strategy
                },
            },
        }
        self._apply_yaml(cr_name, cr_yaml)

    def recover_wrong_update_strategy(self):
        self.recover_fault("deployment-update-strategy-fault")

    def inject_non_existent_storage(self):
        """
        This fault specifies a non-existent storage class.
        """
        cr_name = "non-existent-storage-fault"
        cr_yaml = {
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
                    "storageClassName": "ThisIsAStorageClass",  # non-existent storage class
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
        self._apply_yaml(cr_name, cr_yaml)

    def recover_non_existent_storage(self):
        self.recover_fault("non-existent-storage-fault")

    def recover_fault(self, cr_name: str):
        self._delete_yaml(cr_name)


if __name__ == "__main__":
    namespace = "tidb-cluster"
    tidb_fault_injector = K8SOperatorFaultInjector(namespace)

    tidb_fault_injector.inject_overload_replicas()
    time.sleep(10)
    tidb_fault_injector.recover_overload_replicas()
