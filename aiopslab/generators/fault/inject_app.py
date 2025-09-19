# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Inject faults at the application layer: Code, MongoDB, Redis, etc."""

import copy
import time
from collections.abc import Iterable
from typing import Any, Dict, List

from aiopslab.generators.fault.base import FaultInjector
from aiopslab.service.kubectl import KubeCtl


class ApplicationFaultInjector(FaultInjector):
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.kubectl = KubeCtl()
        self.mongo_service_pod_map = {"mongodb-rate": "rate", "mongodb-geo": "geo"}
        self.revoke_defaults = {
            "mongodb-rate": {
                "service": "mongodb-rate",
                "target_service": "rate",
                "inject_script": "/scripts/revoke-admin-rate-mongo.sh",
                "recover_script": "/scripts/revoke-mitigate-admin-rate-mongo.sh",
                "service_pod_selector": "rate",
            },
            "mongodb-geo": {
                "service": "mongodb-geo",
                "target_service": "geo",
                "inject_script": "/scripts/revoke-admin-geo-mongo.sh",
                "recover_script": "/scripts/revoke-mitigate-admin-geo-mongo.sh",
                "service_pod_selector": "geo",
            },
        }
        self.storage_defaults = {
            "mongodb-rate": {
                "service": "mongodb-rate",
                "target_service": "rate",
                "inject_script": "/scripts/remove-admin-mongo.sh",
                "recover_script": "/scripts/remove-mitigate-admin-rate-mongo.sh",
                "service_pod_selector": "rate",
            },
            "mongodb-geo": {
                "service": "mongodb-geo",
                "target_service": "geo",
                "inject_script": "/scripts/remove-admin-mongo.sh",
                "recover_script": "/scripts/remove-mitigate-admin-geo-mongo.sh",
                "service_pod_selector": "geo",
            },
        }
        self.image_defaults = {
            "geo": {
                "service": "geo",
                "container": "hotel-reserv-geo",
                "inject_image": "yinfangchen/geo:app3",
                "recover_image": "yinfangchen/hotelreservation:latest",
            }
        }

    def delete_service_pods(self, target_service_pods: list[str]):
        """Kill the corresponding service pod to enforce the fault."""
        for pod in target_service_pods:
            delete_pod_command = f"kubectl delete pod {pod} -n {self.namespace}"
            delete_result = self.kubectl.exec_command(delete_pod_command)
            print(f"Deleted service pod {pod} to enforce the fault: {delete_result}")

    ############# FAULT LIBRARY ################
    # A.1 - revoke_auth: Revoke admin privileges in MongoDB - Auth
    def inject_revoke_auth(self, microservices: Iterable[Dict[str, Any] | str]):
        """Inject a fault to revoke admin privileges in MongoDB."""
        print(f"Microservices to inject: {microservices}")
        variants = self._normalize_variants(microservices, self.revoke_defaults)
        for variant in variants:
            service = variant["service"]
            pods = self.kubectl.list_pods(self.namespace)
            target_mongo_pods = self._select_pods(
                pods.items, variant.get("mongo_selector", service)
            )
            print(f"Target MongoDB Pods: {target_mongo_pods}")

            target_service_pods = self._select_pods(
                pods.items,
                variant.get(
                    "service_pod_selector",
                    self.mongo_service_pod_map.get(service, ""),
                ),
                exclude_prefix="mongodb-",
            )
            print(f"Target Service Pods: {target_service_pods}")

            for pod in target_mongo_pods:
                script_path = variant.get("inject_script")
                if not script_path:
                    continue
                revoke_command = (
                    f"kubectl exec -it {pod} -n {self.namespace} "
                    f"-- /bin/bash {script_path}"
                )
                result = self.kubectl.exec_command(revoke_command)
                print(f"Injection result for {service}: {result}")

            self.delete_service_pods(target_service_pods)
            time.sleep(3)

    def recover_revoke_auth(self, microservices: Iterable[Dict[str, Any] | str]):
        variants = self._normalize_variants(microservices, self.revoke_defaults)
        for variant in variants:
            service = variant["service"]
            pods = self.kubectl.list_pods(self.namespace)
            target_mongo_pods = self._select_pods(
                pods.items, variant.get("mongo_selector", service)
            )
            print(f"Target MongoDB Pods for recovery: {target_mongo_pods}")

            target_service_pods = self._select_pods(
                pods.items,
                variant.get(
                    "service_pod_selector",
                    self.mongo_service_pod_map.get(service, ""),
                ),
            )
            for pod in target_mongo_pods:
                script_path = variant.get("recover_script")
                if not script_path:
                    continue
                recover_command = (
                    f"kubectl exec -it {pod} -n {self.namespace} "
                    f"-- /bin/bash {script_path}"
                )
                result = self.kubectl.exec_command(recover_command)
                print(f"Recovery result for {service}: {result}")

            self.delete_service_pods(target_service_pods)

    # A.2 - storage_user_unregistered: User not registered in MongoDB - Storage/Net
    def inject_storage_user_unregistered(
        self, microservices: Iterable[Dict[str, Any] | str]
    ):
        """Inject a fault to create an unregistered user in MongoDB."""
        variants = self._normalize_variants(microservices, self.storage_defaults)
        for variant in variants:
            service = variant["service"]
            pods = self.kubectl.list_pods(self.namespace)
            target_mongo_pods = self._select_pods(
                pods.items, variant.get("mongo_selector", service)
            )
            print(f"Target MongoDB Pods: {target_mongo_pods}")

            target_service_pods = self._select_pods(
                pods.items,
                variant.get(
                    "service_pod_selector",
                    self.mongo_service_pod_map.get(service, ""),
                ),
            )
            for pod in target_mongo_pods:
                script_path = variant.get("inject_script")
                if not script_path:
                    continue
                revoke_command = (
                    f"kubectl exec -it {pod} -n {self.namespace} "
                    f"-- /bin/bash {script_path}"
                )
                result = self.kubectl.exec_command(revoke_command)
                print(f"Injection result for {service}: {result}")

            self.delete_service_pods(target_service_pods)

    def recover_storage_user_unregistered(
        self, microservices: Iterable[Dict[str, Any] | str]
    ):
        variants = self._normalize_variants(microservices, self.storage_defaults)
        for variant in variants:
            service = variant["service"]
            pods = self.kubectl.list_pods(self.namespace)
            target_mongo_pods = self._select_pods(
                pods.items, variant.get("mongo_selector", service)
            )
            print(f"Target MongoDB Pods: {target_mongo_pods}")

            target_service_pods = self._select_pods(
                pods.items,
                variant.get(
                    "service_pod_selector",
                    self.mongo_service_pod_map.get(service, ""),
                ),
            )
            for pod in target_mongo_pods:
                script_path = variant.get("recover_script")
                if not script_path:
                    continue
                revoke_command = (
                    f"kubectl exec -it {pod} -n {self.namespace} "
                    f"-- /bin/bash {script_path}"
                )
                result = self.kubectl.exec_command(revoke_command)
                print(f"Recovery result for {service}: {result}")

            self.delete_service_pods(target_service_pods)

    # A.3 - misconfig_app: pull the buggy config of the application image - Misconfig
    def inject_misconfig_app(self, microservices: Iterable[Dict[str, Any] | str]):
        """Inject a fault by pulling a buggy config of the application image.

        NOTE: currently only the geo microservice has a buggy image.
        """
        variants = self._normalize_variants(microservices, self.image_defaults)
        for variant in variants:
            service = variant["service"]
            deployment = self.kubectl.get_deployment(service, self.namespace)
            if deployment:
                # Modify the image to use the buggy image
                for container in deployment.spec.template.spec.containers:
                    target_container = variant.get(
                        "container", f"hotel-reserv-{service}"
                    )
                    if container.name == target_container:
                        container.image = variant.get("inject_image", container.image)
                self.kubectl.update_deployment(service, self.namespace, deployment)
                time.sleep(10)

    def recover_misconfig_app(self, microservices: Iterable[Dict[str, Any] | str]):
        variants = self._normalize_variants(microservices, self.image_defaults)
        for variant in variants:
            service = variant["service"]
            deployment = self.kubectl.get_deployment(service, self.namespace)
            if deployment:
                for container in deployment.spec.template.spec.containers:
                    target_container = variant.get(
                        "container", f"hotel-reserv-{service}"
                    )
                    if container.name == target_container:
                        container.image = variant.get(
                            "recover_image", container.image
                        )
                self.kubectl.update_deployment(service, self.namespace, deployment)

    def _normalize_variants(
        self,
        variants: Iterable[Dict[str, Any] | str],
        defaults: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for entry in variants:
            if isinstance(entry, dict):
                data = dict(entry)
            else:
                data = {"service": entry}

            data.setdefault("service", data.get("name"))
            service_name = data.get("service")
            default_data = copy.deepcopy(defaults.get(service_name, {}))
            if service_name:
                default_data.setdefault("service", service_name)
            default_data.update(data)
            normalized.append(default_data)

        return normalized

    @staticmethod
    def _select_pods(
        pods: Iterable[Any], selector: str | None, exclude_prefix: str | None = None
    ) -> List[str]:
        if not selector:
            return []
        selected = []
        for pod in pods:
            metadata = getattr(pod, "metadata", None)
            if metadata is not None and hasattr(metadata, "name"):
                name = metadata.name
            else:
                name = getattr(pod, "name", str(pod))
            if exclude_prefix and name.startswith(exclude_prefix):
                continue
            if selector in name:
                selected.append(name)
        return selected


if __name__ == "__main__":
    namespace = "test-hotel-reservation"
    # microservices = ["geo"]
    microservices = ["mongodb-geo"]
    # fault_type = "misconfig_app"
    fault_type = "storage_user_unregistered"
    print("Start injection/recover ...")
    injector = ApplicationFaultInjector(namespace)
    # injector._inject(fault_type, microservices)
    injector._recover(fault_type, microservices)
