# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Inject faults at the application layer: Code, MongoDB, Redis, etc."""

import time
from aiopslab.generators.fault.base import FaultInjector
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.helm import Helm
from aiopslab.paths import TARGET_MICROSERVICES


class ApplicationFaultInjector(FaultInjector):
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.kubectl = KubeCtl()
        self.mongo_service_pod_map = {
            "mongodb-rate": "rate",
            "mongodb-geo": "geo",
            "url-shorten-mongodb": "url-shorten-service",
        }

    def delete_service_pods(self, target_service_pods: list[str]):
        """Kill the corresponding service pod to enforce the fault."""
        for pod in target_service_pods:
            delete_pod_command = f"kubectl delete pod {pod} -n {self.namespace}"
            delete_result = self.kubectl.exec_command(delete_pod_command)
            print(f"Deleted service pod {pod} to enforce the fault: {delete_result}")

    ############# FAULT LIBRARY ################
    # A.1 - revoke_auth: Revoke admin privileges in MongoDB - Auth
    def inject_revoke_auth(self, microservices: list[str]):
        """Inject a fault to revoke admin privileges in MongoDB."""
        print(f"Microservices to inject: {microservices}")
        target_services = ["mongodb-rate", "mongodb-geo"]
        for service in target_services:
            if service in microservices:
                pods = self.kubectl.list_pods(self.namespace)
                # print(pods)
                target_mongo_pods = [
                    pod.metadata.name
                    for pod in pods.items
                    if service in pod.metadata.name
                ]
                print(f"Target MongoDB Pods: {target_mongo_pods}")

                # Find the corresponding service pod
                target_service_pods = [
                    pod.metadata.name
                    for pod in pods.items
                    if self.mongo_service_pod_map[service] in pod.metadata.name
                    and "mongodb-" not in pod.metadata.name
                ]
                print(f"Target Service Pods: {target_service_pods}")

                for pod in target_mongo_pods:
                    if service == "mongodb-rate":
                        revoke_command = f"kubectl exec -it {pod} -n {self.namespace} -- /bin/bash /scripts/revoke-admin-rate-mongo.sh"
                    elif service == "mongodb-geo":
                        revoke_command = f"kubectl exec -it {pod} -n {self.namespace} -- /bin/bash /scripts/revoke-admin-geo-mongo.sh"
                    result = self.kubectl.exec_command(revoke_command)
                    print(f"Injection result for {service}: {result}")

                self.delete_service_pods(target_service_pods)
                time.sleep(3)

    def recover_revoke_auth(self, microservices: list[str]):
        target_services = ["mongodb-rate", "mongodb-geo"]
        for service in target_services:
            print(f"Microservices to recover: {microservices}")
            if service in microservices:
                pods = self.kubectl.list_pods(self.namespace)
                target_mongo_pods = [
                    pod.metadata.name
                    for pod in pods.items
                    if service in pod.metadata.name
                ]
                print(f"Target MongoDB Pods for recovery: {target_mongo_pods}")

                # Find the corresponding service pod
                target_service_pods = [
                    pod.metadata.name
                    for pod in pods.items
                    if self.mongo_service_pod_map[service] in pod.metadata.name
                ]
                for pod in target_mongo_pods:
                    if service == "mongodb-rate":
                        recover_command = f"kubectl exec -it {pod} -n {self.namespace} -- /bin/bash /scripts/revoke-mitigate-admin-rate-mongo.sh"
                    elif service == "mongodb-geo":
                        recover_command = f"kubectl exec -it {pod} -n {self.namespace} -- /bin/bash /scripts/revoke-mitigate-admin-geo-mongo.sh"
                    result = self.kubectl.exec_command(recover_command)
                    print(f"Recovery result for {service}: {result}")

                self.delete_service_pods(target_service_pods)

    # A.2 - storage_user_unregistered: User not registered in MongoDB - Storage/Net
    def inject_storage_user_unregistered(self, microservices: list[str]):
        """Inject a fault to create an unregistered user in MongoDB."""
        target_services = ["mongodb-rate", "mongodb-geo"]
        for service in target_services:
            if service in microservices:
                pods = self.kubectl.list_pods(self.namespace)
                target_mongo_pods = [
                    pod.metadata.name
                    for pod in pods.items
                    if service in pod.metadata.name
                ]
                print(f"Target MongoDB Pods: {target_mongo_pods}")

                target_service_pods = [
                    pod.metadata.name
                    for pod in pods.items
                    if pod.metadata.name.startswith(self.mongo_service_pod_map[service])
                ]
                for pod in target_mongo_pods:
                    revoke_command = f"kubectl exec -it {pod} -n {self.namespace} -- /bin/bash /scripts/remove-admin-mongo.sh"
                    result = self.kubectl.exec_command(revoke_command)
                    print(f"Injection result for {service}: {result}")

                self.delete_service_pods(target_service_pods)

    def recover_storage_user_unregistered(self, microservices: list[str]):
        target_services = ["mongodb-rate", "mongodb-geo"]
        for service in target_services:
            if service in microservices:
                pods = self.kubectl.list_pods(self.namespace)
                target_mongo_pods = [
                    pod.metadata.name
                    for pod in pods.items
                    if service in pod.metadata.name
                ]
                print(f"Target MongoDB Pods: {target_mongo_pods}")

                target_service_pods = [
                    pod.metadata.name
                    for pod in pods.items
                    if pod.metadata.name.startswith(self.mongo_service_pod_map[service])
                ]
                for pod in target_mongo_pods:
                    if service == "mongodb-rate":
                        revoke_command = f"kubectl exec -it {pod} -n {self.namespace} -- /bin/bash /scripts/remove-mitigate-admin-rate-mongo.sh"
                    elif service == "mongodb-geo":
                        revoke_command = f"kubectl exec -it {pod} -n {self.namespace} -- /bin/bash /scripts/remove-mitigate-admin-geo-mongo.sh"
                    result = self.kubectl.exec_command(revoke_command)
                    print(f"Recovery result for {service}: {result}")

                self.delete_service_pods(target_service_pods)

    # A.3 - misconfig_app: pull the buggy config of the application image - Misconfig
    def inject_misconfig_app(self, microservices: list[str]):
        """Inject a fault by pulling a buggy config of the application image.

        NOTE: currently only the geo microservice has a buggy image.
        """
        for service in microservices:
            # Get the deployment associated with the service
            deployment = self.kubectl.get_deployment(service, self.namespace)
            if deployment:
                # Modify the image to use the buggy image
                for container in deployment.spec.template.spec.containers:
                    if container.name == f"hotel-reserv-{service}":
                        container.image = "yinfangchen/geo:app3"
                self.kubectl.update_deployment(service, self.namespace, deployment)
                time.sleep(10)

    def recover_misconfig_app(self, microservices: list[str]):
        for service in microservices:
            deployment = self.kubectl.get_deployment(service, self.namespace)
            if deployment:
                for container in deployment.spec.template.spec.containers:
                    if container.name == f"hotel-reserv-{service}":
                        container.image = "yinfangchen/hotelreservation:latest"
                self.kubectl.update_deployment(service, self.namespace, deployment)

    # A.4 - auth_miss_mongodb: Authentication missing for MongoDB - Auth
    def inject_auth_miss_mongodb(self, microservices: list[str]):
        """Inject a fault to require TLS for MongoDB, breaking app connections."""
        for service in microservices:
            set_values = {
                "url-shorten-mongodb.tls.mode": "requireTLS",
                "url-shorten-mongodb.tls.certificateKeyFile": "/etc/tls/tls.pem",
                "url-shorten-mongodb.tls.CAFile": "/etc/tls/ca.crt",
            }

            helm_args = {
                "release_name": "social-network",
                "chart_path": TARGET_MICROSERVICES
                / "socialNetwork/helm-chart/socialnetwork/",
                "namespace": self.namespace,
                "values_file": TARGET_MICROSERVICES
                / "socialNetwork/helm-chart/socialnetwork/values.yaml",
                "set_values": set_values,
            }

            Helm.upgrade(**helm_args)

            pods = self.kubectl.list_pods(self.namespace)
            service_label = self.mongo_service_pod_map.get(service)
            if not service_label:
                print(
                    f"Unknown service mapping for {service}; skipping pod restart enforcement."
                )
                continue

            target_service_pods = [
                pod.metadata.name
                for pod in pods.items
                if service_label in pod.metadata.name
            ]
            print(f"Target Service Pods: {target_service_pods}")
            self.delete_service_pods(target_service_pods)

            self.kubectl.exec_command(
                f"kubectl rollout restart deployment {service} -n {self.namespace}"
            )

    def recover_auth_miss_mongodb(self, microservices: list[str]):
        for service in microservices:
            if service != "url-shorten-mongodb":
                print(
                    f"Skipping auth_miss_mongodb recovery for non-supported service {service}."
                )
                continue
            set_values = {
                "url-shorten-mongodb.tls.mode": "disabled",
                "url-shorten-mongodb.tls.certificateKeyFile": "",
                "url-shorten-mongodb.tls.CAFile": "",
            }

            helm_args = {
                "release_name": "social-network",
                "chart_path": TARGET_MICROSERVICES
                / "socialNetwork/helm-chart/socialnetwork/",
                "namespace": self.namespace,
                "values_file": TARGET_MICROSERVICES
                / "socialNetwork/helm-chart/socialnetwork/values.yaml",
                "set_values": set_values,
            }

            Helm.upgrade(**helm_args)

            pods = self.kubectl.list_pods(self.namespace)
            service_label = self.mongo_service_pod_map.get(service)
            if not service_label:
                print(
                    f"Unknown service mapping for {service}; skipping pod restart enforcement."
                )
                continue

            target_service_pods = [
                pod.metadata.name
                for pod in pods.items
                if service_label in pod.metadata.name
            ]
            print(f"Target Service Pods: {target_service_pods}")

            self.delete_service_pods(target_service_pods)
            self.kubectl.exec_command(
                f"kubectl rollout restart deployment {service} -n {self.namespace}"
            )


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
