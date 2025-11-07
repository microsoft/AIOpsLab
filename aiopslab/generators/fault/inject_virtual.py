# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Inject faults at the virtualization layer: K8S, Docker, etc."""

import yaml
import time

from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.dock import Docker
from aiopslab.generators.fault.base import FaultInjector
from aiopslab.service.apps.base import Application


class VirtualizationFaultInjector(FaultInjector):
    def __init__(self, namespace: str):
        super().__init__(namespace)
        self.namespace = namespace
        self.kubectl = KubeCtl()
        self.docker = Docker()

    def delete_service_pods(self, target_service_pods: list[str]):
        """Kill the corresponding service pod to enforce the fault."""
        for pod in target_service_pods:
            delete_pod_command = f"kubectl delete pod {pod} -n {self.namespace}"
            delete_result = self.kubectl.exec_command(delete_pod_command)
            print(f"Deleted service pod {pod} to enforce the fault: {delete_result}")

    ############# FAULT LIBRARY ################

    # V.1 - misconfig_k8s: Misconfigure service port in Kubernetes - Misconfig
    def inject_misconfig_k8s(self, microservices: list[str]):
        """Inject a fault to misconfigure service's target port in Kubernetes."""
        for service in microservices:
            service_config = self._modify_target_port_config(
                from_port=9090,
                to_port=9999,
                configs=self.kubectl.get_service_json(service, self.testbed),
            )

            print(f"Misconfig fault for service: {service} | namespace: {self.testbed}")
            self.kubectl.patch_service(service, self.testbed, service_config)

    def recover_misconfig_k8s(self, microservices: list[str]):
        for service in microservices:
            service_config = self._modify_target_port_config(
                from_port=9999,
                to_port=9090,
                configs=self.kubectl.get_service_json(service, self.testbed),
            )

            print(f"Recovering for service: {service} | namespace: {self.testbed}")
            self.kubectl.patch_service(service, self.testbed, service_config)

    # V.3 - scale_pods_to_zero: Scale pods to zero - Deploy/Operation
    def inject_scale_pods_to_zero(self, microservices: list[str]):
        """Inject a fault to scale pods to zero for a service."""
        for service in microservices:
            self.kubectl.exec_command(
                f"kubectl scale deployment {service} --replicas=0 -n {self.namespace}"
            )
            print(
                f"Scaled deployment {service} to 0 replicas | namespace: {self.namespace}"
            )

    def recover_scale_pods_to_zero(self, microservices: list[str]):
        for service in microservices:
            self.kubectl.exec_command(
                f"kubectl scale deployment {service} --replicas=1 -n {self.namespace}"
            )
            print(
                f"Scaled deployment {service} back to 1 replica | namespace: {self.namespace}"
            )

    # V.4 - assign_to_non_existent_node: Assign to non-existent or NotReady node - Dependency
    def inject_assign_to_non_existent_node(self, microservices: list[str]):
        """Inject a fault to assign a service to a non-existent or NotReady node."""
        non_existent_node_name = "extra-node"
        for service in microservices:
            deployment_yaml = self._get_deployment_yaml(service)
            deployment_yaml["spec"]["template"]["spec"]["nodeSelector"] = {
                "kubernetes.io/hostname": non_existent_node_name
            }

            # Write the modified YAML to a temporary file
            modified_yaml_path = self._write_yaml_to_file(service, deployment_yaml)

            delete_command = f"kubectl delete deployment {service} -n {self.namespace}"
            self.kubectl.exec_command(delete_command)

            apply_command = f"kubectl apply -f {modified_yaml_path} -n {self.namespace}"
            self.kubectl.exec_command(apply_command)
            print(f"Redeployed {service} to node {non_existent_node_name}.")

    def recover_assign_to_non_existent_node(self, microservices: list[str]):
        for service in microservices:
            deployment_yaml = self._get_deployment_yaml(service)
            if "nodeSelector" in deployment_yaml["spec"]["template"]["spec"]:
                del deployment_yaml["spec"]["template"]["spec"]["nodeSelector"]

            modified_yaml_path = self._write_yaml_to_file(service, deployment_yaml)

            delete_command = f"kubectl delete deployment {service} -n {self.namespace}"
            self.kubectl.exec_command(delete_command)

            apply_command = f"kubectl apply -f {modified_yaml_path} -n {self.namespace}"
            self.kubectl.exec_command(apply_command)
            print(f"Removed nodeSelector for service {service} and redeployed.")

    # V.5 - redeploy without deleting the PV - only for HotelReservation
    def inject_redeploy_without_pv(self, app: Application):
        """Inject a fault to delete the namespace without deleting the PV."""
        self.kubectl.delete_namespace(self.namespace)
        print(f"Deleting namespace {self.namespace} without deleting the PV.")
        time.sleep(15)
        print(f"Redeploying {self.namespace}.")
        app = type(app)()
        app.deploy_without_wait()

    def recover_redepoly_without_pv(self, app: Application):
        app.cleanup()
        # pass

    # V.6 - wrong binary usage incident
    def inject_wrong_bin_usage(self, microservices: list[str]):
        """Inject a fault to use the wrong binary of a service."""
        for service in microservices:
            deployment_yaml = self._get_deployment_yaml(service)

            # Modify the deployment YAML to use the 'geo' binary instead of the 'profile' binary
            containers = deployment_yaml["spec"]["template"]["spec"]["containers"]
            for container in containers:
                if "command" in container and "profile" in container["command"]:
                    print(
                        f"Changing binary for container {container['name']} from 'profile' to 'geo'."
                    )
                    container["command"] = ["geo"]  # Replace 'profile' with 'geo'

            modified_yaml_path = self._write_yaml_to_file(service, deployment_yaml)

            # Delete the deployment and re-apply
            delete_command = f"kubectl delete deployment {service} -n {self.namespace}"
            apply_command = f"kubectl apply -f {modified_yaml_path} -n {self.namespace}"
            self.kubectl.exec_command(delete_command)
            self.kubectl.exec_command(apply_command)

            print(f"Injected wrong binary usage fault for service: {service}")

    def recover_wrong_bin_usage(self, microservices: list[str]):
        for service in microservices:
            deployment_yaml = self._get_deployment_yaml(service)

            containers = deployment_yaml["spec"]["template"]["spec"]["containers"]
            for container in containers:
                if "command" in container and "geo" in container["command"]:
                    print(
                        f"Reverting binary for container {container['name']} from 'geo' to 'profile'."
                    )
                    container["command"] = [
                        "profile"
                    ]  # Restore 'geo' back to 'profile'

            modified_yaml_path = self._write_yaml_to_file(service, deployment_yaml)

            delete_command = f"kubectl delete deployment {service} -n {self.namespace}"
            apply_command = f"kubectl apply -f {modified_yaml_path} -n {self.namespace}"
            self.kubectl.exec_command(delete_command)
            self.kubectl.exec_command(apply_command)

            print(f"Recovered from wrong binary usage fault for service: {service}")
            
    def inject_container_stop(self, microservices: list[str]):
        """Inject a fault to stop a container."""
        for service in microservices:
            self.docker.get_container(service).stop()
            print(f"Stopped container {service}.")
            
            print("Waiting for faults to propagate...")
            time.sleep(15)
            print("Faults propagated.") 
    
    def recover_container_stop(self, microservices: list[str]):
        for service in microservices:
            self.docker.get_container(service).start()
            print(f"Started container {service}.")
            
    def inject_model_misconfig(self, microservices: list[str]):
        """Inject a fault to misconfigure the model in the Flower application."""
        for service in microservices:
            command = f""" docker exec -it {service} sh -c "sed -i '24s/84/80/' /app/.flwr/apps/*/task.py" """
            self.docker.exec_command(command)
            print(f"Changed model configuration for service: {service}")
            
    def recover_model_misconfig(self, microservices: list[str]):
        for service in microservices:
            command = f""" docker exec -it {service} sh -c "sed -i '24s/80/84/' /app/.flwr/apps/*/task.py" """
            self.docker.exec_command(command)
            print(f"Recovered model configuration for service: {service}")
            
    ############# HELPER FUNCTIONS ################
    def _wait_for_pods_ready(self, microservices: list[str], timeout: int = 30):
        for service in microservices:
            command = f"kubectl wait --for=condition=ready pod -l app={service} -n {self.namespace} --timeout={timeout}s"
            result = self.kubectl.exec_command(command)
            print(f"Wait result for {service}: {result}")

    def _modify_target_port_config(self, from_port: int, to_port: int, configs: dict):
        for port in configs["spec"]["ports"]:
            if port.get("targetPort") == from_port:
                port["targetPort"] = to_port

        return configs

    def _get_values_yaml(self, service_name: str):
        kubectl = KubeCtl()
        values_yaml = kubectl.exec_command(
            f"kubectl get configmap {service_name} -n {self.testbed} -o yaml"
        )
        return yaml.safe_load(values_yaml)

    def _enable_tls(self, values_yaml: dict):
        values_yaml["net"] = {
            "tls": {
                "mode": "requireTLS",
                "certificateKeyFile": "/etc/tls/tls.pem",
                "CAFile": "/etc/tls/ca.crt",
            }
        }
        return yaml.dump(values_yaml)

    def _apply_modified_yaml(self, service_name: str, modified_yaml: str):
        modified_yaml_path = f"/tmp/{service_name}-values.yaml"
        with open(modified_yaml_path, "w") as f:
            f.write(modified_yaml)

        kubectl = KubeCtl()
        kubectl.exec_command(
            f"kubectl create configmap {service_name} -n {self.testbed} --from-file=values.yaml={modified_yaml_path} --dry-run=client -o yaml | kubectl apply -f -"
        )
        kubectl.exec_command(
            f"kubectl rollout restart deployment {service_name} -n {self.testbed}"
        )

    def _get_deployment_yaml(self, service_name: str):
        deployment_yaml = self.kubectl.exec_command(
            f"kubectl get deployment {service_name} -n {self.namespace} -o yaml"
        )
        return yaml.safe_load(deployment_yaml)

    def _change_node_selector(self, deployment_yaml: dict, node_name: str):
        if "spec" in deployment_yaml and "template" in deployment_yaml["spec"]:
            deployment_yaml["spec"]["template"]["spec"]["nodeSelector"] = {
                "kubernetes.io/hostname": node_name
            }
        return yaml.dump(deployment_yaml)

    def _write_yaml_to_file(self, service_name: str, yaml_content: dict):
        """Helper function to write YAML content to a temporary file."""
        import yaml

        file_path = f"/tmp/{service_name}_modified.yaml"
        with open(file_path, "w") as file:
            yaml.dump(yaml_content, file)
        return file_path


if __name__ == "__main__":
    namespace = "test-social-network"
    microservices = ["user-service"]
    fault_type = "scale_pods_to_zero"
    print("Start injection ...")
    injector = VirtualizationFaultInjector(namespace)
    injector._inject(fault_type, microservices)
