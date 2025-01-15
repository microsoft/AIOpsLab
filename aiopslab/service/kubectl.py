# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Interface to K8S controller service."""

import json
import time
import subprocess
from rich.console import Console
from kubernetes import client, config
from kubernetes.client.rest import ApiException


class KubeCtl:
    def __init__(self):
        """Initialize the KubeCtl object and load the Kubernetes configuration."""
        config.load_kube_config()
        self.core_v1_api = client.CoreV1Api()
        self.apps_v1_api = client.AppsV1Api()

    def list_namespaces(self):
        """Return a list of all namespaces in the cluster."""
        return self.core_v1_api.list_namespace()

    def list_pods(self, namespace):
        """Return a list of all pods within a specified namespace."""
        return self.core_v1_api.list_namespaced_pod(namespace)

    def list_services(self, namespace):
        """Return a list of all services within a specified namespace."""
        return self.core_v1_api.list_namespaced_service(namespace)

    def get_cluster_ip(self, service_name, namespace):
        """Retrieve the cluster IP address of a specified service within a namespace."""
        service_info = self.core_v1_api.read_namespaced_service(service_name, namespace)
        return service_info.spec.cluster_ip  # type: ignore

    def get_pod_name(self, namespace, label_selector):
        """Get the name of the first pod in a namespace that matches a given label selector."""
        pod_info = self.core_v1_api.list_namespaced_pod(
            namespace, label_selector=label_selector
        )
        return pod_info.items[0].metadata.name

    def get_pod_logs(self, pod_name, namespace):
        """Retrieve the logs of a specified pod within a namespace."""
        return self.core_v1_api.read_namespaced_pod_log(pod_name, namespace)

    def get_service_json(self, service_name, namespace, deserialize=True):
        """Retrieve the JSON description of a specified service within a namespace."""
        command = f"kubectl get service {service_name} -n {namespace} -o json"
        result = self.exec_command(command)

        return json.loads(result) if deserialize else result

    def get_deployment(self, name: str, namespace: str):
        """Fetch the deployment configuration."""
        return self.apps_v1_api.read_namespaced_deployment(name, namespace)

    def wait_for_state(self, namespace, state, sleep=10, max_wait=300):
        """Wait for all pods in a namespace to reach a specified state, with a timeout."""

        console = Console()

        """
        OpenWhisk is where flight-ticket is deployed. It is handled differently since the setup jobs take
        a long time (~25 minutes) which is significantly longer than all of the other apps. The approach
        of waiting for the pods to be ready won't work because of the k8s jobs which report as completed.
        So, we wait for the load_generator to start running since that's when flight-ticket is fully set up.
        """
        if namespace == "openwhisk":
            console.log(
                "[bold green]Waiting for the load-generator job to start in the openwhisk namespace (this may take a while)..."
            )
            with console.status(
                "[bold green]Waiting for load-generator job..."
            ) as status:
                while True:
                    try:
                        pod_name = self.get_pod_name(
                            namespace, label_selector="job-name=load-generator"
                        )

                        pod_status = self.core_v1_api.read_namespaced_pod_status(
                            pod_name, namespace
                        ).status.phase

                        if pod_status == "Running":
                            console.log(
                                f"[bold green]Load-generator job is {pod_status}."
                            )
                            return

                    except Exception as e:
                        console.log(f"[red]Error checking load-generator status: {e}")

                    time.sleep(sleep)

        else:
            wait = 0
            with console.status("[bold green]Working on deployments...") as status:
                while wait < max_wait:
                    pod_list = self.list_pods(namespace)
                    if all(pod.status.phase == state for pod in pod_list.items):
                        break

                    time.sleep(sleep)
                    wait += sleep
                else:
                    raise Exception(f"App didn't reach the expected state: {state}")

    def update_deployment(self, name: str, namespace: str, deployment):
        """Update the deployment configuration."""
        return self.apps_v1_api.replace_namespaced_deployment(
            name, namespace, deployment
        )

    def patch_service(self, name, namespace, body):
        """Patch a Kubernetes service in a specified namespace."""
        try:
            api_response = self.core_v1_api.patch_namespaced_service(
                name, namespace, body
            )
            return api_response
        except ApiException as e:
            print(f"Exception when patching service: {e}\n")
            return None

    def create_configmap(self, name, namespace, data):
        """Create or update a configmap from a dictionary of data."""
        try:
            api_response = self.update_configmap(name, namespace, data)
            return api_response
        except ApiException as e:
            if e.status == 404:
                return self.create_new_configmap(name, namespace, data)
            else:
                print(f"Exception when updating configmap: {e}\n")
                print(f"Exception status code: {e.status}\n")
                return None

    def create_new_configmap(self, name, namespace, data):
        """Create a new configmap."""
        config_map = client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
            metadata=client.V1ObjectMeta(name=name),
            data=data,
        )
        try:
            return self.core_v1_api.create_namespaced_config_map(namespace, config_map)
        except ApiException as e:
            print(f"Exception when creating configmap: {e}\n")
            return None

    def create_or_update_configmap(self, name: str, namespace: str, data: dict):
        """Create a configmap if it doesn't exist, or update it if it does."""
        try:
            existing_configmap = self.core_v1_api.read_namespaced_config_map(
                name, namespace
            )
            # ConfigMap exists, update it
            existing_configmap.data = data
            self.core_v1_api.replace_namespaced_config_map(
                name, namespace, existing_configmap
            )
            print(f"ConfigMap '{name}' updated in namespace '{namespace}'")
        except ApiException as e:
            if e.status == 404:
                # ConfigMap doesn't exist, create it
                body = client.V1ConfigMap(
                    metadata=client.V1ObjectMeta(name=name), data=data
                )
                self.core_v1_api.create_namespaced_config_map(namespace, body)
                print(f"ConfigMap '{name}' created in namespace '{namespace}'")
            else:
                print(f"Error creating/updating ConfigMap '{name}': {e}")

    def update_configmap(self, name, namespace, data):
        """Update existing configmap with the provided data."""
        config_map = client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
            metadata=client.V1ObjectMeta(name=name),
            data=data,
        )
        try:
            return self.core_v1_api.replace_namespaced_config_map(
                name, namespace, config_map
            )
        except ApiException as e:
            print(f"Exception when updating configmap: {e}\n")
            return

    def apply_configs(self, namespace: str, config_path: str):
        """Apply Kubernetes configurations from a specified path to a namespace."""
        command = f"kubectl apply -Rf {config_path} -n {namespace}"
        self.exec_command(command)

    def delete_configs(self, namespace: str, config_path: str):
        """Delete Kubernetes configurations from a specified path in a namespace."""
        try:
            exists_resource = self.exec_command(
                f"kubectl get all -n {namespace} -o name"
            )
            if exists_resource:
                print(f"Deleting K8S configs in namespace: {namespace}")
                command = (
                    f"kubectl delete -Rf {config_path} -n {namespace} --timeout=10s"
                )
                self.exec_command(command)
            else:
                print(f"No resources found in: {namespace}. Skipping deletion.")
        except subprocess.CalledProcessError as e:
            print(f"Error deleting K8S configs: {e}")
            print(f"Command output: {e.output}")

    def delete_namespace(self, namespace: str):
        """Delete a specified namespace."""
        try:
            self.core_v1_api.delete_namespace(name=namespace)
            print(f"Namespace '{namespace}' deleted successfully.")
        except ApiException as e:
            if e.status == 404:
                print(f"Namespace '{namespace}' not found.")
            else:
                print(f"Error deleting namespace '{namespace}': {e}")

    def create_namespace_if_not_exist(self, namespace: str):
        """Create a namespace if it doesn't exist."""
        try:
            self.core_v1_api.read_namespace(name=namespace)
            print(f"Namespace '{namespace}' already exists.")
        except ApiException as e:
            if e.status == 404:
                print(f"Namespace '{namespace}' not found. Creating namespace.")
                body = client.V1Namespace(metadata=client.V1ObjectMeta(name=namespace))
                self.core_v1_api.create_namespace(body=body)
                print(f"Namespace '{namespace}' created successfully.")
            else:
                print(f"Error checking/creating namespace '{namespace}': {e}")

    def exec_command(self, command: str, input_data=None):
        """Execute an arbitrary kubectl command."""
        if input_data is not None:
            input_data = input_data.encode("utf-8")
        try:
            out = subprocess.run(
                command, shell=True, check=True, capture_output=True, input=input_data
            )
            return out.stdout.decode("utf-8")
        except subprocess.CalledProcessError as e:
            return e.stderr.decode("utf-8")

        # if out.stderr:
        #     return out.stderr.decode("utf-8")
        # else:
        #     return out.stdout.decode("utf-8")


# Example usage:
if __name__ == "__main__":
    kubectl = KubeCtl()
    namespace = "test-social-network"
    frontend_service = "nginx-thrift"
    user_service = "user-service"

    user_service_pod = kubectl.get_pod_name(namespace, f"app={user_service}")
    logs = kubectl.get_pod_logs(user_service_pod, namespace)
    print(logs)