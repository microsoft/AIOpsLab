# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import yaml
import logging
from subprocess import CalledProcessError
from aiopslab.service.helm import Helm
from aiopslab.service.kubectl import KubeCtl
from aiopslab.paths import PROMETHEUS_METADATA, BASE_DIR


class Prometheus:
    def __init__(self):
        self.config_file = PROMETHEUS_METADATA
        self.name = None
        self.namespace = None
        self.helm_configs = {}
        self.pvc_config_file = None

        self.load_service_json()

    def load_service_json(self):
        """Load metric service metadata into attributes."""
        with open(self.config_file, "r") as file:
            metadata = json.load(file)

        self.name = metadata.get("Name")
        self.namespace = metadata.get("Namespace")

        self.helm_configs = metadata.get("Helm Config", {})

        self.name = metadata["Name"]
        self.namespace = metadata["Namespace"]
        if "Helm Config" in metadata:
            self.helm_configs = metadata["Helm Config"]
            if "chart_path" in self.helm_configs:
                chart_path = self.helm_configs["chart_path"]
                self.helm_configs["chart_path"] = str(BASE_DIR / chart_path)

        self.pvc_config_file = os.path.join(
            BASE_DIR, metadata.get("PersistentVolumeClaimConfig")
        )

    def get_service_json(self) -> dict:
        """Get metric service metadata in JSON format."""
        with open(self.config_file, "r") as file:
            return json.load(file)

    def get_service_summary(self) -> str:
        """Get a summary of the metric service metadata."""
        service_json = self.get_service_json()
        service_name = service_json.get("Name", "")
        namespace = service_json.get("Namespace", "")
        desc = service_json.get("Desc", "")
        supported_operations = service_json.get("Supported Operations", [])
        operations_str = "\n".join([f"  - {op}" for op in supported_operations])

        return (
            f"Telemetry Service Name: {service_name}\n"
            f"Namespace: {namespace}\n"
            f"Description: {desc}\n"
            f"Supported Operations:\n{operations_str}"
        )

    def deploy(self):
        """Deploy the metric collector using Helm."""
        if self._is_prometheus_running():
            print("Prometheus is already running. Skipping redeployment.")
            return

        self._delete_pvc()
        Helm.uninstall(**self.helm_configs)

        if self.pvc_config_file:
            pvc_name = self._get_pvc_name_from_file(self.pvc_config_file)
            if not self._pvc_exists(pvc_name):
                self._apply_pvc()

        Helm.install(**self.helm_configs)
        Helm.assert_if_deployed(self.namespace)

    def teardown(self):
        """Teardown the metric collector deployment."""
        Helm.uninstall(**self.helm_configs)

        if self.pvc_config_file:
            self._delete_pvc()

    def _apply_pvc(self):
        """Apply the PersistentVolumeClaim configuration."""
        print(f"Applying PersistentVolumeClaim from {self.pvc_config_file}")
        KubeCtl().exec_command(
            f"kubectl apply -f {self.pvc_config_file} -n {self.namespace}"
        )

    def _delete_pvc(self):
        """Delete the PersistentVolume and associated PersistentVolumeClaim."""
        pvc_name = self._get_pvc_name_from_file(self.pvc_config_file)
        result = KubeCtl().exec_command(f"kubectl get pvc {pvc_name} --ignore-not-found")

        if result:
            print(f"Deleting PersistentVolumeClaim {pvc_name}")
            KubeCtl().exec_command(f"kubectl delete pvc {pvc_name}")
            print(f"Successfully deleted PersistentVolumeClaim from {pvc_name}")
        else:
            print(f"PersistentVolumeClaim {pvc_name} not found. Skipping deletion.")

    def _get_pvc_name_from_file(self, pv_config_file):
        """Extract PVC name from the configuration file."""
        with open(pv_config_file, "r") as file:
            pv_config = yaml.safe_load(file)
            return pv_config["metadata"]["name"]

    def _pvc_exists(self, pvc_name: str) -> bool:
        """Check if the PersistentVolumeClaim exists."""
        command = f"kubectl get pvc {pvc_name}"
        try:
            result = KubeCtl().exec_command(command)
            if "No resources found" in result or "Error" in result:
                return False
        except CalledProcessError as e:
            logging.exception(f"Unexpected error while checking if the PersistentVolumeClaim exists: {e}")
            return False
        return True

    def _is_prometheus_running(self) -> bool:
        """Check if Prometheus Helm release is deployed."""
        try:
            release_name = self.helm_configs.get("release_name")
            namespace = self.helm_configs.get("namespace")
            if not release_name or not namespace:
                return False
            status_output = Helm.status(release_name=release_name, namespace=namespace)
            for line in status_output.splitlines():
                if line.strip().startswith("STATUS:"):
                    status_value = line.split(":", 1)[1].strip().lower()
                    return status_value == "deployed"
            return False
        except Exception as e:
            logging.exception(f"Unexpected error while checking Prometheus status: {e}")
            return False
