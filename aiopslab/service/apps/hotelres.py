# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.base import Application
from aiopslab.paths import FAULT_SCRIPTS, HOTEL_RES_METADATA


class HotelReservation(Application):
    def __init__(self):
        super().__init__(HOTEL_RES_METADATA)
        self.kubectl = KubeCtl()
        self.script_dir = FAULT_SCRIPTS
        self.helm_deploy = False

        self.load_app_json()
        self.create_namespace()
        self.create_configmaps()

    def load_app_json(self):
        super().load_app_json()
        metadata = self.get_app_json()
        self.frontend_service = metadata.get("frontend_service", "frontend")
        self.frontend_port = metadata.get("frontend_port", 5000)

    def create_configmaps(self):
        """Create configmaps for the hotel reservation application."""
        self.kubectl.create_or_update_configmap(
            name="mongo-rate-script",
            namespace=self.namespace,
            data=self._prepare_configmap_data(["k8s-rate-mongo.sh"]),
        )

        self.kubectl.create_or_update_configmap(
            name="mongo-geo-script",
            namespace=self.namespace,
            data=self._prepare_configmap_data(["k8s-geo-mongo.sh"]),
        )

        script_files = [
            "revoke-admin-rate-mongo.sh",
            "revoke-mitigate-admin-rate-mongo.sh",
            "remove-admin-mongo.sh",
            "remove-mitigate-admin-rate-mongo.sh",
        ]

        self.kubectl.create_or_update_configmap(
            name="failure-admin-rate",
            namespace=self.namespace,
            data=self._prepare_configmap_data(script_files),
        )

        script_files = [
            "revoke-admin-geo-mongo.sh",
            "revoke-mitigate-admin-geo-mongo.sh",
            "remove-admin-mongo.sh",
            "remove-mitigate-admin-geo-mongo.sh",
        ]

        self.kubectl.create_or_update_configmap(
            name="failure-admin-geo",
            namespace=self.namespace,
            data=self._prepare_configmap_data(script_files),
        )

    def deploy(self):
        """Deploy the Kubernetes configurations."""
        print(f"Deploying Kubernetes configurations in namespace: {self.namespace}")
        self.kubectl.apply_configs(self.namespace, self.k8s_deploy_path)
        self.kubectl.wait_for_ready(self.namespace)
    
    def deploy_without_wait(self):
        """Deploy the Kubernetes configurations without waiting for ready."""
        
        print(f"Deploying Kubernetes configurations in namespace: {self.namespace}")
        self.kubectl.apply_configs(self.namespace, self.k8s_deploy_path)
        print(f"Waiting for stability...")
        time.sleep(30)

    def delete(self):
        """Delete the configmap."""
        self.kubectl.delete_configs(self.namespace, self.k8s_deploy_path)

    def cleanup(self):
        """Delete the entire namespace for the hotel reservation application."""
        self.kubectl.delete_namespace(self.namespace)
        time.sleep(10)
        pvs = self.kubectl.exec_command(
            "kubectl get pv --no-headers | grep 'test-hotel-reservation' | awk '{print $1}'"
        ).splitlines()

        for pv in pvs:
            # Check if the PV is in a 'Terminating' state and remove the finalizers if necessary
            self._remove_pv_finalizers(pv)
            delete_command = f"kubectl delete pv {pv}"
            delete_result = self.kubectl.exec_command(delete_command)
            print(f"Deleted PersistentVolume {pv}: {delete_result.strip()}")
        time.sleep(5)

    def _remove_pv_finalizers(self, pv_name: str):
        """Remove finalizers from the PersistentVolume to prevent it from being stuck in a 'Terminating' state."""
        # Patch the PersistentVolume to remove finalizers if it is stuck
        patch_command = (
            f'kubectl patch pv {pv_name} -p \'{{"metadata":{{"finalizers":null}}}}\''
        )
        _ = self.kubectl.exec_command(patch_command)

    # helper methods
    def _prepare_configmap_data(self, script_files: list) -> dict:
        data = {}
        for file in script_files:
            data[file] = self._read_script(f"{self.script_dir}/{file}")
        return data

    def _read_script(self, file_path: str) -> str:
        with open(file_path, "r") as file:
            return file.read()
