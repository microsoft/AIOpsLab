# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import time
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.base import Application
from aiopslab.paths import FAULT_SCRIPTS, FLOWER_METADATA


class Flower(Application):
    def __init__(self):
        super().__init__(FLOWER_METADATA)
        self.kubectl = KubeCtl()
        self.script_dir = FAULT_SCRIPTS
        self.helm_deploy = False

        self.load_app_json()
        self.create_namespace()

    def load_app_json(self):
        super().load_app_json()
        metadata = self.get_app_json()
        self.frontend_service = None
        self.frontend_port = None
        
    def deploy(self):
        """Deploy the Kubernetes configurations."""
        print(f"Deploying Kubernetes configurations in namespace: {self.namespace}")
        self.kubectl.apply_configs(self.namespace, self.k8s_deploy_path)
        self.kubectl.wait_for_ready(self.namespace)

    def delete(self):
        """Delete the configmap."""
        self.kubectl.delete_configs(self.namespace, self.k8s_deploy_path)

    def cleanup(self):
        """Delete the entire namespace for the flower application."""
        self.kubectl.delete_namespace(self.namespace)
        time.sleep(10)
        pvs = self.kubectl.exec_command(
            "kubectl get pv --no-headers | grep 'test-flower' | awk '{print $1}'"
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


if __name__ == "__main__":
    flower = Flower()
    flower.deploy()
    # flower.delete()
    # flower.cleanup()