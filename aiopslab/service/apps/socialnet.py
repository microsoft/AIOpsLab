# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Interface to the social network application from DeathStarBench"""

import time

from aiopslab.service.helm import Helm
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.base import Application
from aiopslab.paths import TARGET_MICROSERVICES
from aiopslab.paths import SOCIAL_NETWORK_METADATA


class SocialNetwork(Application):
    def __init__(self):
        super().__init__(SOCIAL_NETWORK_METADATA)
        self.load_app_json()
        self.kubectl = KubeCtl()
        self.local_tls_path = (
            TARGET_MICROSERVICES / "socialNetwork/helm-chart/socialnetwork"
        )
        self.create_namespace()
        self.create_tls_secret()

    def load_app_json(self):
        super().load_app_json()
        metadata = self.get_app_json()
        self.frontend_service = metadata.get("frontend_service", "nginx-thrift")
        self.frontend_port = metadata.get("frontend_port", 8080)

    def create_tls_secret(self):
        check_sec = f"kubectl get secret mongodb-tls -n {self.namespace}"
        result = self.kubectl.exec_command(check_sec)
        if "notfound" in result.lower():
            create_sec_command = (
                f"kubectl create secret generic mongodb-tls "
                f"--from-file=tls.pem={self.local_tls_path}/tls.pem "
                f"--from-file=ca.crt={self.local_tls_path}/ca.crt "
                f"-n {self.namespace}"
            )
            create_result = self.kubectl.exec_command(create_sec_command)
            print(f"TLS secret created: {create_result.strip()}")
        else:
            print("TLS secret already exists. Skipping creation.")

    def deploy(self):
        """Deploy the Helm configurations."""
        Helm.install(**self.helm_configs)
        Helm.assert_if_deployed(self.helm_configs["namespace"])
        time.sleep(30)

    def delete(self):
        """Delete the Helm configurations."""
        Helm.uninstall(**self.helm_configs)

    def cleanup(self):
        """Delete the entire namespace for the social network application."""
        Helm.uninstall(**self.helm_configs)
        # self.kubectl.delete_namespace(self.namespace)
        # time.sleep(15)
