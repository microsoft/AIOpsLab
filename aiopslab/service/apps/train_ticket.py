"""Interface to the Train Ticket application"""
import time

from aiopslab.service.helm import Helm
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.base import Application
from aiopslab.paths import TARGET_MICROSERVICES
from aiopslab.paths import TRAIN_TICKET_METADATA


class TrainTicket(Application):
    def __init__(self):
        super().__init__(TRAIN_TICKET_METADATA)
        self.load_app_json()
        self.kubectl = KubeCtl()
        self.create_namespace()

    def load_app_json(self):
        super().load_app_json()
        metadata = self.get_app_json()
        self.frontend_service = None
        self.frontend_port = None

    def deploy(self):
        """Deploy the Helm configurations."""
        self.kubectl.create_namespace_if_not_exist(self.namespace)
        Helm.install(**self.helm_configs)
        Helm.assert_if_deployed(self.helm_configs["namespace"])
        time.sleep(30)

    def delete(self):
        """Delete the Helm configurations."""
        # Helm.uninstall(**self.helm_configs) # Don't helm uninstall until cleanup job is fixed on train-ticket
        self.kubectl.delete_namespace(self.namespace)
        time.sleep(30)

    def cleanup(self):
        # Helm.uninstall(**self.helm_configs)
        self.kubectl.delete_namespace(self.namespace)

# if __name__ == "__main__":
#     app = TrainTicket()
#     app.deploy()
#     app.delete()
