"""Interface to the Flight Ticket application"""
import time

from aiopslab.service.helm import Helm
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.base import Application
from aiopslab.paths import TARGET_MICROSERVICES
from aiopslab.paths import FLIGHT_TICKET_METADATA
from aiopslab.paths import config


class FlightTicket(Application):
    def __init__(self):
        super().__init__(FLIGHT_TICKET_METADATA)
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
        Helm.add_repo(
            "flight-ticket",
            "https://xlab-uiuc.github.io/flight-ticket",
        )
        
        # Apply image pull policy from config if specified
        helm_config = config.get("helm", {})
        if helm_config and "image_pull_policy" in helm_config:
            if "extra_args" not in self.helm_configs:
                self.helm_configs["extra_args"] = []
            self.helm_configs["extra_args"].append(
                f"--set global.imagePullPolicy={helm_config['image_pull_policy']}"
            )
        
        Helm.install(**self.helm_configs)
        Helm.assert_if_deployed(self.helm_configs["namespace"])

    def delete(self):
        """Delete the Helm configurations."""
        # NOTE: We should probably clear redis?
        Helm.uninstall(**self.helm_configs)
        time.sleep(30)

    def cleanup(self):
        Helm.uninstall(**self.helm_configs)


# if __name__ == "__main__":
#     app = FlightTicket()
#     app.deploy()
    # app.delete()
