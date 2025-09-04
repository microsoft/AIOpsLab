"""Interface to the OpenTelemetry Astronomy Shop application"""

import time

from aiopslab.service.helm import Helm
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.base import Application
from aiopslab.paths import ASTRONOMY_SHOP_METADATA
from aiopslab.paths import config


class AstronomyShop(Application):
    def __init__(self):
        super().__init__(ASTRONOMY_SHOP_METADATA)
        self.load_app_json()
        self.kubectl = KubeCtl()
        self.create_namespace()

    def load_app_json(self):
        super().load_app_json()
        metadata = self.get_app_json()
        self.frontend_service = "frontend-proxy"
        self.frontend_port = 8080

    def deploy(self):
        """Deploy the Helm configurations."""
        self.kubectl.create_namespace_if_not_exist(self.namespace)
        Helm.add_repo(
            "open-telemetry",
            "https://open-telemetry.github.io/opentelemetry-helm-charts",
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
        Helm.uninstall(**self.helm_configs)
        self.kubectl.delete_namespace(self.helm_configs["namespace"])
        time.sleep(30)

    def cleanup(self):
        Helm.uninstall(**self.helm_configs)
        self.kubectl.delete_namespace(self.helm_configs["namespace"])


# Run this code to test installation/deletion
# if __name__ == "__main__":
#     shop = AstronomyShop()
#     shop.deploy()
#     shop.delete()
