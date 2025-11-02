import json
import subprocess
from aiopslab.generators.fault.base import FaultInjector
from aiopslab.service.kubectl import KubeCtl


class OtelFaultInjector(FaultInjector):
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.kubectl = KubeCtl()
        self.configmap_name = "flagd-config"

    def inject_fault(self, feature_flag: str):
        command = (
            f"kubectl get configmap {self.configmap_name} -n {self.namespace} -o json"
        )
        try:
            output = self.kubectl.exec_command(command)
            configmap = json.loads(output)
        except subprocess.CalledProcessError:
            raise ValueError(
                f"ConfigMap '{self.configmap_name}' not found in namespace '{self.namespace}'."
            )
        except json.JSONDecodeError:
            raise ValueError(
                f"Error decoding JSON for ConfigMap '{self.configmap_name}'."
            )

        flagd_data = json.loads(configmap["data"]["demo.flagd.json"])

        if feature_flag in flagd_data["flags"]:
            if feature_flag == "paymentFailure":
                flagd_data["flags"][feature_flag]["defaultVariant"] = "100%"
            elif feature_flag == "imageSlowLoad":
                flagd_data["flags"][feature_flag]["defaultVariant"] = "10sec"
            else:
                flagd_data["flags"][feature_flag]["defaultVariant"] = "on"
        else:
            raise ValueError(
                f"Feature flag '{feature_flag}' not found in ConfigMap '{self.configmap_name}'."
            )

        updated_data = {"demo.flagd.json": json.dumps(flagd_data, indent=2)}
        self.kubectl.create_or_update_configmap(
            self.configmap_name, self.namespace, updated_data
        )

        self.kubectl.exec_command(
            f"kubectl rollout restart deployment flagd -n {self.namespace}"
        )
        
        print(f"Fault injected: Feature flag '{feature_flag}' set to 'on'.")

    def recover_fault(self, feature_flag: str):
        command = (
            f"kubectl get configmap {self.configmap_name} -n {self.namespace} -o json"
        )
        try:
            output = self.kubectl.exec_command(command)
            configmap = json.loads(output)
        except subprocess.CalledProcessError:
            raise ValueError(
                f"ConfigMap '{self.configmap_name}' not found in namespace '{self.namespace}'."
            )
        except json.JSONDecodeError:
            raise ValueError(
                f"Error decoding JSON for ConfigMap '{self.configmap_name}'."
            )

        flagd_data = json.loads(configmap["data"]["demo.flagd.json"])

        if feature_flag in flagd_data["flags"]:
            flagd_data["flags"][feature_flag]["defaultVariant"] = "off"
        else:
            raise ValueError(
                f"Feature flag '{feature_flag}' not found in ConfigMap '{self.configmap_name}'."
            )

        updated_data = {"demo.flagd.json": json.dumps(flagd_data, indent=2)}
        self.kubectl.create_or_update_configmap(
            self.configmap_name, self.namespace, updated_data
        )

        self.kubectl.exec_command(
            f"kubectl rollout restart deployment flagd -n {self.namespace}"
        )
        print(f"Fault recovered: Feature flag '{feature_flag}' set to 'off'.")


# Example usage:
# if __name__ == "__main__":
#     namespace = "astronomy-shop"
#     feature_flag = "adServiceFailure"

#     injector = OtelFaultInjector(namespace)

#     injector.inject_fault(feature_flag)
#     injector.recover_fault(feature_flag)
