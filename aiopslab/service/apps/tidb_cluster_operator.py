import time
import json

from aiopslab.service.helm import Helm
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.base import Application
from aiopslab.paths import TIDB_METADATA


class TiDBCluster(Application):
    def __init__(self):
        super().__init__(TIDB_METADATA)
        self.load_app_json()
        self.kubectl = KubeCtl()
        self.helm_operator_config = self.metadata.get("Helm Operator Config", {})
        self.k8s_config = self.metadata.get("K8S Config", {})

        # self.create_namespace()
        # self.deploy_tidb_operator()

    def load_app_json(self):
        with open(self.config_file, "r") as file:
            self.metadata = json.load(file)

        self.name = self.metadata["Name"]
        self.namespace = self.metadata["Namespace"]  # tidb cluster namespace

    def deploy(self):
        """Deploy the TiDB operator and cluster."""
        # self.install_crd()
        # self.install_tidb_operator()
        # self.deploy_tidb_cluster()
        # NOTE: use Ansible to deploy
        pass

    def install_crd(self):
        """Install the Custom Resource Definitions (CRDs) for TiDB Operator."""
        crd_url = self.helm_operator_config.get("CRD")
        if not crd_url:
            raise ValueError("CRD URL is not specified in the configuration.")

        print("Installing CRDs...")
        command = f"kubectl create -f {crd_url}"
        self.kubectl.exec_command(command)

    def install_tidb_operator(self):
        """Install TiDB Operator using Helm."""
        repo_name = "pingcap"
        repo_url = "https://charts.pingcap.org/"
        Helm.add_repo(repo_name, repo_url)

        operator_namespace = self.helm_operator_config.get("namespace", "tidb-admin")
        self.kubectl.create_namespace_if_not_exist(operator_namespace)

        print("Installing TiDB Operator...")
        Helm.install(**self.helm_operator_config)
        Helm.assert_if_deployed(operator_namespace)
        time.sleep(5)

    def deploy_tidb_cluster(self):
        """Deploy the TiDB Cluster using Helm."""
        # cluster_namespace = self.k8s_config.get("namespace", "tidb-cluster")
        # self.kubectl.create_namespace_if_not_exist(cluster_namespace)
        # cluster_config_url = self.k8s_config.get("config_url")

        # print("Deploying TiDB cluster...")
        # command = f"kubectl -n {cluster_namespace} apply -f {cluster_config_url}"
        # self.kubectl.exec_command(command)
        # NOTE: use Ansible to deploy
        pass

    def delete_tidb_cluster(self):
        """Delete the TiDB Cluster."""
        self.kubectl.exec_command(f"kubectl delete tc basic -n {self.namespace}")

    def delete_tidb_operator(self):
        """Delete the TiDB Operator."""
        Helm.uninstall(**self.helm_operator_config)

    def delete(self):
        """Delete the TiDB Operator and Cluster."""
        # self.kubectl.delete_namespace(self.namespace)
        # self.delete_tidb_operator()
        # time.sleep(5)
        pass

    def cleanup(self):
        """Clean up the namespace and all resources."""
        # self.delete_tidb_cluster()
        # self.delete_tidb_operator()
        # self.kubectl.delete_namespace(self.namespace)
        # time.sleep(15)
        pass


if __name__ == "__main__":
    tidb_app = TiDBCluster()
    tidb_app.deploy_tidb_cluster()
