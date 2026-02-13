# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Interface for helm operations"""

import subprocess
import os

from aiopslab.service.kubectl import KubeCtl


class Helm:
    # Offline mode flag (can be set via config)
    _offline_mode = False
    _images_dir = None
    _images_loaded = False
    
    @classmethod
    def configure_offline_mode(cls, enabled: bool, images_dir: str = None):
        """Configure offline mode for Helm deployments.
        
        Args:
            enabled: Whether to enable offline mode
            images_dir: Directory containing pre-downloaded image tar files
        """
        cls._offline_mode = enabled
        cls._images_dir = images_dir
        cls._images_loaded = False
        if enabled:
            print(f"== Offline Mode Enabled ==")
            print(f"   Images directory: {images_dir}")
    
    @classmethod
    def _ensure_images_loaded(cls):
        """Load images from local tar files if offline mode is enabled."""
        if not cls._offline_mode or cls._images_loaded:
            return
        
        if not cls._images_dir or not os.path.exists(cls._images_dir):
            print(f"Warning: Offline mode enabled but images_dir not found: {cls._images_dir}")
            return
        
        try:
            from aiopslab.plugins.offline_images import ensure_images_loaded
            print("== Loading Offline Images ==")
            ensure_images_loaded(cls._images_dir)
            cls._images_loaded = True
        except ImportError:
            print("Warning: offline_images plugin not found, skipping image loading")
        except Exception as e:
            print(f"Warning: Failed to load offline images: {e}")
    
    @staticmethod
    def install(**args):
        """Install a helm chart

        Args:
            release_name (str): Name of the release
            chart_path (str): Path to the helm chart
            namespace (str): Namespace to install the chart
            version (str): Version of the chart
            extra_args (List[str)]: Extra arguments for the helm install command
            remote_chart (bool): Whether the chart is remote (from a Helm repo)
        """
        # Load offline images if configured
        Helm._ensure_images_loaded()
        
        print("== Helm Install ==")
        release_name = args.get("release_name")
        chart_path = args.get("chart_path")
        namespace = args.get("namespace")
        version = args.get("version")
        extra_args = args.get("extra_args")
        remote_chart = args.get("remote_chart", False)

        if not remote_chart:
            # Install dependencies for chart before installation
            dependency_command = f"helm dependency update {chart_path}"
            dependency_process = subprocess.Popen(
                dependency_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            dependency_output, dependency_error = dependency_process.communicate()

        command = f"helm install {release_name} {chart_path} -n {namespace} --create-namespace"

        if version:
            command += f" --version {version}"

        if extra_args:
            command += " " + " ".join(extra_args)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        output, error = process.communicate()

        if error:
            print(error.decode("utf-8"))
        else:
            print(output.decode("utf-8"))

    @staticmethod
    def uninstall(**args):
        """Uninstall a helm chart

        Args:
            release_name (str): Name of the release
            namespace (str): Namespace to uninstall the chart
        """
        print("== Helm Uninstall ==")
        release_name = args.get("release_name")
        namespace = args.get("namespace")

        if not Helm.exists_release(release_name, namespace):
            print(f"Release {release_name} does not exist. Skipping uninstall.")
            return

        command = f"helm uninstall {release_name} -n {namespace}"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        output, error = process.communicate()

        if error:
            print(error.decode("utf-8"))
        else:
            print(output.decode("utf-8"))

    @staticmethod
    def exists_release(release_name: str, namespace: str) -> bool:
        """Check if a Helm release exists

        Args:
            release_name (str): Name of the release
            namespace (str): Namespace to check

        Returns:
            bool: True if release exists
        """
        command = f"helm list -n {namespace}"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        output, error = process.communicate()

        if error:
            print(error.decode("utf-8"))
            return False
        else:
            return release_name in output.decode("utf-8")

    @staticmethod
    def assert_if_deployed(namespace: str):
        """Assert if all services in the application are deployed

        Args:
            namespace (str): Namespace to check

        Returns:
            bool: True if deployed

        Raises:
            Exception: If not deployed
        """
        kubectl = KubeCtl()
        try:
            kubectl.wait_for_ready(namespace)
        except Exception as e:
            raise e

        return True

    @staticmethod
    def upgrade(**args):
        """Upgrade a helm chart

        Args:
            release_name (str): Name of the release
            chart_path (str): Path to the helm chart
            namespace (str): Namespace to upgrade the chart
            values_file (str): Path to the values.yaml file
            set_values (dict): Key-value pairs for --set options
        """
        print("== Helm Upgrade ==")
        release_name = args.get("release_name")
        chart_path = args.get("chart_path")
        namespace = args.get("namespace")
        values_file = args.get("values_file")
        set_values = args.get("set_values", {})

        command = [
            "helm",
            "upgrade",
            release_name,
            chart_path,
            "-n",
            namespace,
            "-f",
            values_file,
        ]

        # Add --set options if provided
        for key, value in set_values.items():
            command.append("--set")
            command.append(f"{key}={value}")

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, error = process.communicate()

        if error:
            print("Error during helm upgrade:")
            print(error.decode("utf-8"))
        else:
            print("Helm upgrade successful!")
            print(output.decode("utf-8"))

    @staticmethod
    def add_repo(name: str, url: str):
        """Add a Helm repository

        Args:
            name (str): Name of the repository
            url (str): URL of the repository
        """
        print(f"== Helm Repo Add: {name} ==")
        command = f"helm repo add {name} {url}"
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, error = process.communicate()

        if error:
            print(f"Error adding helm repo {name}: {error.decode('utf-8')}")
        else:
            print(f"Helm repo {name} added successfully: {output.decode('utf-8')}")

    @staticmethod
    def status(release_name: str, namespace: str) -> str:
        """Get the status of a Helm release.
        
        Args:
            release_name (str): Name of the release.
            namespace (str): Namespace of the release.
        
        Returns:
            str: Status output from Helm.
        
        Raises:
            ValueError: If either parameter is missing.
            Exception: If the helm status command fails.
        """

        if not release_name or not namespace:
            raise ValueError("Both release_name and namespace must be provided")

        command = f"helm status {release_name} -n {namespace}"
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, error = process.communicate()

        if process.returncode != 0:
            # helm status failed
            raise RuntimeError(f"Failed to get status for release {release_name}: {error.decode('utf-8')}")

        return output.decode("utf-8").strip()

# Example usage
if __name__ == "__main__":
    sn_configs = {
        "release_name": "test-social-network",
        "chart_path": "/home/oppertune/DeathStarBench/socialNetwork/helm-chart/socialnetwork",
        "namespace": "social-network",
    }
    Helm.install(**sn_configs)
    Helm.uninstall(**sn_configs)
