# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from aiopslab.paths import TARGET_MICROSERVICES


class Application:
    """Base class for all microservice applications."""

    def __init__(self, config_file: str):
        self.config_file = config_file
        self.name = None
        self.namespace = None
        self.helm_deploy = True
        self.helm_configs = {}
        self.k8s_deploy_path = None
        self.docker_deploy_path = None

    def load_app_json(self):
        """Load (basic) application metadata into attributes.

        # NOTE: override this method to load additional attributes!
        """
        with open(self.config_file, "r") as file:
            metadata = json.load(file)

        self.name = metadata["Name"]
        self.namespace = metadata["Namespace"]
        if "Helm Config" in metadata:
            self.helm_configs = metadata["Helm Config"]
            chart_path = self.helm_configs.get("chart_path")
            
            if chart_path and not self.helm_configs.get("remote_chart", False):
                self.helm_configs["chart_path"] = str(TARGET_MICROSERVICES / chart_path)

        if "K8S Deploy Path" in metadata:
            self.k8s_deploy_path = TARGET_MICROSERVICES / metadata["K8S Deploy Path"]
        
        if "Docker Deploy Path" in metadata:
            self.docker_deploy_path = TARGET_MICROSERVICES / metadata["Docker Deploy Path"]

    def get_app_json(self) -> dict:
        """Get application metadata in JSON format.

        Returns:
            dict: application metadata
        """
        with open(self.config_file, "r") as file:
            app_json = json.load(file)
        return app_json

    def get_app_summary(self) -> str:
        """Get a summary of the application metadata in string format.
        NOTE: for human and LLM-readable summaries!

        Returns:
            str: application metadata
        """
        app_json = self.get_app_json()
        app_name = app_json.get("Name", "")
        namespace = app_json.get("Namespace", "")
        desc = app_json.get("Desc", "")
        supported_operations = app_json.get("Supported Operations", [])
        operations_str = "\n".join([f"  - {op}" for op in supported_operations])

        description = f"Service Name: {app_name}\nNamespace: {namespace}\nDescription: {desc}\nSupported Operations:\n{operations_str}"

        return description

    def create_namespace(self):
        """Create the namespace for the application if it doesn't exist."""
        result = self.kubectl.exec_command(f"kubectl get namespace {self.namespace}")
        if "notfound" in result.lower():
            print(f"Namespace {self.namespace} not found. Creating namespace.")
            create_namespace_command = f"kubectl create namespace {self.namespace}"
            create_result = self.kubectl.exec_command(create_namespace_command)
            print(f"Namespace {self.namespace} created successfully: {create_result}")
        else:
            print(f"Namespace {self.namespace} already exists.")

    def cleanup(self):
        """Delete the entire namespace for the application."""
        self.kubectl.delete_namespace(self.namespace)
