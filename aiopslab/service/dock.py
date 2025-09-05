# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Interface to Docker controller service."""

import docker
import subprocess


class Docker:
    def __init__(self):
        self.client = docker.from_env()

    def list_containers(self):
        """Get all containers."""
        return self.client.containers.list()
    
    def get_container(self, container_id):
        """Get a container by ID."""
        return self.client.containers.get(container_id)
    
    def get_logs(self, container_id):
        """Get logs for a container."""
        return self.get_container(container_id).logs().decode("utf-8")
    
    def compose_up(self, cwd):
        """Run docker-compose up."""
        command = "docker compose up -d"
        return self.exec_command(command, cwd=cwd)
    
    def compose_down(self, cwd):
        """Run docker-compose down."""
        command = "docker compose down"
        return self.exec_command(command, cwd=cwd)
    
    def cleanup(self):
        """Remove the stopped docker containers."""
        command = "docker container prune -f"
        return self.exec_command(command)
        
    def exec_command(self, command: str, input_data=None, cwd=None):
        """Execute an arbitrary command."""
        if input_data is not None:
            input_data = input_data.encode("utf-8")
        try:
            out = subprocess.run(
                command,
                input=input_data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                cwd=cwd,
            )
            if out is not None:
                return out.stdout.decode("utf-8")
        except subprocess.CalledProcessError as e:
            return e.stderr.decode("utf-8")
