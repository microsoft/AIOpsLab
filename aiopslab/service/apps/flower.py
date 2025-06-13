# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from aiopslab.service.dock import Docker
from aiopslab.service.apps.base import Application
from aiopslab.paths import FLOWER_METADATA


class Flower(Application):
    def __init__(self):
        super().__init__(FLOWER_METADATA)
        self.docker = Docker()
        
        self.load_app_json()
        
    def deploy(self):
        """Deploy the docker compose file."""
        print("Deploying docker compose files")
        self.docker.compose_up(self.docker_deploy_path)
        
    def delete(self):
        """Stop the docker containers."""
        print("Stopping the docker containers")
        self.docker.compose_down(self.docker_deploy_path)
    
    def cleanup(self):
        """Delete all stopped docker containers."""
        print("Deleting stopped containers")
        self.docker.cleanup()
