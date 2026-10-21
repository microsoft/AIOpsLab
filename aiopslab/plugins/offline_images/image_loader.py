# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Image Loader for Offline Deployment

Loads pre-downloaded Docker images from local tar files into Kubernetes clusters.
Supports all three AIOpsLab deployment modes:
  - Kind (k8s_host: kind) — loads via `kind load docker-image`
  - Localhost (k8s_host: localhost) — loads via `docker load` (or ctr/crictl for containerd)
  - Remote (k8s_host: <hostname>) — loads via SSH + `docker load`

This enables AIOpsLab to work in environments without internet access.
"""

import subprocess
import os
from pathlib import Path
from typing import Optional, Set
import logging

logger = logging.getLogger(__name__)


class ImageLoader:
    """
    Loads Docker images from local tar files into Kubernetes clusters.
    
    Supports all three AIOpsLab deployment modes based on k8s_host config:
      - "kind": uses `kind load docker-image` to load images into Kind nodes
      - "localhost": uses `docker load` locally (cluster runs on the same machine)
      - "<hostname>": uses SSH + `docker load` to load images on remote nodes
    
    Tar files should be named in the format: registry_image_tag.tar
    Examples:
        - ghcr.io_open-telemetry_demo_1.0.0.tar
        - docker.io_library_nginx_latest.tar
    """
    
    def __init__(self, images_dir: str, k8s_host: str = "kind",
                 cluster_name: str = "kind", k8s_user: str = None,
                 ssh_key_path: str = None):
        """
        Initialize ImageLoader.
        
        Args:
            images_dir: Directory containing pre-downloaded image tar files
            k8s_host: Cluster host type from config.yml — "kind", "localhost", or a remote hostname
            cluster_name: Name of the Kind cluster (only used when k8s_host is "kind")
            k8s_user: SSH username for remote clusters (only used when k8s_host is a hostname)
            ssh_key_path: SSH key path for remote clusters (only used when k8s_host is a hostname)
        """
        self.images_dir = Path(images_dir)
        self.k8s_host = k8s_host
        self.cluster_name = cluster_name
        self.k8s_user = k8s_user
        self.ssh_key_path = ssh_key_path or "~/.ssh/id_rsa"
        self.loaded_images: Set[str] = set()
        
    def _tar_name_to_image(self, tar_path: Path) -> str:
        """
        Convert tar filename back to Docker image name.
        
        Args:
            tar_path: Path to the tar file
            
        Returns:
            Docker image name (e.g., "ghcr.io/open-telemetry/demo:1.0.0")
        """
        name = tar_path.stem  # Remove .tar extension
        
        # Known registry prefixes
        registry_prefixes = [
            'ghcr.io_',
            'quay.io_',
            'registry.k8s.io_',
            'docker.io_',
            'gcr.io_',
        ]
        
        registry = ''
        for prefix in registry_prefixes:
            if name.startswith(prefix):
                # Convert prefix back to registry URL
                registry = prefix.replace('_', '/', 1).rstrip('_') + '/'
                name = name[len(prefix):]
                break
        
        # Find the last underscore as tag separator
        parts = name.rsplit('_', 1)
        if len(parts) == 2:
            image_path = parts[0].replace('_', '/')
            tag = parts[1]
            return f"{registry}{image_path}:{tag}"
        else:
            return name.replace('_', '/')
    
    def _load_to_kind(self, image_name: str) -> bool:
        """Load image into Kind cluster via `kind load docker-image`."""
        cmd = ["kind", "load", "docker-image", image_name, "--name", self.cluster_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            if "already present" not in result.stderr.lower():
                logger.warning(f"Failed to load {image_name} to Kind: {result.stderr[:200]}")
                return False
        return True
    
    def _load_to_localhost(self, tar_path: Path) -> bool:
        """Load image on localhost via `docker load` (cluster runs locally)."""
        # docker load is already done in load_image_from_tar(), nothing extra needed
        return True
    
    def _load_to_remote(self, tar_path: Path, image_name: str) -> bool:
        """Load image on remote cluster node via SSH + `docker load`."""
        ssh_key = os.path.expanduser(self.ssh_key_path)
        
        # Use SSH to pipe the tar file to docker load on the remote host
        cmd = (
            f"ssh -i {ssh_key} -o StrictHostKeyChecking=no "
            f"{self.k8s_user}@{self.k8s_host} 'docker load'"
        )
        
        try:
            with open(tar_path, 'rb') as f:
                result = subprocess.run(
                    cmd, shell=True, stdin=f,
                    capture_output=True, text=True, timeout=300
                )
            
            if result.returncode != 0:
                logger.warning(f"Failed to load {image_name} to remote: {result.stderr[:200]}")
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout loading {image_name} to remote host")
            return False
        except Exception as e:
            logger.warning(f"Error loading {image_name} to remote: {e}")
            return False
    
    def load_image_from_tar(self, tar_path: Path) -> bool:
        """
        Load a single image from tar file into the cluster.
        
        The loading method depends on k8s_host:
          - "kind": docker load + kind load docker-image
          - "localhost": docker load only
          - "<hostname>": SSH + docker load on remote node
        
        Args:
            tar_path: Path to the tar file
            
        Returns:
            True if successful, False otherwise
        """
        if not tar_path.exists():
            logger.warning(f"Tar file not found: {tar_path}")
            return False
        
        # Step 1: Load into local Docker (needed for Kind and Localhost modes)
        image_name = None
        
        if self.k8s_host in ("kind", "localhost"):
            load_cmd = ["docker", "load", "-i", str(tar_path)]
            result = subprocess.run(load_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"Failed to docker load {tar_path.name}: {result.stderr[:200]}")
                return False
            
            # Extract image name from output
            for line in result.stdout.split('\n'):
                if 'Loaded image:' in line:
                    image_name = line.split('Loaded image:')[-1].strip()
                    break
        
        if not image_name:
            image_name = self._tar_name_to_image(tar_path)
        
        # Step 2: Load into cluster based on k8s_host mode
        if self.k8s_host == "kind":
            if not self._load_to_kind(image_name):
                return False
        elif self.k8s_host == "localhost":
            if not self._load_to_localhost(tar_path):
                return False
        else:
            # Remote host
            if not self._load_to_remote(tar_path, image_name):
                return False
        
        self.loaded_images.add(image_name)
        logger.info(f"Loaded image: {image_name} (mode: {self.k8s_host})")
        return True
    
    def load_all_from_directory(self) -> int:
        """
        Load all tar files from the images directory into the cluster.
        
        Returns:
            Number of successfully loaded images
        """
        if not self.images_dir.exists():
            logger.warning(f"Images directory not found: {self.images_dir}")
            return 0
        
        tar_files = list(self.images_dir.glob("*.tar"))
        if not tar_files:
            logger.warning(f"No tar files found in {self.images_dir}")
            return 0
        
        mode_desc = {
            "kind": "Kind cluster (docker load + kind load)",
            "localhost": "Localhost (docker load)",
        }.get(self.k8s_host, f"Remote host '{self.k8s_host}' (SSH + docker load)")
        
        logger.info(f"Loading {len(tar_files)} images from {self.images_dir}")
        logger.info(f"  Mode: {mode_desc}")
        
        success_count = 0
        for i, tar_file in enumerate(tar_files, 1):
            logger.info(f"[{i}/{len(tar_files)}] Loading {tar_file.name}...")
            if self.load_image_from_tar(tar_file):
                success_count += 1
        
        logger.info(f"Loaded {success_count}/{len(tar_files)} images successfully")
        return success_count
    
    def is_image_loaded(self, image_name: str) -> bool:
        """Check if an image has been loaded."""
        return image_name in self.loaded_images


# Global instance (lazily initialized)
_loader: Optional[ImageLoader] = None


def get_loader() -> Optional[ImageLoader]:
    """Get the global ImageLoader instance."""
    return _loader


def init_loader(images_dir: str, **kwargs) -> ImageLoader:
    """
    Initialize the global ImageLoader.
    
    Args:
        images_dir: Directory containing image tar files
        **kwargs: Additional arguments passed to ImageLoader
        
    Returns:
        The initialized ImageLoader instance
    """
    global _loader
    _loader = ImageLoader(images_dir, **kwargs)
    return _loader


def ensure_images_loaded(images_dir: Optional[str] = None, **kwargs) -> bool:
    """
    Ensure all images from the directory are loaded into the cluster.
    
    This is the main entry point for the offline images plugin.
    Call this before deploying applications.
    
    Args:
        images_dir: Directory containing image tar files (optional if already initialized)
        **kwargs: Additional arguments (k8s_host, cluster_name, k8s_user, ssh_key_path)
        
    Returns:
        True if images were loaded successfully, False otherwise
    """
    global _loader
    
    if _loader is None:
        if images_dir is None:
            logger.warning("ImageLoader not initialized and no images_dir provided")
            return False
        _loader = ImageLoader(images_dir, **kwargs)
    
    count = _loader.load_all_from_directory()
    return count > 0
