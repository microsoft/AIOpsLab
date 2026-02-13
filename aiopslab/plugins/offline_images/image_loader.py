# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Image Loader for Offline Deployment

Loads pre-downloaded Docker images from local tar files into Kind clusters.
This enables AIOpsLab to work in environments without internet access.
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class ImageLoader:
    """
    Loads Docker images from local tar files into Kind clusters.
    
    Tar files should be named in the format: registry_image_tag.tar
    Examples:
        - ghcr.io_open-telemetry_demo_1.0.0.tar
        - docker.io_library_nginx_latest.tar
    """
    
    def __init__(self, images_dir: str, cluster_name: str = "kind"):
        """
        Initialize ImageLoader.
        
        Args:
            images_dir: Directory containing pre-downloaded image tar files
            cluster_name: Name of the Kind cluster (default: "kind")
        """
        self.images_dir = Path(images_dir)
        self.cluster_name = cluster_name
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
    
    def load_image_from_tar(self, tar_path: Path) -> bool:
        """
        Load a single image from tar file into Kind cluster.
        
        Args:
            tar_path: Path to the tar file
            
        Returns:
            True if successful, False otherwise
        """
        if not tar_path.exists():
            logger.warning(f"Tar file not found: {tar_path}")
            return False
        
        # Step 1: Load into local Docker
        load_cmd = ["docker", "load", "-i", str(tar_path)]
        result = subprocess.run(load_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"Failed to load {tar_path.name}: {result.stderr[:100]}")
            return False
        
        # Extract image name from output or infer from filename
        image_name = None
        for line in result.stdout.split('\n'):
            if 'Loaded image:' in line:
                image_name = line.split('Loaded image:')[-1].strip()
                break
        
        if not image_name:
            # Infer from filename
            image_name = self._tar_name_to_image(tar_path)
        
        # Step 2: Load into Kind cluster
        kind_cmd = ["kind", "load", "docker-image", image_name, "--name", self.cluster_name]
        kind_result = subprocess.run(kind_cmd, capture_output=True, text=True)
        
        if kind_result.returncode != 0:
            if "already present" not in kind_result.stderr.lower():
                logger.warning(f"Failed to load {image_name} to Kind: {kind_result.stderr[:100]}")
                return False
        
        self.loaded_images.add(image_name)
        logger.info(f"Loaded image: {image_name}")
        return True
    
    def load_all_from_directory(self) -> int:
        """
        Load all tar files from the images directory into Kind cluster.
        
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
        
        logger.info(f"Loading {len(tar_files)} images from {self.images_dir}...")
        
        success_count = 0
        for i, tar_file in enumerate(tar_files, 1):
            logger.info(f"[{i}/{len(tar_files)}] Loading {tar_file.name}...")
            if self.load_image_from_tar(tar_file):
                success_count += 1
        
        logger.info(f"Loaded {success_count}/{len(tar_files)} images successfully")
        return success_count
    
    def is_image_loaded(self, image_name: str) -> bool:
        """
        Check if an image has been loaded.
        
        Args:
            image_name: Docker image name
            
        Returns:
            True if loaded, False otherwise
        """
        return image_name in self.loaded_images


# Global instance (lazily initialized)
_loader: Optional[ImageLoader] = None


def get_loader() -> Optional[ImageLoader]:
    """Get the global ImageLoader instance."""
    return _loader


def init_loader(images_dir: str, cluster_name: str = "kind") -> ImageLoader:
    """
    Initialize the global ImageLoader.
    
    Args:
        images_dir: Directory containing image tar files
        cluster_name: Name of the Kind cluster
        
    Returns:
        The initialized ImageLoader instance
    """
    global _loader
    _loader = ImageLoader(images_dir, cluster_name)
    return _loader


def ensure_images_loaded(images_dir: Optional[str] = None, cluster_name: str = "kind") -> bool:
    """
    Ensure all images from the directory are loaded into the Kind cluster.
    
    This is the main entry point for the offline images plugin.
    Call this before deploying applications.
    
    Args:
        images_dir: Directory containing image tar files (optional if already initialized)
        cluster_name: Name of the Kind cluster
        
    Returns:
        True if images were loaded successfully, False otherwise
    """
    global _loader
    
    if _loader is None:
        if images_dir is None:
            logger.warning("ImageLoader not initialized and no images_dir provided")
            return False
        _loader = ImageLoader(images_dir, cluster_name)
    
    count = _loader.load_all_from_directory()
    return count > 0
