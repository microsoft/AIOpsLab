# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Offline Images Plugin for AIOpsLab

This plugin enables offline/local image loading for Kind clusters,
which is useful in environments with restricted network access.

Usage:
    1. Download images: ./scripts/download_images.sh ./images
    2. Enable in config.yml:
       offline_mode: true
       images_dir: ./images
    3. Run AIOpsLab as usual - images will be loaded from local tars
"""

from .image_loader import ImageLoader, ensure_images_loaded

__all__ = ["ImageLoader", "ensure_images_loaded"]
