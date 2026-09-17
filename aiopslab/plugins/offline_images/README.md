# Offline Images Plugin

This plugin enables AIOpsLab to work in environments with restricted or no internet access by loading pre-downloaded Docker images from local tar files into Kind clusters.

## Why Use This?

- **Network Restrictions**: Some environments don't have access to Docker Hub, GHCR, or other registries
- **Slow Networks**: Pulling large images repeatedly can be time-consuming
- **Reproducibility**: Pre-downloaded images ensure consistent versions across deployments

## Quick Start

### 1. Download Images (with internet access)

```bash
# Run this on a machine with internet access
./scripts/download_images.sh ./images
```

This will download all required images and save them as tar files.

### 2. Transfer Images (if needed)

Copy the `./images` directory to your target machine.

### 3. Configure AIOpsLab

Edit `aiopslab/config.yml`:

```yaml
# Enable offline mode
offline_mode: true
images_dir: ./images
```

### 4. Run as Usual

```bash
python cli.py
# or
python service.py
```

Images will be automatically loaded from local tars before deploying applications.

## Manual Usage

You can also use the ImageLoader programmatically:

```python
from aiopslab.plugins.offline_images import ImageLoader

# Initialize loader
loader = ImageLoader(images_dir="./images", cluster_name="kind")

# Load all images
count = loader.load_all_from_directory()
print(f"Loaded {count} images")

# Or load a specific image
loader.load_image_from_tar(Path("./images/nginx_latest.tar"))
```

## Tar File Naming Convention

Image tar files should be named in the format:
```
{registry}_{image_path}_{tag}.tar
```

Examples:
- `ghcr.io_open-telemetry_demo_1.11.1.tar`
- `docker.io_library_nginx_latest.tar`
- `quay.io_prometheus_prometheus_v2.47.2.tar`

The `download_images.sh` script handles this naming automatically.

## Supported Registries

- `ghcr.io` (GitHub Container Registry)
- `docker.io` (Docker Hub)
- `quay.io`
- `registry.k8s.io`
- `gcr.io` (Google Container Registry)

## Troubleshooting

### Images not loading

1. Check that the tar files exist in the images directory
2. Ensure Docker daemon is running
3. Verify Kind cluster is running: `kind get clusters`

### Image name mismatch

If an image fails to load, check that the tar filename follows the naming convention.
You can manually check the image name in a tar:

```bash
docker load -i image.tar
# Output: Loaded image: registry/image:tag
```
