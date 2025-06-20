#!/usr/bin/env bash
# filepath: /home/ecs-user/projects/AIOpsLab/kind/pull_images.sh

IMAGE_LIST_FILE="kind/images.txt"
PULL_ERRORS=0
PULL_SUCCESS=0

# Check if Docker is installed
if ! command -v docker &>/dev/null; then
  echo "Error: 'docker' command not found. Please install Docker."
  exit 1
fi

# Check if the image list file exists
if [ ! -f "$IMAGE_LIST_FILE" ]; then
  echo "Error: Image list file '$IMAGE_LIST_FILE' not found."
  exit 1
fi

echo "Starting to pull Docker images from '$IMAGE_LIST_FILE'..."
echo "-------------------------------------"

# Read the file and pull each image
while IFS= read -r image_name || [ -n "$image_name" ]; do
  # Skip comments or empty lines
  [[ -z "$image_name" || "$image_name" == //* ]] && continue
  
  echo "Pulling image: $image_name"
  if docker pull "$image_name"; then
    echo "Successfully pulled $image_name"
    ((PULL_SUCCESS++))
  else
    echo "Failed to pull $image_name"
    ((PULL_ERRORS++))
  fi
  echo "-------------------------------------"
done < "$IMAGE_LIST_FILE"

# Print summary
echo "Pull complete!"
echo "Summary: $PULL_SUCCESS images pulled successfully, $PULL_ERRORS failures."

if [ $PULL_ERRORS -gt 0 ]; then
  echo "Some images failed to pull. Check the log above for details."
  exit 1
fi

exit 0