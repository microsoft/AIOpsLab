#!/usr/bin/env bash

IMAGE_LIST_FILE="kind/images.txt"
CLUSTER_NAME="$1"

if ! command -v kind &>/dev/null; then
  echo "Error: 'kind' command not found. Please install kind."
  exit 1
fi

# Check if the image list file exists
if [ ! -f "$IMAGE_LIST_FILE" ]; then
  echo "Error: Image list file '$IMAGE_LIST_FILE' not found."
  exit 1
fi

if [ -n "$CLUSTER_NAME" ]; then
  echo "Loading images into cluster '$CLUSTER_NAME'..."
else
  echo "Loading images into default kind cluster..."
fi

while IFS= read -r image_name || [ -n "$image_name" ]; do
  [[ -z "$image_name" || "$image_name" == //* ]] && continue
  echo "Loading $image_name to kind..."
  if [ -n "$CLUSTER_NAME" ]; then
    kind load docker-image "$image_name" --name "$CLUSTER_NAME" || echo "Failed to load $image_name."
  else
    kind load docker-image "$image_name" || echo "Failed to load $image_name."
  fi
  echo "-------------------------------------"
done < "$IMAGE_LIST_FILE"

echo "Done loading images."