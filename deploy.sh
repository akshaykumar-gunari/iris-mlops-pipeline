#!/bin/bash
set -e

if [[ -z "$DOCKERHUB_USERNAME" ]]; then
    echo "❌ ERROR: DOCKERHUB_USERNAME is not set."
    exit 1
fi

echo "Stopping old container if running..."
docker stop iris-api || true
docker rm iris-api || true

IMAGE_NAME="${DOCKERHUB_USERNAME}/iris-api:latest"
echo "Pulling latest image: $IMAGE_NAME"
docker pull "$IMAGE_NAME"

echo "Running new container..."
docker run -d --name iris-api -p 8050:5000 "$IMAGE_NAME"
