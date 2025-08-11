#!/bin/bash
set -e

echo "Stopping old container if running..."
docker stop iris-api || true
docker rm iris-api || true

echo "Pulling latest image..."
docker pull $DOCKER_USERNAME/iris-api:latest

echo "Running new container..."
docker run -d --name iris-api -p 8050:5000 $DOCKER_USERNAME/iris-api:latest

