#!/bin/bash
# Restart FoundationPose container with GPU access

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

echo "==================================================================="
echo "Restarting FoundationPose container with GPU access"
echo "==================================================================="

# Stop and remove existing container
echo "Stopping existing container..."
sudo docker stop foundation_pose 2>/dev/null || true
sudo docker rm foundation_pose 2>/dev/null || true

# Start new container with GPU
echo "Starting new container with GPU support..."
cd "$REPO_ROOT"
sudo ./perception/docker/foundation_pose/start.sh

# Wait a moment for container to start
sleep 2

# Test GPU access
echo ""
echo "==================================================================="
echo "Testing GPU access in container..."
echo "==================================================================="

if sudo docker exec foundation_pose nvidia-smi > /dev/null 2>&1; then
    echo "✓ GPU is accessible in container!"
    echo ""
    echo "GPU Information:"
    sudo docker exec foundation_pose nvidia-smi
    echo ""
    echo "==================================================================="
    echo "Container ready! Enter with: sudo ./into.sh"
    echo "==================================================================="
else
    echo "✗ GPU not accessible in container"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check GPU on host: nvidia-smi"
    echo "2. Check NVIDIA Docker runtime: docker info | grep -i nvidia"
    echo "3. Install nvidia-container-toolkit if missing"
    echo "4. See TROUBLESHOOTING.md for detailed steps"
    exit 1
fi
