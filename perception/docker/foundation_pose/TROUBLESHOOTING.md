# Docker GPU Access Troubleshooting

## Issue: "Failed to initialize NVML: Unknown Error" inside container

This means the Docker container doesn't have access to the NVIDIA GPU.

### Solution: Restart the container with GPU access

```bash
# 1. Stop and remove the existing container
sudo docker stop foundation_pose
sudo docker rm foundation_pose

# 2. Verify NVIDIA Docker runtime is installed on host
nvidia-smi  # Should work on host

# 3. Check Docker can see NVIDIA runtime
docker info | grep -i nvidia

# 4. Restart the container with GPU support
cd perception/docker/foundation_pose
sudo ./start.sh

# 5. Verify GPU is accessible inside container
sudo ./into.sh
nvidia-smi  # Should now work inside container
```

### If still not working, check NVIDIA Container Toolkit

```bash
# Install NVIDIA Container Toolkit (if not installed)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker daemon
sudo systemctl restart docker

# Now restart your container
cd perception/docker/foundation_pose
sudo docker stop foundation_pose
sudo docker rm foundation_pose
sudo ./start.sh
```

### Quick Fix Script

Run this to restart the container properly:

```bash
#!/bin/bash
cd perception/docker/foundation_pose

# Stop and remove old container
sudo docker stop foundation_pose 2>/dev/null || true
sudo docker rm foundation_pose 2>/dev/null || true

# Restart with GPU
sudo ./start.sh

# Test GPU access
echo "Testing GPU access in container..."
sudo docker exec foundation_pose nvidia-smi

if [ $? -eq 0 ]; then
    echo "✓ GPU is accessible in container!"
else
    echo "✗ GPU still not accessible. Check NVIDIA Docker runtime installation."
fi
```

### Verify GPU in Container

After restarting, run inside the container:

```bash
# Should show GPU info
nvidia-smi

# Should return True
python -c "import torch; print(torch.cuda.is_available())"
```
