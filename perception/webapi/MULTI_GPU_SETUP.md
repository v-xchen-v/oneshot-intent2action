# Running APIs on Different GPUs

## Quick Start

Run each API on a different GPU using `CUDA_VISIBLE_DEVICES`:

```bash
# Terminal 1: Mesh Generation API on GPU 0
CUDA_VISIBLE_DEVICES=0 ./perception/webapi/start_mesh_api.sh --port 5001

# Terminal 2: Pose Tracking API on GPU 1
CUDA_VISIBLE_DEVICES=1 ./perception/webapi/start_api.sh --port 5000
```

That's it! Each API will now use its assigned GPU.

## Check Available GPUs

```bash
nvidia-smi
```

## Common Scenarios

### Single GPU System
```bash
# Both APIs share GPU 0 (may cause memory issues)
CUDA_VISIBLE_DEVICES=0 ./perception/webapi/start_mesh_api.sh --port 5001
CUDA_VISIBLE_DEVICES=3 ./perception/webapi/start_api.sh --port 5000
```

### Different GPU Numbers
```bash
# Use any GPU ID (0, 1, 2, 3, etc.)
CUDA_VISIBLE_DEVICES=2 ./perception/webapi/start_mesh_api.sh --port 5001
CUDA_VISIBLE_DEVICES=3 ./perception/webapi/start_api.sh --port 5000
```

### Multiple GPUs for One API
```bash
# Mesh API uses GPU 0 and 1
CUDA_VISIBLE_DEVICES=0,1 ./perception/webapi/start_mesh_api.sh --port 5001

# Pose API uses GPU 2
CUDA_VISIBLE_DEVICES=2 ./perception/webapi/start_api.sh --port 5000
```

## Monitor GPU Usage

```bash
# Watch in real-time
watch -n 1 nvidia-smi

# Check which process uses which GPU
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

## Troubleshooting

### Out of Memory Error
→ Use different GPUs for each API

### Wrong GPU Being Used
```bash
# Make sure to set the variable before running
export CUDA_VISIBLE_DEVICES=0
./perception/webapi/start_mesh_api.sh
```

### Check CUDA Availability
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```
