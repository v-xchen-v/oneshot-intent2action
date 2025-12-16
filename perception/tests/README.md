# Perception Tests

Test scripts for validating perception modules.

## Available Tests

### `test_obj_mask.py`
Test object mask generation with text and bounding box inputs.

```bash
python perception/tests/test_obj_mask.py
```

### `test_webapi.sh` - WebAPI Tracking Test
Tests 6D pose tracking via WebAPI using test case data.

**Quick Start:**
```bash
# 1. Start API server (in one terminal)
python perception/webapi/pose_tracking_api.py --host 0.0.0.0 --port 5000

# 2. Run test (in another terminal)
./perception/tests/test_webapi.sh
```

**Custom Configuration:**
```bash
# Custom API URL
./perception/tests/test_webapi.sh http://192.168.1.100:5000

# Custom test case and output
./perception/tests/test_webapi.sh http://localhost:5000 /path/to/testcase /path/to/output
```

**Python Script (Advanced):**
```bash
python perception/tests/test_webapi_tracking.py \
    --api_url http://localhost:5000 \
    --test_case /path/to/test_case \
    --output /path/to/output \
    --no-visualize  # Faster without visualization
```

## Test Data Structure

```
test_case/<test_name>/
├── color/          # RGB images
├── depth/          # Depth images (16-bit PNG, mm)
├── mesh/           # 3D mesh (.stl, .ply, .obj)
└── 0_mask.png      # Initial mask
```

## Common Issues

**API server not responding:**
```bash
# Check if server is running
curl http://localhost:5000/api/health

# Check port usage
lsof -i :5000

# Restart server
python perception/webapi/pose_tracking_api.py --host 0.0.0.0 --port 5000
```

**Tracking fails:**
- Verify depth images are 16-bit PNG in millimeters
- Check initial mask covers the object
- Ensure mesh file is valid

**GPU memory errors:**
- Restart API server: kills old process and clears GPU memory