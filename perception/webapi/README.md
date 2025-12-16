# Web APIs for 3D Perception

This directory contains REST API services for 3D perception tasks.

## Available APIs

1. **[6D Object Pose Tracking API](#pose-tracking-api)** - Real-time 6D object pose tracking using FoundationPose++
2. **[3D Mesh Generation API](#mesh-generation-api)** - Generate 3D meshes from RGB images using SAM3 + SAM3D

---

# Pose Tracking API

A REST API service for real-time 6D object pose tracking using FoundationPose++ with camera streaming support.

## Features

- **Session-based tracking**: Create multiple independent tracking sessions
- **Real-time streaming**: Process RGB-D frames from live camera streams
- **Multiple tracking modes**:
  - Pure 6D pose estimation (FoundationPose)
  - With 2D object tracker (Cutie) for improved robustness
  - With Kalman filtering for smooth pose estimates
- **Visualization**: Optional pose visualization with 3D bounding boxes and axes
- **Remote deployment**: Designed for deployment on remote GPU servers

## Architecture

```
Client (Camera) → REST API → FoundationPose++ → Pose Estimates
                              ├─ Cutie (2D Tracker)
                              └─ Kalman Filter
```

## API Endpoints

### Health Check
```http
GET /api/health
```

### Create Session
```http
POST /api/session/create
Content-Type: application/json

{
  "mesh_file": "base64_encoded_mesh",
  "mesh_scale": 0.01,
  "cam_K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "activate_2d_tracker": true,
  "activate_kalman_filter": true
}
```

### Initialize Tracking
```http
POST /api/session/{session_id}/initialize
Content-Type: application/json

{
  "rgb": "base64_encoded_rgb",
  "depth": "base64_encoded_depth",
  "mask": "base64_encoded_mask",
  "depth_scale": 1000
}
```

### Track Frame
```http
POST /api/session/{session_id}/track
Content-Type: application/json

{
  "rgb": "base64_encoded_rgb",
  "depth": "base64_encoded_depth",
  "depth_scale": 1000,
  "visualize": true
}
```

### Get Pose History
```http
GET /api/session/{session_id}/history
```

### Close Session
```http
DELETE /api/session/{session_id}/close
```

### List Sessions
```http
GET /api/sessions
```

## Installation

### On Remote Server

1. **Install dependencies**:
```bash
cd perception/webapi
pip install -r requirements.txt
```

2. **Ensure FoundationPose++ is set up**:
```bash
# The API automatically adds these to Python path:
# - external/FoundationPose-plus-plus/src
# - external/FoundationPose-plus-plus/FoundationPose
```

3. **Start the API server**:
```bash
python pose_tracking_api.py --host 0.0.0.0 --port 5000
```

### Docker Deployment (Recommended)

```bash
# Build Docker image
docker build -t pose-tracking-api -f perception/docker/Dockerfile.api .

# Run container
docker run --gpus all -p 5000:5000 pose-tracking-api
```

## Usage Examples

### Python Client

```python
from client_example import PoseTrackingClient
import numpy as np
import cv2

# Initialize client
client = PoseTrackingClient(api_url="http://your-server:5000")

# Camera intrinsics
cam_K = np.array([
    [912.7, 0.0, 667.6],
    [0.0, 911.0, 360.5],
    [0.0, 0.0, 1.0]
])

# Create session
client.create_session(
    mesh_path="toy_bear.stl",
    cam_K=cam_K,
    mesh_scale=1.0,
    activate_2d_tracker=True,
    activate_kalman_filter=True
)

# Initialize with first frame
rgb = cv2.imread("frame_0000.png")
depth = cv2.imread("depth_0000.png", -1)
mask = cv2.imread("mask_0000.png", cv2.IMREAD_GRAYSCALE)

pose = client.initialize(rgb, depth, mask)

# Track subsequent frames
for i in range(1, num_frames):
    rgb = cv2.imread(f"frame_{i:04d}.png")
    depth = cv2.imread(f"depth_{i:04d}.png", -1)
    
    result = client.track(rgb, depth, visualize=True)
    pose = result['pose']
    print(f"Translation: {pose['translation']}")
    print(f"Quaternion: {pose['quaternion']}")

# Cleanup
client.close_session()
```

### JavaScript Client

```javascript
// Create session
const response = await fetch('http://server:5000/api/session/create', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    mesh_file: meshBase64,
    cam_K: [[912.7, 0, 667.6], [0, 911.0, 360.5], [0, 0, 1]],
    mesh_scale: 0.01,
    activate_2d_tracker: true
  })
});
const { session_id } = await response.json();

// Initialize
const initResponse = await fetch(`http://server:5000/api/session/${session_id}/initialize`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    rgb: rgbBase64,
    depth: depthBase64,
    mask: maskBase64
  })
});

// Track frames
const trackResponse = await fetch(`http://server:5000/api/session/${session_id}/track`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    rgb: rgbBase64,
    depth: depthBase64,
    visualize: true
  })
});
const { pose, visualization } = await trackResponse.json();
```

### cURL Examples

```bash
# Health check
curl http://localhost:5000/api/health

# Create session
curl -X POST http://localhost:5000/api/session/create \
  -H "Content-Type: application/json" \
  -d @session_config.json

# Initialize tracking
curl -X POST http://localhost:5000/api/session/{SESSION_ID}/initialize \
  -H "Content-Type: application/json" \
  -d @init_frame.json

# Track frame
curl -X POST http://localhost:5000/api/session/{SESSION_ID}/track \
  -H "Content-Type: application/json" \
  -d @frame.json
```

## Camera Streaming Integration

### RealSense Camera

```python
import pyrealsense2 as rs
import numpy as np

# Configure RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Get camera intrinsics
profile = pipeline.get_active_profile()
color_profile = profile.get_stream(rs.stream.color)
intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
cam_K = np.array([
    [intrinsics.fx, 0, intrinsics.ppx],
    [0, intrinsics.fy, intrinsics.ppy],
    [0, 0, 1]
])

# Stream frames
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue
        
        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        
        # Send to API
        result = client.track(rgb, depth, depth_scale=1000.0)
        
finally:
    pipeline.stop()
```

### WebRTC Streaming

For browser-based streaming, use WebRTC to capture and send frames:

```javascript
// Get camera stream
const stream = await navigator.mediaDevices.getUserMedia({ video: true });
const video = document.createElement('video');
video.srcObject = stream;

// Capture frames
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

setInterval(() => {
  ctx.drawImage(video, 0, 0);
  const imageData = canvas.toDataURL('image/png');
  const base64 = imageData.split(',')[1];
  
  // Send to API
  fetch(`http://server:5000/api/session/${sessionId}/track`, {
    method: 'POST',
    body: JSON.stringify({ rgb: base64, depth: depthBase64 })
  });
}, 33); // ~30 FPS
```

## Response Format

### Pose Format
```json
{
  "pose": {
    "matrix": [[4x4 transformation matrix]],
    "translation": [x, y, z],
    "rotation_matrix": [[3x3 rotation matrix]],
    "quaternion": [x, y, z, w]
  },
  "frame_count": 42,
  "bbox_2d": [x, y, width, height]
}
```

## Performance Considerations

- **GPU Memory**: Each session uses ~2-4GB GPU memory
- **Throughput**: ~10-30 FPS depending on refinement iterations
- **Latency**: ~30-100ms per frame (network + processing)
- **Optimization**:
  - Reduce `track_refine_iter` (default: 5) for faster tracking
  - Disable visualization when not needed
  - Use 2D tracker to reduce refinement iterations needed

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh_scale` | float | 0.01 | Mesh scale factor (e.g., 0.01 for mm→m) |
| `est_refine_iter` | int | 10 | Initial pose refinement iterations |
| `track_refine_iter` | int | 5 | Tracking refinement iterations |
| `activate_2d_tracker` | bool | false | Enable Cutie 2D tracker |
| `activate_kalman_filter` | bool | false | Enable Kalman filtering |
| `kf_measurement_noise_scale` | float | 0.05 | Kalman filter noise scale |
| `depth_scale` | float | 1000.0 | Depth scale (mm to meters) |

## Troubleshooting

### Common Issues

1. **"Session not found"**: Session expired or invalid ID
2. **"Session not initialized"**: Call `/initialize` before `/track`
3. **GPU OOM**: Reduce number of concurrent sessions
4. **Slow tracking**: Reduce `track_refine_iter` parameter

### Debug Mode

Run with debug logging:
```bash
python pose_tracking_api.py --debug
```

---

# Mesh Generation API

A REST API service for generating 3D meshes (PLY/STL) from RGB images using text prompts and/or bounding boxes.

## Features

- **Multiple prompting modes**:
  - Text description only
  - Bounding box only
  - Combined text + bounding box (recommended for best results)
- **Automatic 3D reconstruction**: Uses SAM3 for segmentation and SAM3D for 3D mesh generation
- **Multiple output formats**: PLY (Gaussian splat) or STL (surface mesh)
- **Base64 I/O**: Send and receive files as base64 for easy integration
- **Reproducible results**: Configurable random seed

## Architecture

```
RGB Image + Text/BBox → SAM3 (Segmentation) → Mask
                         ↓
Mask + RGB Image → SAM3D (3D Reconstruction) → PLY/STL Mesh
```

## Quick Start

### 1. Start the API Server

```bash
# Using the startup script (recommended)
./perception/webapi/start_mesh_api.sh

# Or directly with Python
python perception/webapi/mesh_generation_api.py --port 5001
```

The server will load SAM3 and SAM3D models (this takes ~30 seconds on first start).

### 2. Use the Client

```bash
# Example 1: Text + bounding box (best results)
python perception/webapi/mesh_client_example.py \
    --image ./test_case/images/shutterstock_stylish_kidsroom_1640806567/image.png \
    --text "toy bear" \
    --bbox_xywh 100,150,200,300 \
    --output toy_bear.ply \
    --save_mask toy_bear_mask.png

# Example 2: Text only
python perception/webapi/mesh_client_example.py \
    --image ./test_case/images/shutterstock_stylish_kidsroom_1640806567/image.png \
    --text "toy bear" \
    --output toy_bear.ply

# Example 3: Bounding box only
python perception/webapi/mesh_client_example.py \
    --image ./test_case/images/shutterstock_stylish_kidsroom_1640806567/image.png \
    --bbox_xywh 100,150,200,300 \
    --output toy_bear.ply

# Example 4: Generate STL instead of PLY
python perception/webapi/mesh_client_example.py \
    --image ./test_case/images/shutterstock_stylish_kidsroom_1640806567/image.png \
    --text "toy bear" \
    --bbox_xywh 100,150,200,300 \
    --output toy_bear.stl \
    --format stl
```

## API Endpoints

### Health Check
```http
GET /health

Response:
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cuda:0",
  "trimesh_available": true,
  "open3d_available": true
}
```

### Generate Mesh
```http
POST /api/generate_mesh
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "text": "toy bear",
  "bbox_xywh": [100, 150, 200, 300],
  "output_format": "ply",
  "mask_id": 0,
  "seed": 42,
  "return_mask": true,
  "return_file": true
}

Response:
{
  "status": "success",
  "mesh_info": {
    "format": "ply",
    "file_size": 12345678,
    "point_count": 50000
  },
  "mask_score": 0.95,
  "mask_base64": "...",
  "file_base64": "...",
  "file_name": "output.ply",
  "session_id": "uuid-string"
}
```

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | string | Yes | Base64 encoded RGB image |
| `text` | string | No* | Text description of object to segment |
| `bbox_xywh` | array[4] | No* | Bounding box [x, y, width, height] |
| `bbox_xyxy` | array[4] | No* | Alternative bbox format [x1, y1, x2, y2] |
| `output_format` | string | No | "ply" or "stl" (default: "ply") |
| `mask_id` | integer | No | Which mask to use if multiple found (default: 0) |
| `seed` | integer | No | Random seed for reproducibility (default: 42) |
| `return_mask` | boolean | No | Return the generated mask (default: false) |
| `return_file` | boolean | No | Return the mesh file as base64 (default: true) |

*At least one of `text`, `bbox_xywh`, or `bbox_xyxy` must be provided.

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "success" or "error" |
| `mesh_info` | object | Mesh file information |
| `mesh_info.format` | string | File format ("ply" or "stl") |
| `mesh_info.file_size` | integer | File size in bytes |
| `mesh_info.point_count` | integer | Number of points (PLY only) |
| `mesh_info.vertex_count` | integer | Number of vertices (STL only) |
| `mesh_info.face_count` | integer | Number of faces (STL only) |
| `mask_score` | float | Confidence score of the mask (0-1) |
| `mask_base64` | string | Base64 encoded mask PNG (if requested) |
| `file_base64` | string | Base64 encoded mesh file (if requested) |
| `file_name` | string | Generated file name |
| `session_id` | string | Unique session identifier |

## Python Client Example

```python
import requests
import base64
from PIL import Image

# Encode image
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Call API
response = requests.post("http://localhost:5001/api/generate_mesh", json={
    "image": image_b64,
    "text": "red coffee mug",
    "bbox_xywh": [100, 150, 200, 300],
    "output_format": "ply",
    "return_mask": True
})

result = response.json()

# Save mesh file
mesh_bytes = base64.b64decode(result["file_base64"])
with open("output.ply", "wb") as f:
    f.write(mesh_bytes)

# Save mask
if "mask_base64" in result:
    mask_bytes = base64.b64decode(result["mask_base64"])
    with open("mask.png", "wb") as f:
        f.write(mask_bytes)

print(f"Mesh saved! Score: {result['mask_score']:.3f}")
```

### Process Mesh (Scale & Convert)
```http
POST /api/process_mesh
Content-Type: application/json

{
  "mesh_file": "base64_encoded_ply_file",
  "reference_size_meters": 0.5,
  "reference_axis": "width",
  "return_file": true
}

Response:
{
  "status": "success",
  "mesh_info": {
    "format": "stl",
    "file_size": 1234567,
    "vertex_count": 10000,
    "triangle_count": 20000,
    "scale_factor": 0.012345,
    "original_dimensions": {
      "width": 40.5,
      "height": 30.2,
      "depth": 25.1
    },
    "scaled_dimensions_meters": {
      "width": 0.5,
      "height": 0.373,
      "depth": 0.310
    },
    "reference_axis": "width",
    "reference_size_meters": 0.5
  },
  "mesh_file_base64": "..."
}
```

**Process Mesh Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mesh_file` | string | Yes | Base64 encoded PLY mesh file |
| `reference_size_meters` | float | Yes | Known real-world size in meters |
| `reference_axis` | string | No | "width", "height", or "depth" (default: "width") |
| `return_file` | boolean | No | Return STL file as base64 (default: true) |

**Example using client:**
```bash
python perception/webapi/process_mesh_client_example.py \
    --mesh toy_bear.ply \
    --reference-size 0.5 \
    --reference-axis width \
    --output toy_bear_scaled.stl
```

## Installation

### Prerequisites

1. SAM3D checkpoints downloaded:
```bash
# See external/sam-3d-objects/README.md for download instructions
```

2. Conda environment activated:
```bash
conda activate sam3d-objects
```

### Install Dependencies

```bash
pip install -r perception/webapi/requirements_mesh.txt
```

## Configuration

### Server Options

```bash
python perception/webapi/mesh_generation_api.py \
    --host 0.0.0.0 \
    --port 5001 \
    --debug
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | 0.0.0.0 | Host to bind to |
| `--port` | 5001 | Port to bind to |
| `--debug` | False | Enable debug mode |

### Using the Startup Script

```bash
# Default settings (port 5001)
./perception/webapi/start_mesh_api.sh

# Custom port
./perception/webapi/start_mesh_api.sh --port 8080

# Debug mode
./perception/webapi/start_mesh_api.sh --debug

# Custom host and port
./perception/webapi/start_mesh_api.sh --host localhost --port 8080
```

## Performance

- **Model Loading**: ~30 seconds on first startup
- **Inference Time**: ~30-60 seconds per mesh generation
  - SAM3 (segmentation): ~5-10 seconds
  - SAM3D (3D reconstruction): ~20-40 seconds
  - STL conversion: ~2-5 seconds (if requested)
- **GPU Memory**: ~8GB VRAM required
- **Output Size**: 
  - PLY: 10-50 MB typical
  - STL: 5-20 MB typical

## Tips for Best Results

1. **Use combined prompts**: Text + bounding box gives best results
2. **Be specific with text**: "red coffee mug" better than "mug"
3. **Tight bounding boxes**: Box should closely fit the object
4. **Good lighting**: Clear, well-lit images work best
5. **Single objects**: Works best with one clear object in view
6. **Adjust mask_id**: If first mask isn't correct, try mask_id=1, 2, etc.

## Troubleshooting

### Common Issues

1. **"No masks generated"**: 
   - Try adjusting text prompt or bounding box
   - Check if object is clearly visible in image
   
2. **"API not responding"**: 
   - Check if server is running: `curl http://localhost:5001/health`
   - Check firewall settings
   
3. **"SAM3D config not found"**: 
   - Download SAM3D checkpoints first
   - Verify path: `external/sam-3d-objects/checkpoints/hf/pipeline.yaml`
   
4. **GPU Out of Memory**: 
   - Reduce input image resolution
   - Close other GPU processes
   - Use smaller model if available

5. **Mask score too low**: 
   - Refine text description
   - Provide or adjust bounding box
   - Try different mask_id values

### Debug Mode

Enable detailed logging:
```bash
python perception/webapi/mesh_generation_api.py --debug
```

## Viewing Generated Meshes

### Desktop Applications
- **CloudCompare**: Best for PLY point clouds
- **MeshLab**: Good for both PLY and STL
- **Blender**: Advanced 3D editing

### Online Viewers
- https://3dviewer.net/
- https://viewstl.com/
- https://imagetostl.com/view

## Architecture Details

### Pipeline Overview

1. **Input Processing**:
   - Decode base64 image
   - Parse text/bounding box prompts

2. **Mask Generation (SAM3)**:
   - Load SAM3 model
   - Process prompts (text, bbox, or both)
   - Run segmentation inference
   - Select best mask by confidence score

3. **3D Reconstruction (SAM3D)**:
   - Combine RGB image + mask
   - Run SAM3D inference
   - Generate Gaussian splat point cloud
   - Export as PLY file

4. **Format Conversion (Optional)**:
   - Load PLY point cloud
   - Reconstruct surface mesh
   - Export as STL file

5. **Response**:
   - Encode files as base64
   - Return metadata and files

### Technology Stack

- **Flask**: Web framework
- **SAM3**: Segment Anything Model 3 (Meta)
- **SAM3D**: 3D object reconstruction from single image
- **PyTorch**: Deep learning backend
- **Transformers**: SAM3 model loading
- **Trimesh**: Mesh processing and conversion
- **Pillow**: Image processing

## License

This API wrapper follows the licenses of:
- SAM3 (Meta)
- SAM3D
- FoundationPose++

## Citation

If you use this API, please cite:
- SAM3: [Meta AI Research]
- SAM3D: [SAM3D paper]

---

# License

This API wrapper follows the same license as FoundationPose++.

## References

- FoundationPose: [Original paper]
- Cutie: [Cutie paper]
- SAM3: [Meta AI]
- SAM3D: [SAM3D repository]
