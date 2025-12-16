# 6D Object Pose Tracking REST API

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

## License

This API wrapper follows the same license as FoundationPose++.

## Citation

If you use this API, please cite:
- FoundationPose: [Original paper]
- Cutie: [Cutie paper]
